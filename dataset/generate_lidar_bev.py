import numpy as np
import imageio
import os
import glob
import json
import gzip
from pathlib import Path
import sys
sys.path.append(Path(__file__).resolve().parent.parent)
from time import sleep
import laspy

BOUNDARY = {
    "minX": -32,
    "maxX": 32,
    "minY": -32,
    "maxY": 32,
    "minZ": -2.73,
    "maxZ": 1.27,
    "center_boundary_x": 2.4,
    "center_boundary_y": 1,
}

def removePoints(PointCloud, BoundaryCond):
    # Boundary condition
    minX = BoundaryCond['minX']
    maxX = BoundaryCond['maxX']
    minY = BoundaryCond['minY']
    maxY = BoundaryCond['maxY']
    minZ = BoundaryCond['minZ']
    maxZ = BoundaryCond['maxZ']
    center_boundary_x = BoundaryCond['center_boundary_x']
    center_boundary_y = BoundaryCond['center_boundary_y']
    # Remove the point out of range x,y,z
    valid_mask = (PointCloud[:, 0] >= minX) & \
                (PointCloud[:, 0] <= maxX) & \
                (PointCloud[:, 1] >= minY) & \
                (PointCloud[:, 1] <= maxY) & \
                (PointCloud[:, 2] >= minZ) & \
                (PointCloud[:, 2] <= maxZ)

    center_mask = (np.abs(PointCloud[:, 0]) < center_boundary_x) & (np.abs(PointCloud[:, 1]) < center_boundary_y)
    PointCloud = PointCloud[valid_mask & (~center_mask)]
    # PointCloud = PointCloud[valid_mask]

    PointCloud[:, 2] = PointCloud[:, 2] - minZ
    return PointCloud

def makeBVFeature(PointCloud_, BoundaryCond, img_height, img_width, Discretization):
    Height = img_height + 1
    Width = img_width + 1

    # Discretize Feature Map
    PointCloud = np.copy(PointCloud_)
    PointCloud[:, 0] = np.int_(np.floor(PointCloud[:, 0] / Discretization))
    PointCloud[:, 1] = np.int_(np.floor(PointCloud[:, 1] / Discretization))

    # sort-3times
    indices = np.lexsort((-PointCloud[:, 2], PointCloud[:, 1], PointCloud[:, 0]))
    PointCloud = PointCloud[indices]

    # Height Map
    heightMap = np.zeros((Height, Width))

    # because of the rounding of points, there are many identical points
    _, indices = np.unique(PointCloud[:, 0:2], axis=0, return_index=True)
    PointCloud_frac = PointCloud[indices]

    # Some important problem is image coordinate is (y,x), not (x,y)
    max_height = float(np.abs(BoundaryCond['maxZ'] - BoundaryCond['minZ']))
    heightMap[np.int_(PointCloud_frac[:, 0]), np.int_(PointCloud_frac[:, 1])] = PointCloud_frac[:, 2] / max_height

    # Intensity Map & DensityMap
    intensityMap = np.zeros((Height, Width))
    densityMap = np.zeros((Height, Width))

    # 'counts': The number of times each of the unique values comes up in the original array
    _, indices, counts = np.unique(PointCloud[:, 0:2], axis=0, return_index=True, return_counts=True)
    PointCloud_top = PointCloud[indices]
    normalizedCounts = np.minimum(1.0, np.log(counts + 1) / np.log(64))
    intensityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = PointCloud_top[:, 3]
    densityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = normalizedCounts

    RGB_Map = np.zeros((Height-1, Width-1, 3))
    RGB_Map[:, :, 0] = densityMap[0:img_height, 0:img_width]  # r_map
    RGB_Map[:, :, 1] = heightMap[0:img_height, 0:img_width]  # g_map
    RGB_Map[:, :, 2] = intensityMap[0:img_height, 0:img_width]  # / 255  # b_map

    # normalize RGB_Map from [0, 1] to [0, 255]
    RGB_Map = (255 * RGB_Map).astype(np.uint8)

    return RGB_Map

def draw_centerpoint_bev(bev_image, centerpoints):
    centerpoints[:, [0, 1]] = centerpoints[:, [1, 0]]
    centerpoints[:, 1] = centerpoints[:, 1] * -1
    centerpoints[:, 0] = centerpoints[:, 0] + BOUNDARY["maxX"]
    centerpoints[:, 1] = centerpoints[:, 1] + BOUNDARY["maxY"]
    discretization = (BOUNDARY["maxX"] - BOUNDARY["minX"]) / 448
    
    centerpoints[:, 0] = centerpoints[:, 0] / discretization
    centerpoints[:, 1] = centerpoints[:, 1] / discretization
    
    bev_image[np.int_(centerpoints[:, 1]), np.int_(centerpoints[:, 0])] = 255
    imageio.imwrite('bev_centerpoints.png', bev_image)

def generate_lidar_bev_images(lidar_pc, saving_name=None, img_height=448, img_width=448):
    # Coordinate transform
    lidar_pc[:, 0] = lidar_pc[:, 0] * -1
    
    if lidar_pc.shape[-1] == 3:
        lidar_pc = np.concatenate([lidar_pc, np.ones((*lidar_pc.shape[:-1], 1))], axis=-1)
    lidar_pc = removePoints(lidar_pc, BOUNDARY)

    lidar_pc_filtered = np.copy(lidar_pc)
    lidar_pc_filtered[:, 0] = lidar_pc_filtered[:, 0] + BOUNDARY["maxX"]
    lidar_pc_filtered[:, 1] = lidar_pc_filtered[:, 1] + BOUNDARY["maxY"]

    # create Bird's Eye View
    discretization = (BOUNDARY["maxX"] - BOUNDARY["minX"]) / img_height
    lidar_bev = makeBVFeature(lidar_pc_filtered, BOUNDARY, img_height, img_width, discretization)

    lidar_bev[:, :, 2] = 0.0
    if saving_name is not None:
        imageio.imwrite(saving_name, lidar_bev)
    return lidar_bev

def get_combined_lidar(lidar_path_front, lidar_path_back):
    lidar_front = np.load(lidar_path_front, allow_pickle=True)
    lidar_back = np.load(lidar_path_back, allow_pickle=True)
    lidar_full = np.concatenate([lidar_front, lidar_back], axis=0)
    return lidar_full

def get_bev_bboxs_pdm_lite(bbox_data, distance_threshold=40):
    '''
    Extract bounding boxes from PDM Lite dataset format
    PDM Lite boxes are already in ego vehicle coordinate system
    
    Args:
        bbox_data: List of bounding box dictionaries or path to json.gz file
        distance_threshold: Maximum distance to include objects
    
    Returns:
        obj_dict: Dictionary of object types and their positions
    '''
    obj_dict = {}
    
    # Load bbox data if it's a file path
    if isinstance(bbox_data, str):
        bbox_data = load_measurement_data(bbox_data)
    
    if not isinstance(bbox_data, list):
        return obj_dict
    
    # Separate objects by class
    vehicles = []
    pedestrians = []
    traffic_lights = []
    stop_signs = []
    
    for obj in bbox_data:
        if 'class' not in obj or 'position' not in obj:
            continue
        
        obj_class = obj['class'].lower()
        position = obj['position']  # [x, y, z]
        
        # Skip ego vehicle
        if obj_class == 'ego_car':
            continue
        
        # Filter by distance
        distance = np.sqrt(position[0]**2 + position[1]**2)
        if distance > distance_threshold:
            continue
        
        # Categorize objects
        if 'vehicle' in obj_class or 'car' in obj_class or 'truck' in obj_class or 'bicycle' in obj_class:
            vehicles.append([position[0], position[1]])
        elif 'pedestrian' in obj_class or 'walker' in obj_class:
            pedestrians.append([position[0], position[1]])
        elif 'traffic' in obj_class:
            traffic_lights.append([position[0], position[1]])
        elif 'stop' in obj_class:
            stop_signs.append([position[0], position[1]])
    
    # Convert to numpy arrays
    if vehicles:
        obj_dict['vehicle'] = np.array(vehicles)
    if pedestrians:
        obj_dict['pedestrian'] = np.array(pedestrians)
    if traffic_lights:
        obj_dict['traffic_light'] = np.array(traffic_lights)
    if stop_signs:
        obj_dict['stop_sign'] = np.array(stop_signs)
    
    return obj_dict

def generate_3d_data(bbox_dict, scene_name, bev_dir, data_index):
    bev_image = imageio.imread(bev_dir)
    bev_points = []
    for key, value in bbox_dict.items():
        if value.shape[0] > 0:
            bev_points.append(value)
    
    if len(bev_points) == 0:
        return []
    
    bev_points = np.concatenate(bev_points, axis=0)
    draw_centerpoint_bev(bev_image, bev_points)
    data_list = []
    data_index = int(data_index)
    
    # Build image list (handle cases where we don't have enough history)
    image_list = []
    for id in range(max(0, data_index-3), data_index+1):
        img_path = os.path.join('data/carla/DATASET', scene_name, 'rgb_full', '%04d.jpg' % id)
        if os.path.exists(img_path):
            image_list.append(img_path)
    
    # Add BEV image
    image_list.append(bev_dir)
    
    question = ("<image>\nHere are four images of the first view and a lidar bird-eye view image. "
                "The first three images are history images. "
                "Cars and cyclists are equally regarded as vehicles. "
                "Please give the bird-eye view coordinate of a vehicle or traffic sign you see.")

    for key, value in bbox_dict.items():
        if value.shape[0] == 0:
            continue
            
        index = np.random.choice(value.shape[0], size=1, replace=False)
        coords = value[index].round(2)
        answer = f"There is a {key} at [{coords[0, 0]}, {coords[0, 1]}]."
        data_dict = {
                    "image_id": scene_name,
                    'question_id': len(data_list),
                    "image": image_list,
                    "conversations": [
                        {
                            "from": "human",
                            "value": question
                        },
                        {
                            "from": "gpt",
                            "value": answer
                        }
                    ]
                }
        
        data_list.append(data_dict)

    return data_list

def get_lidar_pts(lidar_path):
    # load .laz file
    las = laspy.read(lidar_path)
    points = np.vstack((las.x, las.y, las.z)).transpose()
    intensities = las.intensity
    points = np.concatenate([points, intensities[:, None]], axis=-1)
    return points

def load_measurement_data(measurement_file):
    """Load measurement data from json or json.gz file"""
    try:
        if measurement_file.endswith('.gz'):
            with gzip.open(measurement_file, 'rt') as f:
                return json.load(f)
        else:
            with open(measurement_file, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading measurement file {measurement_file}: {e}")
        return None

def attach_debugger():
    import debugpy
    debugpy.listen(5678)
    print("Waiting for debugger!")
    debugpy.wait_for_client()
    print("Attached!")

def process_all_scenes_pdm_lite(dataset_root, output_dir, process_all=False, scenario_filter=None, save_in_place=False):
    """
    Process all scenes in PDM Lite dataset to generate BEV images
    
    Args:
        dataset_root: Path to the PDM Lite dataset root (/home/wang/dataset/data)
        output_dir: Path to the output directory for BEV images (only used if save_in_place=False)
        process_all: If False, only process the first scenario (for debugging)
        scenario_filter: List of scenario names to process (e.g., ['Accident', 'ControlLoss'])
        save_in_place: If True, save BEV images in the same directory structure as lidar files
    """
    # Get all scenario directories
    if scenario_filter:
        scenarios = [os.path.join(dataset_root, s) for s in scenario_filter if os.path.isdir(os.path.join(dataset_root, s))]
    else:
        scenarios = [d for d in glob.glob(os.path.join(dataset_root, '*')) if os.path.isdir(d)]
    
    total_scenarios = len(scenarios)
    processed_scenarios = 0
    total_scenes = 0
    total_frames = 0
    
    print(f"Found {total_scenarios} scenarios to process")
    if save_in_place:
        print(f"BEV images will be saved in original dataset structure (lidar_bev directory)")
    else:
        print(f"BEV images will be saved to: {output_dir}")
    
    for scenario_idx, scenario_path in enumerate(scenarios):
        scenario_name = os.path.basename(scenario_path)
        print(f"\n{'='*70}")
        print(f"[{scenario_idx+1}/{total_scenarios}] Processing scenario: {scenario_name}")
        print(f"{'='*70}")
        
        # Get all scenes in this scenario
        scenes = [d for d in glob.glob(os.path.join(scenario_path, '*')) if os.path.isdir(d)]
        scenario_frames = 0
        
        for scene_idx, scene_path in enumerate(scenes):
            scene_name = os.path.basename(scene_path)
            print(f"\n  [{scene_idx+1}/{len(scenes)}] Processing scene: {scene_name}")
            
            # Check if scene has required directories
            lidar_dir = os.path.join(scene_path, 'lidar')
            measurements_dir = os.path.join(scene_path, 'measurements')
            boxes_dir = os.path.join(scene_path, 'boxes')
            
            if not os.path.exists(lidar_dir):
                print(f"    ⚠ Warning: No lidar directory found, skipping scene")
                continue
            
            # Get file lists
            lidar_files = sorted(glob.glob(os.path.join(lidar_dir, '*.laz')))
            measurement_files = sorted(glob.glob(os.path.join(measurements_dir, '*.json*'))) if os.path.exists(measurements_dir) else []
            bbox_files = sorted(glob.glob(os.path.join(boxes_dir, '*.json*'))) if os.path.exists(boxes_dir) else []
            
            if not lidar_files:
                print(f"    ⚠ Warning: No lidar files found, skipping scene")
                continue
            
            # Create output directories
            if save_in_place:
                # Save in the same directory structure as lidar files
                bev_image_dir = os.path.join(scene_path, 'lidar_bev')
            else:
                # Save in separate output directory
                output_scene_dir = os.path.join(output_dir, scenario_name, scene_name)
                bev_image_dir = os.path.join(output_scene_dir, 'bev_images')
            
            # Check if BEV images already exist and are complete
            if os.path.exists(bev_image_dir):
                existing_bev_files = sorted(glob.glob(os.path.join(bev_image_dir, '*.png')))
                if len(existing_bev_files) == len(lidar_files):
                    # Check if all corresponding BEV images exist
                    all_exist = True
                    for lidar_file in lidar_files:
                        frame_id = os.path.basename(lidar_file).split('.')[0]
                        bev_file = os.path.join(bev_image_dir, f'{frame_id}.png')
                        if not os.path.exists(bev_file):
                            all_exist = False
                            break
                    
                    if all_exist:
                        print(f"    ✓ Scene already processed ({len(existing_bev_files)} BEV images), skipping...")
                        continue
                    else:
                        print(f"    ⚠ Incomplete BEV sequence found ({len(existing_bev_files)}/{len(lidar_files)}), reprocessing...")
                else:
                    print(f"    ⚠ Incomplete BEV sequence found ({len(existing_bev_files)}/{len(lidar_files)}), reprocessing...")
            
            os.makedirs(bev_image_dir, exist_ok=True)
            
            print(f"    Found {len(lidar_files)} lidar files, {len(measurement_files)} measurement files, {len(bbox_files)} bbox files")
            
            # Process each frame
            scene_frame_count = 0
            for i in range(len(lidar_files)):
                try:
                    lidar_file = lidar_files[i]
                    
                    # Progress indicator
                    if i % 20 == 0 or i == len(lidar_files) - 1:
                        print(f"    Processing frame {i+1}/{len(lidar_files)}: {os.path.basename(lidar_file)}")
                    
                    # Load lidar data
                    lidar_points = get_lidar_pts(lidar_file)
                    
                    # Generate BEV image
                    frame_id = os.path.basename(lidar_file).split('.')[0]
                    saving_bev_image = os.path.join(bev_image_dir, f'{frame_id}.png')
                    
                    # Generate and save BEV image
                    bev_image = generate_lidar_bev_images(
                        lidar_points, 
                        saving_bev_image, 
                        img_height=448,  # Standard size for PDM Lite
                        img_width=448
                    )
                    
                    # Process bounding boxes if available
                    if i < len(bbox_files):
                        bbox_file = bbox_files[i]
                        
                        try:
                            # Get bounding boxes in BEV coordinates (PDM Lite format)
                            bbox_dict = get_bev_bboxs_pdm_lite(bbox_file)
                            
                            if bbox_dict and not save_in_place:
                                # Only save bbox data to separate output dir if not saving in place
                                bbox_output_dir = os.path.join(output_dir, scenario_name, scene_name)
                                os.makedirs(bbox_output_dir, exist_ok=True)
                                bbox_output_file = os.path.join(bbox_output_dir, f'bbox_data_{frame_id}.json')
                                with open(bbox_output_file, 'w') as f:
                                    # Convert numpy arrays to lists for JSON serialization
                                    bbox_dict_serializable = {
                                        k: v.tolist() if isinstance(v, np.ndarray) else v 
                                        for k, v in bbox_dict.items()
                                    }
                                    json.dump(bbox_dict_serializable, f, indent=2)
                        except Exception as e:
                            print(f"      Warning: Error processing bboxes for frame {i}: {str(e)}")
                    
                    scene_frame_count += 1
                    scenario_frames += 1
                    total_frames += 1
                    
                except Exception as e:
                    print(f"      Error processing frame {i}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            total_scenes += 1
            print(f"    ✓ Completed scene: {scene_frame_count} frames processed")
            
            # Break after first scene if not processing all
            if not process_all and scene_idx >= 0:
                break
        
        processed_scenarios += 1
        print(f"\n  ✓ Completed scenario {scenario_name}: {scenario_frames} frames processed")
        
        # Break after first scenario if not processing all
        if not process_all:
            break
    
    print(f"\n{'='*70}")
    print(f"=== Processing Complete ===")
    print(f"{'='*70}")
    print(f"Processed scenarios: {processed_scenarios}/{total_scenarios}")
    print(f"Total scenes: {total_scenes}")
    print(f"Total frames: {total_frames}")
    if save_in_place:
        print(f"BEV images saved in original dataset structure (lidar_bev directories)")
    else:
        print(f"Output directory: {output_dir}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate LiDAR BEV images from PDM Lite dataset')
    parser.add_argument('--input_dir', type=str, 
                       default='/home/wang/dataset/data',
                       help='Input directory containing the PDM Lite dataset')
    parser.add_argument('--output_dir', type=str, 
                       default='/home/wang/projects/diffusion_policy_z/data/pdm_lite_bev',
                       help='Output directory for BEV images (only used if --save_in_place is not set)')
    parser.add_argument('--save_in_place', action='store_true', default=True,
                       help='Save BEV images in the original dataset structure (creates lidar_bev folder alongside lidar folder)')
    parser.add_argument('--process_all', action='store_true', default=True,
                       help='Process all scenarios and scenes (default: only first scenario/scene for testing)')
    parser.add_argument('--scenarios', type=str, nargs='+',
                       help='Specific scenario names to process (e.g., Accident ControlLoss)')
    parser.add_argument('--debug', action='store_true', 
                       help='Enable debugger')
    parser.add_argument('--img_size', type=int, default=448,
                       help='BEV image size (height and width)')
    
    args = parser.parse_args()
    
    # Enable debugger if requested
    if args.debug:
        attach_debugger()
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist!")
        sys.exit(1)
    
    # Create output directory if it doesn't exist and not saving in place
    if not args.save_in_place:
        os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"PDM Lite BEV Image Generation")
    print(f"{'='*70}")
    print(f"Input directory: {args.input_dir}")
    if args.save_in_place:
        print(f"Save mode: In-place (lidar_bev folders in original structure)")
    else:
        print(f"Output directory: {args.output_dir}")
    print(f"Process all: {args.process_all}")
    print(f"BEV image size: {args.img_size}x{args.img_size}")
    if args.scenarios:
        print(f"Filtering scenarios: {args.scenarios}")
    print(f"{'='*70}\n")
    
    # Process the data
    process_all_scenes_pdm_lite(
        dataset_root=args.input_dir,
        output_dir=args.output_dir,
        process_all=args.process_all,
        scenario_filter=args.scenarios,
        save_in_place=args.save_in_place
    )
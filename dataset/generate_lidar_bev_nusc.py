#!/usr/bin/env python3
import numpy as np
import imageio
import os
import glob
from pathlib import Path
import sys
sys.path.append(Path(__file__).resolve().parent.parent)

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
    RGB_Map[:, :, 2] = intensityMap[0:img_height, 0:img_width]  # b_map

    # normalize RGB_Map from [0, 1] to [0, 255]
    RGB_Map = (255 * RGB_Map).astype(np.uint8)

    return RGB_Map

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

def get_lidar_pts(lidar_path):
    """Load lidar points from nuScenes .pcd.bin file."""
    points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)
    # Extract x, y, z, intensity (ignore ring index)
    return points[:, :4]

def process_nuscenes_dataset(dataset_root, output_dir=None, process_all=False, save_in_place=True, img_size=448):
    """Process nuScenes dataset to generate BEV images from LIDAR data."""
    # Find LIDAR_TOP directory
    lidar_dir = os.path.join(dataset_root, 'samples', 'LIDAR_TOP')
    
    if not os.path.exists(lidar_dir):
        print(f"Error: LIDAR_TOP directory not found at {lidar_dir}")
        return
    
    # Get all lidar files
    lidar_files = sorted(glob.glob(os.path.join(lidar_dir, '*.pcd.bin')))
    
    if not lidar_files:
        print(f"Error: No .pcd.bin files found in {lidar_dir}")
        return
    
    # Create output directory
    if save_in_place:
        bev_image_dir = os.path.join(dataset_root, 'samples', 'LIDAR_TOP_BEV')
    else:
        if output_dir is None:
            output_dir = os.path.join(dataset_root, 'lidar_bev_output')
        bev_image_dir = output_dir
    
    os.makedirs(bev_image_dir, exist_ok=True)
    
    # Limit files if not processing all
    if not process_all:
        print(f"Processing first 10 frames only (use --process_all to process all {len(lidar_files)} frames)")
        lidar_files = lidar_files[:10]
    
    print(f"\n{'='*70}")
    print(f"nuScenes BEV Image Generation")
    print(f"{'='*70}")
    print(f"LIDAR directory: {lidar_dir}")
    print(f"Output directory: {bev_image_dir}")
    print(f"Total files to process: {len(lidar_files)}")
    print(f"BEV image size: {img_size}x{img_size}")
    print(f"{'='*70}\n")
    
    # Check for existing files
    existing_bev_files = glob.glob(os.path.join(bev_image_dir, '*.png'))
    if len(existing_bev_files) == len(lidar_files):
        print(f"All {len(existing_bev_files)} BEV images already exist. Skipping...")
        return
    elif len(existing_bev_files) > 0:
        print(f"Found {len(existing_bev_files)}/{len(lidar_files)} existing BEV images. Resuming...")
    
    # Process each frame
    processed_count = 0
    skipped_count = 0
    
    for i, lidar_file in enumerate(lidar_files):
        try:
            frame_name = os.path.basename(lidar_file).replace('.pcd.bin', '')
            saving_bev_image = os.path.join(bev_image_dir, f'{frame_name}.png')
            
            if os.path.exists(saving_bev_image):
                skipped_count += 1
                continue
            
            if i % 10 == 0 or i == len(lidar_files) - 1:
                print(f"\rProcessing frame {i+1}/{len(lidar_files)} (processed: {processed_count}, skipped: {skipped_count})...", end="")
            
            lidar_points = get_lidar_pts(lidar_file)
            generate_lidar_bev_images(lidar_points, saving_bev_image, img_height=img_size, img_width=img_size)
            processed_count += 1
            
        except Exception as e:
            print(f"\nError processing frame {os.path.basename(lidar_file)}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n\n{'='*70}")
    print(f"=== Processing Complete ===")
    print(f"{'='*70}")
    print(f"Total frames processed: {processed_count}")
    print(f"Total frames skipped: {skipped_count}")
    print(f"Output directory: {bev_image_dir}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate LiDAR BEV images from nuScenes dataset')
    parser.add_argument('--input_dir', type=str, default='/home/wang/Dataset/v1.0-mini',
                       help='Input directory containing the nuScenes dataset')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for BEV images (only used if --save_in_place is not set)')
    parser.add_argument('--save_in_place', action='store_true', default=True,
                       help='Save BEV images in the original dataset structure')
    parser.add_argument('--process_all', action='store_true', default=True,
                       help='Process all frames (default: only first 10 frames for testing)')
    parser.add_argument('--img_size', type=int, default=448,
                       help='BEV image size (height and width)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist!")
        sys.exit(1)
    
    process_nuscenes_dataset(
        dataset_root=args.input_dir,
        output_dir=args.output_dir,
        process_all=args.process_all,
        save_in_place=args.save_in_place,
        img_size=args.img_size
    )

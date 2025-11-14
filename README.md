# Diffusion Policy Implementation

A PyTorch implementation of Diffusion Policy

## Dataset

### Extract data from Nuscence Dataset

Download the NuScenes dataset and CAN bus expansion, put CAN bus expansion in /path/to/nuscenes, create symbolic links.

```bash
cd ${MOT-DP}
mkdir data
ln -s path/to/nuscenes ./data/nuscenes
sh DiffusionDrive/scripts/create_data.sh
```
Pack the meta-information and labels of the dataset, and generate the required pkl files to data/infos. Note that we also generate map_annos in data_converter, with a roi_size of (30, 60) as default, if you want a different range, you can modify roi_size in tools/data_converter/nuscenes_converter.py.

Refer to https://github.com/swc-17/SparseDrive/blob/main/docs/quick_start.md
### Generate BEV Figure in Dataset

Add BEV image to the current Nuscence dataset:

```bash
cd dataset
python generate_lidar_bev_nusc.py  
```

### Generate BEV Feature in Dataset

Add BEV feature based on the generated BEV fugure:

```bash
cd dataset
python preprocess_bev_feature_nusc.py  
```


### Preprocess Dataset
Generate trainable sequences and divide:

```bash
python preprocess_nusc.py 
```
You can visualize some random samples:


```bash
python nusc_dataset.py 
```
### Train


```bash
cd ..
cd training
python train_nusc_bev.py
```






# Diffusion Policy Implementation

A PyTorch implementation of Diffusion Policy

## Dataset

### Generate BEV Figure in Dataset

Add BEV image to the current PDM-LITE dataset:

```bash
cd dataset
python generate_lidar_bev_pdm.py  # for pdm-lite
python generate_lidar_bev_b2d.py  # for bench2drive
```

### Generate BEV Feature in Dataset

Add BEV feature based on the generated BEV fugure:

```bash
cd dataset
python preprocess_bev_feature_pdm.py  # for pdm-lite
python preprocess_bev_feature_b2d.py  # for bench2drive
```


### Preprocess Dataset
For Bench2Drive, a smooth process is necessary before preprocess:
```bash
python smooth_trajectory_b2d.py
```

Generate trainable sequences and divide:

```bash
python preprocess_pdm_lite.py # for pdm-lite
python preprocess_b2d.py # for bench2drive
```


### Train


```bash
cd ..
cd training
python train_carla_bev_full.py
```

### Test


```bash
cd testing
python test_carla_bev_full.py
```




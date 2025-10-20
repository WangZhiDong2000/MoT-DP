# Diffusion Policy Implementation

A PyTorch implementation of Diffusion Policy

## Dataset

### Generate Dataset

Add BEV image to the current PDM-LITE dataset:

```bash
cd dataset
python generate_lidar_bev.py
```

### Generate BEV Feature

Add BEV feature to the current PDM-LITE dataset:

```bash
cd dataset
python preprocess_lidar_bev.py
```

### Smooth Global Trajectory

Deal with unstable recorded path data:

```bash
cd dataset
python smooth_trajectory.py
```

### Preprocess Dataset

Generate trainable sequences and divide:

```bash
python preprocess_b2d.py
```


### Get min-max range

Before training, get the range and revise training and testing script:

```bash
python compute_action_stats.py
```


### Train

```bash
cd ..
cd training
python train_carla_bev.py
```

### Test


```bash
cd testing
python test_carla_bev_full.py
```




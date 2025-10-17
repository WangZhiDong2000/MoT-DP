# Diffusion Policy Implementation

A PyTorch implementation of Diffusion Policy

## Dataset

### Generate Dataset

Add BEV image to the current PDM-LITE dataset:

```bash
cd dataset
python generate_lidar_bev.py
```


### Preprocess Dataset

Generate trainable sequences and divide:

```bash
python preprocess_pdm_lite.py
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




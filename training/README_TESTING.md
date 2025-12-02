# 多GPU测试脚本使用说明

## 文件说明

- `test_carla_bev_multi_gpu.py`: 多GPU测试的Python主脚本
- `test_carla_bev_multi_gpu.sh`: 多GPU测试的Bash启动脚本
- `test_carla_bev_single_gpu.sh`: 单GPU测试的Bash启动脚本

## 快速开始

### 多GPU测试（推荐）

使用4个GPU在测试集上评估最佳模型：

```bash
bash training/test_carla_bev_multi_gpu.sh checkpoints/carla_dit_best/best_model.pth 4 test
```

### 单GPU测试

使用GPU 0在验证集上评估：

```bash
bash training/test_carla_bev_single_gpu.sh checkpoints/carla_dit_best/best_model.pth 0 val
```

### Python直接调用

```bash
# 单GPU
python training/test_carla_bev_multi_gpu.py \
    --checkpoint checkpoints/carla_dit_best/best_model.pth \
    --test_split test

# 多GPU (使用torchrun)
torchrun --nproc_per_node=4 training/test_carla_bev_multi_gpu.py \
    --checkpoint checkpoints/carla_dit_best/best_model.pth \
    --test_split test
```

## 参数说明

### test_carla_bev_multi_gpu.sh

```bash
bash test_carla_bev_multi_gpu.sh [checkpoint_path] [num_gpus] [test_split]
```

- `checkpoint_path`: 模型权重文件路径（默认：checkpoints/carla_dit_best/best_model.pth）
- `num_gpus`: 使用的GPU数量（默认：4）
- `test_split`: 测试数据集分割，可选 `test` 或 `val`（默认：test）

### test_carla_bev_single_gpu.sh

```bash
bash test_carla_bev_single_gpu.sh [checkpoint_path] [gpu_id] [test_split]
```

- `checkpoint_path`: 模型权重文件路径
- `gpu_id`: GPU设备ID（默认：0）
- `test_split`: 测试数据集分割

### Python脚本参数

```bash
python training/test_carla_bev_multi_gpu.py --help
```

- `--config`: 配置文件路径（默认：config/pdm_server.yaml）
- `--checkpoint`: 模型checkpoint路径（必需）
- `--test_split`: 数据集分割，`test` 或 `val`（默认：test）

## 评估指标

脚本会计算并报告以下指标：

### 主要指标
- **Loss**: 模型的diffusion loss
- **L2_1s**: 1秒预测的L2误差
- **L2_2s**: 2秒预测的L2误差  
- **L2_3s**: 3秒预测的L2误差
- **L2_avg**: 平均L2误差

## 输出结果

测试完成后会：

1. **终端输出**: 显示所有评估指标的表格
2. **保存JSON文件**: 结果保存在 `results/test_results_[checkpoint_name]_[split].json`

JSON文件包含：
- checkpoint路径
- 测试数据集分割
- 配置文件路径
- 所有评估指标
- 使用的GPU数量

## 示例输出

```
================================================================================
Test Results
================================================================================

Metric               Value
----------------------------------------
Loss                  0.0234
L2_1s                 0.8234
L2_2s                 1.4567
L2_3s                 2.1234
L2_avg                1.4678
================================================================================

✓ Results saved to: results/test_results_best_model_test.json
```

## 常见问题

### 1. 找不到checkpoint文件

确保checkpoint路径正确，例如：
```bash
ls checkpoints/carla_dit_best/best_model.pth
```

### 2. CUDA out of memory

减少batch size或使用更少的GPU：
```bash
# 编辑 config/pdm_server.yaml
dataloader:
  batch_size: 16  # 减小batch size
```

### 3. 多GPU不工作

确保PyTorch支持分布式训练：
```python
python -c "import torch; print(torch.distributed.is_available())"
```

### 4. 使用不同的配置文件

```bash
python training/test_carla_bev_multi_gpu.py \
    --config config/custom_config.yaml \
    --checkpoint checkpoints/model.pth
```

## 性能优化

- **多GPU**: 使用更多GPU可以加快测试速度
- **num_workers**: 在config中调整DataLoader的worker数量
- **batch_size**: 根据GPU内存调整batch size

## 与训练脚本的对应关系

测试脚本使用与 `train_carla_bev.py` 相同的：
- 数据集加载逻辑
- 评估指标计算
- 分布式训练框架

确保配置文件与训练时一致以获得准确的比较。

# 多GPU并行训练 - 已完成 ✅

## 修改完成时间
2025年11月28日

## 当前系统配置
- **GPU**: 8 × NVIDIA A100-SXM4-80GB (每卡80GB显存)
- **CUDA**: 12.8
- **PyTorch**: 2.9.0+cu128
- **分布式后端**: NCCL ✅

## 完成的修改

### ✅ 核心文件修改
- [x] `train_carla_bev.py` - 已升级为支持多GPU分布式训练
  - 添加 `@record` 装饰器
  - 初始化分布式进程组
  - 启用CUDA性能优化（TF32, cuDNN benchmark）
  - 使用 `DistributedSampler`
  - 使用 `DistributedDataParallel`
  - Rank 0 专属日志和保存

### ✅ 新增工具脚本
- [x] `train_carla_bev_multi_gpu.sh` - 便捷启动脚本
- [x] `check_gpu.py` - GPU环境检测工具

### ✅ 文档
- [x] `QUICKSTART.md` - 快速开始指南
- [x] `MULTI_GPU_TRAINING.md` - 详细多GPU训练指南  
- [x] `CHANGES_SUMMARY.md` - 代码修改总结
- [x] `FILE_CHANGES.md` - 文件修改清单

## 快速开始

### 1. 检查环境（推荐第一步）
```bash
cd /root/z_projects/code/MoT-DP-1/training
python check_gpu.py
```

### 2. 启动训练

#### 选项A: 使用8个GPU（推荐）
```bash
bash train_carla_bev_multi_gpu.sh 8
```

#### 选项B: 使用4个GPU
```bash
bash train_carla_bev_multi_gpu.sh 4
```

#### 选项C: 使用2个GPU
```bash
bash train_carla_bev_multi_gpu.sh 2
```

#### 选项D: 单GPU训练（向后兼容）
```bash
python train_carla_bev.py --config_path /root/z_projects/code/MoT-DP-1/config/pdm_server.yaml
```

### 3. 使用自定义配置
```bash
bash train_carla_bev_multi_gpu.sh 8 /path/to/your/config.yaml
```

## 性能提升预期

在您的8×A100系统上：

| 配置 | 加速比 | 训练时间（相对） |
|------|--------|-----------------|
| 1 GPU | 1.0x | 100% |
| 2 GPUs | ~1.8x | 56% |
| 4 GPUs | ~3.5x | 29% |
| 8 GPUs | ~6.5x | 15% |

**注意**: 实际加速比取决于模型大小和batch size

## 关键优化

### 已启用的CUDA优化
```python
torch.backends.cuda.matmul.allow_tf32 = True       # TF32加速
torch.backends.cudnn.benchmark = True              # 自动选择最优算法
torch.backends.cudnn.deterministic = False         # 非确定性以获得速度
torch.backends.cudnn.allow_tf32 = True             # cuDNN TF32
```

### Batch Size建议
- 1 GPU: batch_size = 32
- 2 GPUs: batch_size = 32 (有效64)
- 4 GPUs: batch_size = 32 (有效128)
- 8 GPUs: batch_size = 16-32 (有效128-256)

**调整**: 在配置文件 `config/pdm_server.yaml` 中修改 `dataloader.batch_size`

### 学习率建议
使用多GPU时，考虑线性缩放学习率：
- 1 GPU: lr = 5e-5
- 2 GPUs: lr = 1e-4
- 4 GPUs: lr = 2e-4
- 8 GPUs: lr = 4e-4

**调整**: 在配置文件中修改 `optimizer.lr`

## 监控和调试

### 查看详细的分布式日志
```bash
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
bash train_carla_bev_multi_gpu.sh 8
```

### 监控GPU使用率
```bash
# 另开一个终端
watch -n 1 nvidia-smi
```

### 查看WandB日志
训练时会自动上传到WandB（如果配置了）

## 常见问题

### Q: 为什么只看到1个GPU在工作？
A: 确保使用 `torchrun` 或便捷脚本启动，而不是直接 `python train_carla_bev.py`

### Q: 出现 "CUDA out of memory" 错误？
A: 减小batch_size，例如从32改为16

### Q: 训练速度没有明显提升？
A: 
1. 检查 `num_workers` 是否足够（推荐8-16）
2. 检查数据加载是否成为瓶颈
3. 确认使用了 `DistributedSampler`

### Q: 如何恢复中断的训练？
A: 检查checkpoint目录，最新的模型会自动保存

## 文件结构

```
training/
├── train_carla_bev.py              # 主训练脚本（已升级）
├── train_carla_bev_multi_gpu.sh    # 便捷启动脚本
├── check_gpu.py                    # GPU检测工具
├── README_MULTI_GPU.md             # 本文件
├── QUICKSTART.md                   # 快速开始
├── MULTI_GPU_TRAINING.md           # 详细指南
├── CHANGES_SUMMARY.md              # 修改总结
└── FILE_CHANGES.md                 # 文件清单
```

## 下一步

1. **测试单GPU**: `python train_carla_bev.py --config_path config.yaml`
2. **测试2 GPUs**: `bash train_carla_bev_multi_gpu.sh 2`
3. **测试8 GPUs**: `bash train_carla_bev_multi_gpu.sh 8`
4. **优化超参数**: 调整batch_size和学习率
5. **长时间训练**: 使用nohup或tmux保持会话

## 相关文档

- 📘 [QUICKSTART.md](QUICKSTART.md) - 3分钟快速开始
- 📗 [MULTI_GPU_TRAINING.md](MULTI_GPU_TRAINING.md) - 完整指南
- 📙 [CHANGES_SUMMARY.md](CHANGES_SUMMARY.md) - 技术细节
- 📕 [FILE_CHANGES.md](FILE_CHANGES.md) - 修改清单

## 技术支持

参考 `train.py` 中的实现，或查看上述文档。

---

**状态**: ✅ 已完成并通过环境检测  
**系统**: 8×A100-80GB, CUDA 12.8, PyTorch 2.9.0  
**日期**: 2025-11-28

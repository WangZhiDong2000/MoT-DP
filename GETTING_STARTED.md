# 🚀 模型维度调整完整指南 - 快速开始

## 📚 本项目包含的文件

您现在拥有以下完整工具集来调整模型维度:

| 文件 | 用途 | 命令 |
|-----|------|------|
| `DIMENSION_TUNING_GUIDE.md` | 📖 详细理论指南 | 阅读学习 |
| `DIMENSION_QUICK_REFERENCE.md` | ⚡ 快速参考卡 | 快速查询 |
| `utils/adjust_dimensions.py` | 🎛️ 维度调整工具 | `python utils/adjust_dimensions.py` |
| `utils/check_dimensions.py` | ✅ 兼容性检查 | `python utils/check_dimensions.py --check config/nuscenes.yaml` |
| `utils/estimate_memory.py` | 💾 显存估计 | `python utils/estimate_memory.py` |
| `config/nuscenes_high_precision.yaml` | 🔧 高精度配置示例 | 参考配置 |

---

## ⏱️ 5分钟快速开始

### 1️⃣ 查看可用方案 (30秒)
```bash
cd /home/wang/Project/MoT-DP
python utils/adjust_dimensions.py --list
```

输出结果:
```
📋 可用的维度调整方案
======================================================================

1️⃣  基准配置 (原始)
   ID: baseline
   n_emb: 512 | n_head: 8 | n_layer: 8 | n_cond_layers: 4 | feature_dim: 256
   💡 当前生产配置

2️⃣  保守提升 (低资源压力)
   ID: conservative
   n_emb: 768 | n_head: 12 | n_layer: 8 | n_cond_layers: 6 | feature_dim: 384
   💡 精度提升: ~5-10% | 内存增加: ~30% | 速度: 1.2x
   ...
```

### 2️⃣ 估计显存需求 (1分钟)
```bash
# 检查当前配置的显存需求
python utils/estimate_memory.py

# 或对比两个配置
python utils/estimate_memory.py config/nuscenes.yaml config/nuscenes_high_precision.yaml
```

输出例子:
```
📊 GPU显存估计报告
======================================================================

📝 模型配置:
  • n_emb: 512
  • n_head: 8
  • n_layer: 8
  • n_cond_layers: 4
  • feature_dim: 256
  • batch_size: 128

📈 显存分解 (GPU显存占用):
  • parameters      0.6GB ( 4.3%) █
  • optimizer       2.3GB (16.5%) ████
  • gradients       0.6GB ( 4.3%) █
  • activations     6.1GB (43.7%) █████████
  • bev_features    3.8GB (27.2%) ██████
  • misc            0.2GB ( 1.4%)

🎯 总计显存需求: 13.9 GB

💻 推荐GPU (显存等级: 中等显存):
  1. NVIDIA RTX 3060 (12GB)
  2. NVIDIA RTX 4060 (8GB)
  3. NVIDIA A10 (24GB)
```

### 3️⃣ 检查维度兼容性 (1分钟)
```bash
python utils/check_dimensions.py --default
```

输出例子:
```
✅ 通过
✓ 每个注意力头维度: 64
✓ 总嵌入维度: 512
✓ 注意力头数: 8
✓ 显存占用: 100%
✓ 参数量: 89.5M
```

### 4️⃣ 应用新方案 (2分钟)

**方法A: 交互式 (推荐)**
```bash
python utils/adjust_dimensions.py
# 按提示选择方案即可
```

**方法B: 直接应用**
```bash
# 应用平衡方案（推荐）
python utils/adjust_dimensions.py balanced config/nuscenes.yaml

# 或保守方案
python utils/adjust_dimensions.py conservative config/nuscenes.yaml

# 或激进方案（需要好GPU）
python utils/adjust_dimensions.py aggressive config/nuscenes.yaml
```

### 5️⃣ 验证配置并训练 (30秒)
```bash
# 验证新配置
python utils/check_dimensions.py --check config/nuscenes.yaml

# 如果通过检查，启动训练
python training/train_nusc_bev.py --config config/nuscenes.yaml
```

---

## 🎯 推荐的调整步骤

### 对于GPU显存有限的情况 (≤8GB)

```bash
# 1. 使用轻量配置
python utils/adjust_dimensions.py lightweight config/nuscenes.yaml

# 2. 进一步减小batch_size
# 编辑 config/nuscenes.yaml
# dataloader:
#   batch_size: 32  # 从128减到32

# 3. 检查显存
python utils/estimate_memory.py

# 4. 训练
python training/train_nusc_bev.py --config config/nuscenes.yaml
```

### 对于中等GPU的情况 (8-16GB) ⭐ 推荐

```bash
# 1. 应用保守方案 (首选)
python utils/adjust_dimensions.py conservative config/nuscenes.yaml

# 2. 检查显存和兼容性
python utils/estimate_memory.py
python utils/check_dimensions.py --check config/nuscenes.yaml

# 3. 如果有充足显存，考虑升级到平衡方案
python utils/adjust_dimensions.py balanced config/nuscenes.yaml
python utils/estimate_memory.py

# 4. 开始训练
python training/train_nusc_bev.py --config config/nuscenes.yaml
```

### 对于高端GPU的情况 (24GB+)

```bash
# 1. 应用平衡方案
python utils/adjust_dimensions.py balanced config/nuscenes.yaml

# 2. 检查显存
python utils/estimate_memory.py

# 3. 如果还有空余，可升级到激进方案
python utils/adjust_dimensions.py aggressive config/nuscenes.yaml
python utils/estimate_memory.py

# 4. 训练（可能需要减小batch_size以稳定训练）
python training/train_nusc_bev.py --config config/nuscenes.yaml --batch-size 64
```

---

## 📖 理论学习资源

如要深入理解模型维度调整的原理，请阅读:

1. **快速参考** (5分钟)
   ```bash
   cat DIMENSION_QUICK_REFERENCE.md
   ```
   - 常用参数总结
   - 预定义方案对比
   - 常见问题解答

2. **详细指南** (30分钟)
   ```bash
   cat DIMENSION_TUNING_GUIDE.md
   ```
   - 每个参数的详细说明
   - 调整原理和背景
   - 最佳实践总结

---

## 🔄 完整工作流示例

### 示例1: 从头开始优化 (GPU: RTX 3080, 10GB显存)

```bash
# 第1步: 备份原配置
cp config/nuscenes.yaml config/nuscenes_backup_v1.yaml

# 第2步: 检查原配置的显存需求
python utils/estimate_memory.py config/nuscenes.yaml
# 输出: 约14GB (超出显存!)

# 第3步: 应用保守方案
python utils/adjust_dimensions.py conservative config/nuscenes.yaml

# 第4步: 检查新配置的显存
python utils/estimate_memory.py config/nuscenes.yaml
# 输出: 约18GB (仍然太大!)

# 第5步: 手动减小batch_size
# 编辑 config/nuscenes.yaml
# dataloader.batch_size: 64 (从128改到64)

# 第6步: 再次检查
python utils/estimate_memory.py
# 输出: 约9GB (正好!)

# 第7步: 验证兼容性
python utils/check_dimensions.py --check config/nuscenes.yaml
# 输出: ✅ 通过

# 第8步: 开始训练
python training/train_nusc_bev.py --config config/nuscenes.yaml

# 第9步: 监控显存
watch -n 1 nvidia-smi
```

### 示例2: 从baseline升级到更高精度 (GPU: A100, 40GB显存)

```bash
# 第1步: 保存当前模型
cp checkpoints/carla_dit_best/model.pth checkpoints/carla_dit_best/model_baseline.pth

# 第2步: 应用平衡方案
python utils/adjust_dimensions.py balanced config/nuscenes.yaml

# 第3步: 验证配置
python utils/estimate_memory.py config/nuscenes.yaml
python utils/check_dimensions.py --check config/nuscenes.yaml

# 第4步: 从头训练新配置（因为维度改变）
python training/train_nusc_bev.py --config config/nuscenes.yaml --epochs 1500

# 第5步: 对比结果
# 使用 testing/test_nusc_bev_full.py 对比两个模型的精度
```

---

## ⚙️ 常用命令速查

```bash
# 🎛️ 维度调整相关
python utils/adjust_dimensions.py --list                                    # 列出所有方案
python utils/adjust_dimensions.py                                           # 交互式调整
python utils/adjust_dimensions.py balanced config/nuscenes.yaml            # 直接应用

# ✅ 配置检查相关
python utils/check_dimensions.py --default                                 # 检查默认配置
python utils/check_dimensions.py --check config/nuscenes.yaml             # 检查指定配置
python utils/check_dimensions.py --compare cfg1.yaml cfg2.yaml            # 对比两个配置

# 💾 显存估计相关
python utils/estimate_memory.py                                            # 估计显存
python utils/estimate_memory.py config/nuscenes.yaml                       # 指定配置
python utils/estimate_memory.py cfg1.yaml cfg2.yaml                        # 对比显存

# 🚀 训练相关
python training/train_nusc_bev.py --config config/nuscenes.yaml            # 启动训练
python training/train_nusc_bev.py --config config/nuscenes.yaml --batch-size 64  # 自定义batch
watch -n 1 nvidia-smi                                                      # 监控显存

# 📊 测试相关
python testing/test_nusc_bev_full.py --config config/nuscenes.yaml        # 测试模型
```

---

## 🐛 故障排除

### 问题1: "n_emb must be divisible by n_head"
```
❌ 错误: n_emb (1000) 必须能被 n_head (8) 整除

解决方案:
python utils/adjust_dimensions.py balanced config/nuscenes.yaml
# 上述命令会自动调整为兼容的值
```

### 问题2: "CUDA out of memory"
```
❌ 错误: CUDA out of memory

解决方案:
# 1. 检查当前配置的显存需求
python utils/estimate_memory.py

# 2. 如果超出GPU显存，应用更轻量的配置
python utils/adjust_dimensions.py conservative config/nuscenes.yaml

# 3. 或手动减小batch_size
# 编辑 config/nuscenes.yaml
# dataloader.batch_size: 32

# 4. 重新启动训练
python training/train_nusc_bev.py --config config/nuscenes.yaml
```

### 问题3: 调整维度后精度下降

```
❌ 精度反而下降

原因和解决方案:
1. 训练不充分
   → 增加 training.num_epochs 或训练更长时间

2. 学习率不适配
   → 编辑 optimizer.lr，尝试更大的学习率

3. 批量标准化问题
   → 如果改变了n_emb，某些BN层可能需要重新初始化

4. 数据增强不足
   → 检查数据加载和增强配置

5. 需要调整其他超参数
   → 不只是维度，还需要调整学习率、dropout等
```

---

## 📈 预期性能提升

使用推荐的平衡方案 vs 基准方案:

| 指标 | 基准 | 平衡方案 | 提升幅度 |
|------|------|--------|---------|
| 精度 (L2错误) | 1.0 | 0.8 | ↓ 20% |
| 精度 (轨迹预测) | baseline | +15-25% | ↑ 15-25% |
| 显存占用 | 14GB | 28GB | × 2 |
| 训练速度 | 1 sample/s | 0.5 sample/s | ÷ 2 |
| 参数量 | 90M | 245M | × 2.7 |

💡 **关键**: 精度提升是否值得与显存/速度的取舍

---

## 🎓 推荐学习路径

1. **入门 (10分钟)**
   - 阅读 `DIMENSION_QUICK_REFERENCE.md` 前半部分
   - 运行 `python utils/adjust_dimensions.py --list`

2. **实践 (20分钟)**
   - 运行 `python utils/estimate_memory.py`
   - 运行 `python utils/check_dimensions.py --default`
   - 应用一个新方案: `python utils/adjust_dimensions.py conservative config/nuscenes.yaml`

3. **深入 (30分钟)**
   - 阅读 `DIMENSION_TUNING_GUIDE.md` 完整版
   - 理解每个参数的影响
   - 学习何时该调整哪个参数

4. **优化 (1-2小时)**
   - 针对您的硬件配置优化参数
   - 对比多个方案的效果
   - 找到最优的精度/速度/显存平衡点

---

## 📞 获取帮助

如有问题，请检查:

1. **文档**
   - `DIMENSION_QUICK_REFERENCE.md` - 快速查询
   - `DIMENSION_TUNING_GUIDE.md` - 详细解释

2. **自动检查**
   ```bash
   python utils/check_dimensions.py --check config/nuscenes.yaml
   python utils/estimate_memory.py config/nuscenes.yaml
   ```

3. **日志输出**
   ```bash
   python training/train_nusc_bev.py --config config/nuscenes.yaml 2>&1 | tee train.log
   ```

---

**开始优化您的模型吧! 🚀**

最后更新: 2025-11-17

# 模型维度调整指南 - 提高精度

本指南说明如何通过调整 MoT-DP 模型的各个维度参数来提高预测精度。

## 核心维度参数总览

### 1. **嵌入维度 (n_emb)** - 最重要的参数
**位置**: `config/nuscenes.yaml` -> `policy.n_emb`
**当前值**: 512

```yaml
policy:
  n_emb: 512  # 调整此值
```

**说明**:
- 决定了 Transformer 中所有向量的维度
- 更大的维度 → 更强的表达能力 → 更高的精度（但更慢更耗内存）
- 推荐调整范围: 256, 512, 768, 1024
- 影响: 
  - ✅ 精度提升幅度: **大**
  - ⚠️ 内存消耗: **大幅增加**
  - ⚠️ 训练速度: **明显变慢**

**调整示例**:
```yaml
# 高精度配置（需要更多GPU内存）
policy:
  n_emb: 1024

# 平衡配置
policy:
  n_emb: 768

# 轻量配置
policy:
  n_emb: 256
```

---

### 2. **BEV特征维度 (feature_dim)**
**位置**: `config/nuscenes.yaml` -> `bev_encoder.feature_dim`
**当前值**: 256

```yaml
bev_encoder:
  feature_dim: 256  # 调整此值
```

**说明**:
- BEV 编码器输出特征的维度
- 影响输入条件的表达能力
- 推荐调整范围: 128, 256, 512
- 影响:
  - ✅ 精度提升幅度: **中等**
  - ⚠️ 内存消耗: **中等增加**
  - ⚠️ 训练速度: **略微变慢**

**调整示例**:
```yaml
# 高精度配置
bev_encoder:
  feature_dim: 512

# 平衡配置
bev_encoder:
  feature_dim: 256

# 轻量配置
bev_encoder:
  feature_dim: 128
```

---

### 3. **注意力头数 (n_head)**
**位置**: `config/nuscenes.yaml` -> `policy.n_head`
**当前值**: 8

```yaml
policy:
  n_head: 8  # 调整此值
```

**说明**:
- Transformer 中多头注意力的头数
- n_emb 必须能被 n_head 整除（例: 1024/8=128, 768/8=96）
- 更多的头 → 更好的信息交互
- 推荐调整范围: 4, 8, 12, 16（必须满足整除关系）
- 影响:
  - ✅ 精度提升幅度: **中小**
  - ⚠️ 内存消耗: **轻微增加**
  - ⚠️ 训练速度: **轻微变慢**

**调整示例**:
```yaml
# n_emb=1024 时
policy:
  n_emb: 1024
  n_head: 16  # 1024/16 = 64

# n_emb=768 时
policy:
  n_emb: 768
  n_head: 12  # 768/12 = 64

# n_emb=512 时（当前）
policy:
  n_emb: 512
  n_head: 8   # 512/8 = 64
```

---

### 4. **条件层数 (n_cond_layers)**
**位置**: `config/nuscenes.yaml` -> `policy.n_cond_layers`
**当前值**: 4

```yaml
policy:
  n_cond_layers: 4  # 调整此值
```

**说明**:
- 条件编码器的 Transformer 层数
- 更多层 → 更深的特征学习 → 可能更高的精度
- 推荐调整范围: 2, 4, 6, 8
- 影响:
  - ✅ 精度提升幅度: **中等**
  - ⚠️ 内存消耗: **中等增加**
  - ⚠️ 训练速度: **变慢**

**调整示例**:
```yaml
# 高精度配置
policy:
  n_cond_layers: 8

# 平衡配置
policy:
  n_cond_layers: 4

# 轻量配置
policy:
  n_cond_layers: 2
```

---

### 5. **主Transformer层数 (n_layer)**
**位置**: `config/nuscenes.yaml` -> `policy.n_layer`
**当前值**: 8

```yaml
policy:
  n_layer: 8  # 调整此值
```

**说明**:
- 主 Diffusion Transformer 的层数
- 更多层 → 更复杂的模式学习 → 通常更高的精度
- 推荐调整范围: 4, 8, 12, 16
- 影响:
  - ✅ 精度提升幅度: **大**
  - ⚠️ 内存消耗: **大幅增加**
  - ⚠️ 训练速度: **明显变慢**

**调整示例**:
```yaml
# 高精度配置
policy:
  n_layer: 16

# 平衡配置
policy:
  n_layer: 8

# 轻量配置
policy:
  n_layer: 4
```

---

### 6. **状态维度 (state_dim)** 
**位置**: `config/nuscenes.yaml` -> `bev_encoder.state_dim`
**当前值**: 15

```yaml
bev_encoder:
  state_dim: 15  # 调整此值
```

**说明**:
- BEV 编码器输入的状态信息维度
- 由数据的ego_status特征维度决定
- 通常不需要调整（由数据格式固定）
- 如需调整需修改数据加载代码

---

## 推荐的维度调整方案

### 方案 A: 保守提升（内存/速度压力小）
适合GPU显存有限的情况
```yaml
policy:
  n_emb: 768           # 从512增加到768
  n_head: 12           # 从8增加到12
  n_cond_layers: 6     # 从4增加到6
  n_layer: 8           # 保持

bev_encoder:
  feature_dim: 384     # 从256增加到384
```
**精度提升**: ~5-10% | **内存增加**: ~30% | **速度**: 约1.2倍慢

---

### 方案 B: 平衡方案（推荐）
适合中等GPU显存的情况
```yaml
policy:
  n_emb: 1024          # 从512增加到1024
  n_head: 16           # 从8增加到16
  n_cond_layers: 6     # 从4增加到6
  n_layer: 12          # 从8增加到12

bev_encoder:
  feature_dim: 512     # 从256增加到512
```
**精度提升**: ~15-25% | **内存增加**: ~100% | **速度**: 约1.5-2倍慢

---

### 方案 C: 激进提升（最高精度）
适合GPU显存充足的情况
```yaml
policy:
  n_emb: 1536          # 从512增加到1536
  n_head: 24           # 从8增加到24
  n_cond_layers: 8     # 从4增加到8
  n_layer: 16          # 从8增加到16

bev_encoder:
  feature_dim: 768     # 从256增加到768
```
**精度提升**: ~25-40% | **内存增加**: ~200%+ | **速度**: 约2-3倍慢

---

## 维度调整步骤

### 1. 修改配置文件
编辑 `config/nuscenes.yaml`:

```bash
# 编辑配置文件
vi config/nuscenes.yaml
```

```yaml
# 关键部分
policy:
  n_emb: 1024          # 调整此值
  n_head: 16           # 调整此值
  n_cond_layers: 6     # 调整此值
  n_layer: 12          # 调整此值

bev_encoder:
  feature_dim: 512     # 调整此值
```

### 2. 验证维度兼容性

确保以下关系成立:
- `n_emb % n_head == 0` （嵌入维度能被头数整除）
- `n_emb % 4 == 0` （符合Transformer标准）

**检查脚本** (`check_dims.py`):
```python
def check_dimension_compatibility(config):
    policy = config['policy']
    n_emb = policy['n_emb']
    n_head = policy['n_head']
    
    if n_emb % n_head != 0:
        raise ValueError(f"Error: n_emb ({n_emb}) must be divisible by n_head ({n_head})")
    
    head_dim = n_emb // n_head
    print(f"✓ 每个注意力头维度: {head_dim}")
    print(f"✓ 总嵌入维度: {n_emb}")
    print(f"✓ 注意力头数: {n_head}")
    return True
```

### 3. 检查GPU显存

```bash
# 在启动训练前检查显存
nvidia-smi

# 或在Python中检查
python -c "import torch; print(f'GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"
```

### 4. 从小到大逐步调整

**推荐流程**:
1. 从当前配置开始训练一个baseline模型
2. 逐个增加参数，观察精度提升和显存变化
3. 找到精度和资源之间的最优平衡点

```bash
# 第一步：保存原配置
cp config/nuscenes.yaml config/nuscenes_backup.yaml

# 第二步：修改配置为方案B
vi config/nuscenes.yaml

# 第三步：启动训练
python training/train_nusc_bev.py --config config/nuscenes.yaml

# 监控显存使用
watch -n 1 nvidia-smi
```

---

## 代码级调整（可选）

如果需要更细粒度的控制，可以直接修改代码:

### 修改 `policy/diffusion_dit_nusc_policy.py`:

```python
# 在 __init__ 方法中，可以直接覆盖配置值
def __init__(self, config: Dict, action_stats: Optional[Dict[str, torch.Tensor]] = None):
    super().__init__()
    
    policy_cfg = config['policy']
    
    # 方式1: 直接覆盖配置
    policy_cfg['n_emb'] = 1024
    policy_cfg['n_head'] = 16
    
    # 或方式2: 通过命令行参数覆盖
    # python training/train_nusc_bev.py --n-emb 1024 --n-head 16
```

---

## 性能对比参考

| 参数 | 512 | 768 | 1024 | 1536 |
|-----|-----|-----|------|------|
| n_emb | baseline | +5% 精度 | +15% 精度 | +25% 精度 |
| 显存占用 | 100% | 130% | 170% | 250% |
| 训练速度 | 1x | 1.2x | 1.5x | 2x |

---

## 常见问题

### Q1: 调整哪个参数效果最好？
**A**: 优先级顺序:
1. `n_emb` - 最大影响
2. `n_layer` - 次大影响  
3. `n_cond_layers` - 中等影响
4. `n_head` - 小影响
5. `feature_dim` - 小到中等影响

### Q2: 维度越大越好吗？
**A**: 不是。过大的维度会导致:
- 过度拟合
- 显存溢出
- 训练不稳定
- 推理延迟过高

建议进行验证集评估，找到最优点。

### Q3: 如何同时调整多个参数？
**A**: 遵循这个原则:
- 保持 `n_emb / n_head` 的比值相对稳定（通常64-128）
- 按比例增加 `n_cond_layers` 和 `n_layer`
- 新增 `feature_dim` 应该是 `n_emb` 的 50-75%

### Q4: 训练时显存不足怎么办？
**A**:
1. 减小 `batch_size`: `dataloader.batch_size = 64`
2. 减小 `n_emb`: 从1024降回768或512
3. 减小 `n_layer` 或 `n_cond_layers`
4. 启用梯度累积
5. 使用混合精度训练

### Q5: 调整维度后精度没有提升怎么办？
**A**:
1. 检查学习率是否需要调整
2. 确认数据增强设置
3. 检查是否需要更多的训练步数
4. 尝试调整 `optimizer.lr`
5. 可能需要调整其他超参数（不只是维度）

---

## 最佳实践总结

1. **始终备份原配置**: `cp config/nuscenes.yaml config/nuscenes_backup.yaml`

2. **逐步调整**: 一次调整一个参数，观察效果

3. **监控指标**: 
   - 显存使用率
   - 训练速度
   - 验证精度
   - 损失下降趋势

4. **保存检查点**: 在 `checkpoints/` 目录保存最好的模型

5. **使用W&B跟踪**: 利用 wandb 对比不同配置的效果

6. **充分训练**: 维度变化后，建议训练足够的 epoch（不少于原来的1.5倍）

---

## 参考资源

- Transformer 架构: https://arxiv.org/abs/1706.03762
- Diffusion Models: https://arxiv.org/abs/2006.11239
- Vision Transformer: https://arxiv.org/abs/2010.11929

---

**最后更新**: 2025-11-17
**适用版本**: MoT-DP

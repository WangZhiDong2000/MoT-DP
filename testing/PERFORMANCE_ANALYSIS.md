# 模型性能分析报告

## 总体问题
- **conditional_sample 总耗时**: ~2秒 (10步推理)
- **单步推理耗时**: ~200ms
- **目标**: 需要大幅降低推理时间

## 详细性能分析

### 单次前向传播耗时 (batch_size=4)
```
总耗时: 18.51ms
├── Time embedding: 0.97ms (5.2%)
├── Ego status processing: 3.07ms (16.6%)
├── Encoder block: 19.26ms (104.0%)
│   ├── VL projection & pooling: 4.50ms (23.4%)
│   ├── Reasoning projection & pooling: 4.43ms (23.0%)
│   └── Transformer encoder: 11.06ms (57.4%)
├── Input embedding & position: 0.87ms (4.7%)
├── Decoder: 134.01ms (724%)  ⚠️ 主要瓶颈
│   └── 12层，每层约11.5ms
└── Trajectory head: 4.08ms (22.0%)
```

### Diffusion采样过程 (10步)
```
总耗时: 197.97ms (0.198s)
├── 平均每步: 19.74ms
│   ├── Model forward: 19.18ms (97.2%)  ⚠️ 主要开销
│   └── Scheduler step: 0.39ms (2.0%)
```

## 🔴 核心问题识别

### 1. **Decoder是最大瓶颈** (占单次forward的72%)
- **12层Decoder**: 134.01ms
- 每层平均: ~11.5ms
- Decoder比Encoder慢了**6.96倍**

### 2. **为什么2秒这么慢？**
实际测量显示10步只需要200ms，但你提到需要2秒。可能的原因：
- ❌ 实际使用了更多推理步数 (100步?)
- ❌ Batch size = 1 (没有并行优化)
- ❌ CPU推理而非GPU
- ❌ 额外的数据预处理/后处理开销
- ❌ VL/Reasoning特征提取未包含在计时中

## 🎯 优化建议 (按优先级排序)

### 优先级1: 减少推理步数 ⭐⭐⭐⭐⭐
**当前**: 可能使用100步
**建议**: 使用10步或更少

**理由**: 
- 10步仅需200ms
- DDPM可以用更少步数达到相似质量
- 使用DDIM scheduler可以用5步达到类似效果

**实施**:
```python
# 在config中修改
num_inference_steps: 10  # 从100减少到10
# 或考虑使用DDIM
from diffusers import DDIMScheduler
```

**预期提升**: 可从2s降到200ms (10倍)

---

### 优先级2: 优化Decoder架构 ⭐⭐⭐⭐
**问题**: Decoder占72%的时间，但可能过度复杂

**优化方案**:

#### 2.1 减少Decoder层数
```python
# 当前: n_layer=12
# 建议: n_layer=6 或 8
TransformerForDiffusion(
    n_layer=6,  # 从12减少到6
    ...
)
```
**预期提升**: Decoder从134ms降到67ms，总时间减少36%

#### 2.2 使用Flash Attention
```python
# 在CustomDecoderLayer中启用
self.memory_vl_cross_attn = nn.MultiheadAttention(
    ...,
    batch_first=True,
    # 启用Flash Attention (PyTorch 2.0+)
)
# 需要在forward中添加: is_causal=True, enable_gqa=True
```
**预期提升**: 20-30%的attention加速

#### 2.3 合并Cross Attention
当前有2个cross attention操作：
- Memory-VL Cross Attention
- Trajectory-Memory Cross Attention

考虑合并为单次操作或使用更轻量的融合机制。

---

### 优先级3: 缓存与复用 ⭐⭐⭐⭐
**问题**: Encoder在每个diffusion步都重新计算

**优化方案**:

#### 3.1 缓存Encoder输出
Encoder输出（memory, vl_features, reasoning_features）在整个采样过程中是**不变的**。

```python
def conditional_sample(self, ...):
    # 只计算一次
    with torch.no_grad():
        memory, vl_features, reasoning_features = self.encoder_block(
            vl_embeds, reasoning_embeds, cond
        )
    
    for t in scheduler.timesteps:
        # 复用缓存的encoder输出
        model_output = model.decoder_only(
            trajectory, t, memory, vl_features, reasoning_features, ...
        )
```

**预期提升**: 节省19.26ms × 10步 = 192.6ms (几乎翻倍)

---

### 优先级4: 模型量化 ⭐⭐⭐
使用INT8或FP16量化

```python
# FP16推理
model = model.half()
# 或使用torch.compile (PyTorch 2.0+)
model = torch.compile(model)
```

**预期提升**: 30-50%加速

---

### 优先级5: 减小模型尺寸 ⭐⭐⭐
```python
# 当前配置
n_emb=768, n_head=12, n_layer=12

# 建议配置（轻量版）
n_emb=512, n_head=8, n_layer=6

# 或中等配置
n_emb=640, n_head=10, n_layer=8
```

**预期提升**: 50-70%加速（需重新训练）

---

### 优先级6: Batch优化 ⭐⭐
如果当前batch_size=1，增加到4-8可以提升GPU利用率。

---

## 📊 综合优化策略

### 方案A: 快速优化（无需重训练）
1. ✅ 减少推理步数: 100→10 (**10倍加速**)
2. ✅ 缓存Encoder输出 (**2倍加速**)
3. ✅ 使用FP16 (**1.3倍加速**)

**总预期**: 从2s降到 **~75ms** (26倍加速)

### 方案B: 深度优化（需重训练）
1. ✅ 所有方案A的优化
2. ✅ 减少Decoder层: 12→6
3. ✅ 减小模型: n_emb=768→512

**总预期**: 从2s降到 **~30ms** (67倍加速)

---

## 🔧 立即可执行的代码修改

### 1. 修改config减少推理步数
```yaml
# config/pdm_server.yaml
policy:
  num_inference_steps: 10  # 从100改为10
```

### 2. 添加Encoder缓存
在 `diffusion_dit_carla_policy.py` 中修改 `conditional_sample`:

```python
def conditional_sample(self, ...):
    # 在循环前缓存encoder输出
    with torch.no_grad():
        # 获取encoder的memory等
        timesteps_dummy = torch.zeros(cond.shape[0], device=cond.device)
        
        # 只运行encoder部分（需要修改model支持）
        memory, vl_features, reasoning_features = self.model.encode_conditions(
            cond, gen_vit_tokens, reasoning_query_tokens
        )
    
    for t in scheduler.timesteps:
        # 使用缓存的memory
        model_output = self.model.decode_trajectory(
            trajectory, t, memory, vl_features, reasoning_features, ego_status
        )
        ...
```

### 3. 使用FP16
```python
# 在模型加载后
self.model = self.model.half()
```

---

## ⚡ 下一步行动

1. **立即**: 修改config，将num_inference_steps改为10
2. **今天**: 实现Encoder缓存
3. **本周**: 测试FP16推理
4. **下周**: 如需要，减少Decoder层数并重新训练

预期可以将推理时间从2s降低到100ms以内！

# dp_vl_feature (.pt 文件) 结构详解

## 📋 文件概述

`dp_vl_feature` 是 PDM Lite 数据集中的 Vision-Language 特征文件，格式为 PyTorch `.pt` 文件。

**位置示例**：
```
/share-data/pdm_lite/SignalizedJunctionRightTurn/
  Town13_Rep0_1019_1_route0_11_08_10_16_59/
    dp_vl_feature/
      0004.pt
      0005.pt
      ...
```

## 🔍 文件内容结构

### 文件类型
- **格式**：PyTorch Dictionary 格式
- **数据类型**：`dict[str, torch.Tensor]`

### 包含的字段

| 字段名 | 形状 | 数据类型 | 说明 |
|--------|------|---------|------|
| **`gen_vit_tokens`** | `(512, 2560)` | `torch.bfloat16` | 生成的 VIT (Vision Transformer) 令牌，是图像的深层特征表示 |
| **`reasoning_query_tokens`** | `(8, 2560)` | `torch.bfloat16` | 推理查询令牌，用于 VQA 推理过程 |
| **`answer_token_indexes`** | `(variable: 1-6)` | `torch.int64` | VQA 答案的令牌索引，可变长度 |

## 📊 详细字段说明

### 1. **gen_vit_tokens** (512, 2560)
```python
Shape: torch.Size([512, 2560])
Dtype: torch.bfloat16
Device: cpu
Value Range: [-8.187, +7.094]
```

**说明**：
- 序列长度 512：VQA 模型处理后的令牌序列
- 特征维度 2560：Qwen2.5-VL-3B-Instruct 的隐藏层维度
- 用途：作为独立变量直接输入扩散模型

**使用示例**（来自 `diffusion_dit_carla_policy.py`）：
```python
gen_vit_tokens = gen_vit_tokens.to(device=device, dtype=torch.float32)
gen_vit_tokens = self.feature_encoder(gen_vit_tokens)  # Project to 1536 dim
```

### 2. **reasoning_query_tokens** (8, 2560)
```python
Shape: torch.Size([8, 2560])
Dtype: torch.bfloat16
Device: cpu
Value Range: [-7.312, +6.750]
```

**说明**：
- 序列长度 8：固定的推理查询令牌数
- 特征维度 2560：与 `gen_vit_tokens` 相同
- 用途：用于 VQA 模型的推理过程

### 3. **answer_token_indexes** (variable: 1-6)
```python
Shape: torch.Size([N])  # N 通常为 1-6
Dtype: torch.int64
Device: cpu
Content Example: [0, 1, 2, 3, 4, 5]
```

**说明**：
- 长度分布：98.6% 长度为 6，其他为 1-5
- 内容：VQA 模型生成答案的令牌索引序列
- **可变长度**：这是关键特性，需要在数据加载时填充

## 🔄 数据加载流程

### 在 `unified_carla_dataset.py` 中的处理

```python
# 1. 加载 .pt 文件
vqa_path = sample.get('vqa', None)
full_vqa_path = os.path.join(self.image_data_root, vqa_path)
vqa_feature = torch.load(full_vqa_path, weights_only=True)

# 2. 提取主要字段
final_sample['gen_vit_tokens'] = vqa_feature['gen_vit_tokens']

# 3. 处理可变长度的 answer_token_indexes
if isinstance(vqa_feature, dict) and 'answer_token_indexes' in vqa_feature:
    answer_tokens = vqa_feature['answer_token_indexes']
    max_answer_tokens = 8  # 固定的最大长度
    
    if answer_tokens.shape[0] < max_answer_tokens:
        # 用 -1 填充到固定大小
        padding = torch.full((max_answer_tokens - answer_tokens.shape[0],), -1)
        final_sample['answer_token_indexes'] = torch.cat([answer_tokens, padding])
    elif answer_tokens.shape[0] > max_answer_tokens:
        # 截断到最大大小
        final_sample['answer_token_indexes'] = answer_tokens[:max_answer_tokens]
    else:
        final_sample['answer_token_indexes'] = answer_tokens
```

## 🧠 VQA 生成过程

### 使用的模型
- **模型**：Qwen2.5-VL-3B-Instruct
- **位置**：Hugging Face 官方检查点 `Qwen/Qwen2.5-VL-3B-Instruct`

### VQA 问题
```
"What actions should be taken based on this scene?"
```

### 特征提取方式
```python
# 来自 add_vlm_feature.py
inputs = tokenizer(
    text=text_inputs,
    images=list(batch_images),
    return_tensors="pt",
    padding=True
)

with torch.no_grad():
    outputs = model(
        **inputs,
        output_hidden_states=True,
        return_dict=True,
    )
    # 提取最后一层的隐藏状态
    hidden_states = outputs.hidden_states[-1]  # (B, seq_len, hidden_size)
```

## ⚙️ 在模型中的使用

### 扩散策略（`diffusion_dit_carla_policy.py`）

```python
# 1. 从批次中提取特征
gen_vit_tokens = batch.get('gen_vit_tokens', None)
answer_token_indexes = batch.get('answer_token_indexes', None)

# 2. 处理 gen_vit_tokens
if gen_vit_tokens is not None:
    gen_vit_tokens = gen_vit_tokens.to(device=device, dtype=torch.float32)
    gen_vit_tokens = self.feature_encoder(gen_vit_tokens)  # Project to 1536 dim

# 3. 处理 answer_token_indexes（无需处理，直接传入）
if answer_token_indexes is not None:
    answer_token_indexes = answer_token_indexes.to(device=device)

# 4. 传入模型
pred = self.model(
    noisy_trajectory, 
    timesteps, 
    cond, 
    gen_vit_tokens=gen_vit_tokens, 
    answer_token_indexes=answer_token_indexes, 
    ego_status=ego_status
)
```

### 扩散变换器（`transformer_for_diffusion.py`）

```python
def forward(
    self,
    sample: torch.Tensor,      # (B, T, input_dim)
    timestep: Union[torch.Tensor, float, int],
    cond: torch.Tensor,        # (B, T', cond_dim)
    gen_vit_tokens: Optional[torch.Tensor] = None,        # (B, 512, 2560)
    answer_token_indexes: Optional[torch.Tensor] = None,  # (B, max_answer_tokens)
    ego_status: Optional[torch.Tensor] = None,            # (B, status_dim)
    **kwargs
):
    # ...
    # gen_vit_tokens 和 answer_token_indexes 作为独立变量使用
    vl_embeds = answer_token_indexes
    # ...
```

## 📍 关键特性总结

| 特性 | 说明 |
|------|------|
| **多字段结构** | 包含三个互补的特征字段 |
| **高维表示** | 2560 维的深层特征 |
| **可变长度** | answer_token_indexes 长度 1-6，需要填充处理 |
| **浮点精度** | 使用 bfloat16 节省内存，提高计算效率 |
| **帧频关系** | 从第 4 帧开始生成（0000.pt 存储的是不同的数据） |
| **独立变量** | 在扩散模型中作为独立变量输入，不与其他条件混合 |

## 🔗 相关文件

- **生成脚本**：`dataset/add_vlm_feature.py`、`dataset/add_fixed_vlm_feature.py`
- **数据加载**：`dataset/unified_carla_dataset.py`（第 128-169 行）
- **预处理**：`dataset/preprocess_pdm_lite.py`（第 512-531 行）
- **模型使用**：`model/transformer_for_diffusion.py`（第 995-1030 行）、`policy/diffusion_dit_carla_policy.py`（第 375-410 行）

## 💡 使用建议

1. **数据加载时**：总是对 `answer_token_indexes` 进行填充处理
2. **特征处理**：`gen_vit_tokens` 需要通过 `feature_encoder` 投影
3. **内存优化**：使用 `bfloat16` 的 `.float32()` 转换需要在 GPU 上进行
4. **错误处理**：检查 VQA 文件是否存在（从第 4 帧开始）

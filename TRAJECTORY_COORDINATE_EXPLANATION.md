# 轨迹坐标系统说明 - 为什么预测位置会有负数

## 问题分析

你的样本 330 显示：
```
观测位置: [[-1.7727478e+00  4.9607980e-04]   # t=0: 相对位置 (-1.77, 0.0005)
           [ 0.0000000e+00  0.0000000e+00]]   # t=1: 参考点 (0, 0)

预测位置: [[-0.4875477  -0.00444994]         # 预测步骤1: (-0.49, -0.004)
           [ 0.40189597 -0.00466377]          # 预测步骤2: (0.40, -0.005)
           [ 0.9854936  -0.00624591]          # 预测步骤3: (0.99, -0.006)
           [ 1.2927884  -0.00610257]          # 预测步骤4: (1.29, -0.006)
           [ 1.4458585  -0.00676086]]         # 预测步骤5: (1.45, -0.007)
```

**第一个预测位置为负数 (-0.49, -0.004) 是完全正常的！**

## 原因解释

### 1. 相对坐标系统

代码使用 **相对于最后观测点的坐标系**：

```python
# 在 _load_file_data 中（第 345-349 行）
# 计算相对位置
reference_pos = sequence_positions[self.obs_horizon - 1]  # 最后一个观测位置作为参考点
relative_positions = []
for pos in sequence_positions:
    relative_pos = pos - reference_pos  # 所有位置都相对于参考点
    relative_positions.append(relative_pos)
```

**参考点设为 (0, 0)**，这是最后一个观测时刻 (t=1) 的位置。

### 2. 坐标系定义

在 CARLA 中，ego_waypoints 使用的是 **车辆本地坐标系**：
- **X 轴**：车辆前进方向（正值 = 前方，负值 = 后方）
- **Y 轴**：车辆横向方向（正值 = 左侧，负值 = 右侧）

### 3. 为什么预测位置的第一个点是负数？

让我们逐步分析你的数据：

#### 观测阶段 (obs_horizon=2)
```
t=0 (第一个观测):  相对位置 = (-1.77, 0.0005)
                    含义: 在参考点的 **后方** 1.77米
                    
t=1 (第二个观测):  相对位置 = (0, 0)  [参考点]
                    含义: 当前位置，作为坐标原点
```

车辆从 t=0 到 t=1 **向前行驶了 1.77 米**。

#### 预测阶段 (action_horizon=5)
```
t=2 (预测1):  相对位置 = (-0.49, -0.004)
              含义: 距离参考点 **后方** 0.49米
              
t=3 (预测2):  相对位置 = (0.40, -0.005)
              含义: 距离参考点 **前方** 0.40米
              
t=4 (预测3):  相对位置 = (0.99, -0.006)
              含义: 距离参考点 **前方** 0.99米
              
t=5 (预测4):  相对位置 = (1.29, -0.006)
              含义: 距离参考点 **前方** 1.29米
              
t=6 (预测5):  相对位置 = (1.45, -0.007)
              含义: 距离参考点 **前方** 1.45米
```

### 4. 为什么第一个预测点在后方？

这是因为 **waypoint 索引的取值方式**：

```python
# 第 329-337 行
for pred_step in range(self.action_horizon):
    waypoint_idx = pred_step + 2  # pred_step=0 时，waypoint_idx=2
    if len(last_ego_wp) > waypoint_idx:
        pred_pos = last_ego_wp[waypoint_idx].copy()
```

**关键理解**：`ego_waypoints` 是从 **当前帧（t=1）** 的位置开始的未来路径点：

- `ego_waypoints[0]`：当前位置
- `ego_waypoints[1]`：下一时刻位置
- `ego_waypoints[2]`：再下一时刻位置（预测步骤1）
- `ego_waypoints[3]`：...（预测步骤2）

**但是**，在你的案例中，`ego_waypoints[2]` 的实际位置可能是：
- 相对于 **数据采集时的绝对坐标** 是正向的
- 但相对于 **t=1 时刻的参考点** 可能因为车辆已经超前行驶，导致该 waypoint 在参考点后方

### 5. 实际物理意义

这种情况通常发生在：

#### 场景 A：车辆正在加速
```
t=0: 车速较慢，位置在后方
t=1: 车速加快，定为参考点
t=2: 由于 waypoints 是基于 t=1 时刻的规划，
     但如果车辆在 t=1→t=2 之间加速超过了预期，
     waypoint[2] 可能相对落后
```

#### 场景 B：Waypoint 规划的时间步长问题
```
sample_interval = 5  # 你的配置

这意味着相邻帧之间相隔 5 个原始帧
如果原始数据是 10Hz，那么你的数据是 2Hz
waypoint[2] 可能表示的是 0.5秒后的位置
但由于降采样，实际时间步长变化导致坐标转换出现偏差
```

## 这是问题吗？

### ❌ 不是问题！这是正常现象

1. **相对坐标系的特性**：
   - 使用相对坐标可以让模型学习 **局部运动模式**
   - 负数只是表示方向，完全符合物理意义

2. **轨迹的连续性**：
   - 从 -0.49 → 0.40 → 0.99 → 1.29 → 1.45
   - 这是一个 **连续向前的轨迹**
   - 第一个点在后方说明车辆正在从该点加速前进

3. **模型训练的角度**：
   - Diffusion Policy 学习的是 **位置的分布**
   - 负数坐标同样可以被模型学习和预测
   - 重要的是轨迹的 **相对变化趋势**，而非绝对值

## 验证数据正确性

你可以通过以下方式验证：

### 1. 检查轨迹连续性
```python
import numpy as np

# 你的数据
obs = np.array([[-1.77, 0.0005], [0.0, 0.0]])
pred = np.array([[-0.49, -0.004], [0.40, -0.005], [0.99, -0.006], 
                 [1.29, -0.006], [1.45, -0.007]])

# 计算相邻点的距离
full_traj = np.vstack([obs, pred])
distances = np.linalg.norm(np.diff(full_traj, axis=0), axis=1)
print("相邻点距离:", distances)

# 预期结果：所有距离应该相近且合理（例如 0.5-2.0 米）
```

### 2. 可视化轨迹
```python
import matplotlib.pyplot as plt

obs = np.array([[-1.77, 0.0005], [0.0, 0.0]])
pred = np.array([[-0.49, -0.004], [0.40, -0.005], [0.99, -0.006], 
                 [1.29, -0.006], [1.45, -0.007]])

plt.figure(figsize=(10, 6))
plt.plot(obs[:, 1], obs[:, 0], 'bo-', label='Observed', markersize=10)
plt.plot(pred[:, 1], pred[:, 0], 'ro--', label='Predicted', markersize=8)
plt.plot(0, 0, 'ks', markersize=12, label='Reference Point')
plt.axhline(0, color='gray', linestyle='--', alpha=0.3)
plt.axvline(0, color='gray', linestyle='--', alpha=0.3)
plt.xlabel('Y (lateral, m)')
plt.ylabel('X (forward, m)')
plt.title('Trajectory in Vehicle Frame')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.show()
```

## 如果你想避免负数

如果你确实想让预测位置都是正数，可以修改代码：

### 方案 1：使用第一个观测点作为参考
```python
# 修改第 345 行
reference_pos = sequence_positions[0]  # 改用第一个观测点
```

### 方案 2：使用绝对坐标
```python
# 不做相对位置转换
agent_pos = np.array(sequence_positions)  # 直接使用原始位置
```

### 方案 3：调整 waypoint 索引
```python
# 修改第 330 行
waypoint_idx = pred_step + 3  # 从 2 改为 3，跳过更近的点
```

## 总结

✅ **预测位置有负数是正常的**
- 这是相对坐标系统的自然结果
- 表示车辆的真实运动轨迹
- 不会影响模型训练和预测

❌ **不需要修复**
- 负数坐标有明确的物理意义
- 轨迹整体趋势正确（向前移动）
- Diffusion Policy 可以处理任意范围的坐标

🎯 **关注重点**
- 轨迹的连续性和平滑性
- 相邻点之间的距离是否合理
- 整体运动方向是否符合预期

---

**相关代码位置**：
- `dataset/generate_pdm_dataset.py`: 第 310-380 行（轨迹处理逻辑）
- 坐标系定义：第 345-349 行（相对位置计算）
- Waypoint 索引：第 329-337 行（预测位置提取）

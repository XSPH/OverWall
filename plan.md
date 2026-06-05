---
## 一、研究现状总结

### 与你场景最相关的论文

| 论文 | 关键思路 | 与你场景的关系 |
|------|----------|----------------|
| **Acrobotics (2025)** — 四足机器人跑酷 | 单一通用策略完成攀爬、跳跃、跨越，攀爬高度达75cm（机器人髋高），仅用25%的训练智能体 | 核心参考：证明 **不需要多专家混合**，one policy + curriculum 即可学会攀爬 |
| **Wheel-Legged Quadruped RL (IROS 2025)** | 非对称 Actor-Critic + 速度估计网络，纯 proprioception 在台阶/障碍/低附着力表面行走 | 与你当前 teacher-student 架构一致 |
| **RL Framework for Quadrupedal Wall Climbing (2025)** | 三阶段课程：平地爬行 → 重力旋转 → 附着不确定性，磁吸附爬墙成功率~90% | 课程设计思路可借鉴，但你是越墙不是爬墙 |
| **SoloParkour (CoRL 2024)** | 约束RL：在物理极限内最大化敏捷性，从 privileged experience 蒸馏到视觉策略 | 约束RL思路可用于安全攀爬 |
| **ANYmal Parkour (Science Robotics 2024)** | 分层框架：技能库(走/跳/爬/蹲) + 高层导航策略 | 你不需要分层，one policy 即可 |

### 关键结论

- **One policy 足够**：不需要多专家/技能库，一个策略 + 课程学习 + 好的奖励设计即可学会越墙
- **Teacher-student 蒸馏有效**：你现在用的就是这个架构，teacher 有完整地形信息，student 只靠本体感觉
- **课程学习是关键**：从低墙开始，逐步增加高度
- **没有现成的轮足越墙方案**：这是2024-2025文献的空白点，需要你自己设计

---

## 二、物理可行性分析（30cm × 5cm 墙）

### 你的机器人关键参数

- 站立基座高度：**0.43m**
- 腿部最大伸展：大腿 0.24m + 小腿 0.23m = **0.47m**
- 轮子半径：**0.1m**
- 关节力矩：髋/大腿/小腿 = 36/36/35 Nm，轮子 17 Nm
- 总质量：~19.4 kg

### 30cm 墙的物理挑战

30cm 墙是机器人站立高度的 **70%**。机器人无法直接"开过去"，必须执行一个攀爬动作：

```
阶段1: 接近      阶段2: 前轮上台     阶段3: 引体向上     阶段4: 后轮过渡     阶段5: 跨越
  🤖 → ═══       🤖↗ ═══           🤖╲  ═══          🤖 → ═══          🤖 →→
  ○ ○ │墙│       ○  ○│墙│           ○  ○│墙│           │墙│○  ○           │墙│ ○ ○
  ↓ ↓ │  │        ↓  ↗│  │            ↘ ↗│  │           │  │↘ ↗            │  │ ↓ ↓
══════╧══╧═     ══════╧══╧═       ══════╧══╧═       ═══╧══╧═════      ═══╧══╧═════
```

- 前轮需要达到 0.3m 高度 → 前腿需要从倾斜的基座伸展够到墙顶
- 身体需要倾斜 ~30-40° 来完成过渡
- 后腿需要在 COM 过墙后快速抬起

**结论：物理上可行，但需要学习一个非平凡的协调动作。**

---

## 三、当前代码的优势与不足

### 已有优势

- Teacher-student 架构已就绪（privileged encoder → proprioceptive encoder 蒸馏）
- 墙体地形生成已完成（`low_wall_terrain`，高度课程 10-45cm）
- 身高扫描点（187个点覆盖机器人周围 1.6m × 1.0m）已接入 privileged observation
- 域随机化框架完善（摩擦、质量、KP/KD、电机偏移等）

### 核心不足

1. **奖励函数完全是通用 locomotion 奖励**：只激励速度跟踪，没有越墙相关的奖励信号
2. **终止条件对攀爬不友好**：trap-static（5秒卡住就终止）会在机器人尝试攀爬时触发
3. **命令课程与墙体课程不协调**：当墙很高时，如果仍然命令高速前进，机器人不可能跟踪上
4. **没有越墙过程的阶段性引导**：policy 需要自己"发现"攀爬行为，样本效率极低

---

## 四、可行方案

核心思路：**不改变你的 teacher-student 架构和 terrain 生成，只在奖励函数、终止条件、命令课程上做针对性修改，让 policy 自己在课程学习中"发现"越墙行为。**

### 4.1 新增奖励函数（关键）

在 `legged_robot.py` 中新增以下奖励函数，并在 `go2w_config.py` 中配置权重：

```python
# ============= 越墙核心奖励 =============

def _reward_wall_progress(self):
    """奖励机器人COM在墙体区域的向前位移。
    仅在检测到前方有墙时生效。"""
    # COM 前进位移（相对墙体接触时刻的位置）
    progress = self.base_position[:, 0] - self.wall_contact_x
    # 只在墙附近时有效
    wall_nearby = self._is_wall_nearby()  # 前方0.5m内检测到高度突变
    return torch.clamp(progress, min=0.) * wall_nearby.float()

def _reward_wall_front_lift(self):
    """奖励前轮高度超过墙体高度的一定比例。
    这是引导机器人"抬起前腿"的关键信号。"""
    wall_height = self._get_wall_height_ahead()  # 从高度扫描获取
    front_wheel_height = self.foot_positions[:, [0, 2], 2]  # FL, FR 的 z 坐标
    # 前轮超过墙高80%时给显著奖励
    lift_ratio = front_wheel_height / (wall_height + 0.05)
    return torch.sum(torch.sigmoid((lift_ratio - 0.8) * 10), dim=1)

def _reward_wall_crossed(self):
    """稀疏奖励：机器人COM越过墙体中心线时给大奖励。
    这是一次性触发奖励，通过检测COM x坐标超过墙体x坐标来判断。"""
    crossed = (self.base_position[:, 0] > self.wall_center_x) & \
              (self.last_base_position[:, 0] <= self.wall_center_x)
    return crossed.float() * 10.0  # 一次 10 的大奖励

def _reward_wall_height_gain(self):
    """奖励基座高度接近目标越墙高度。
    帮助机器人学会在爬墙时保持合适的高度。"""
    wall_nearby = self._is_wall_nearby()
    # 目标高度 = 墙高 + 正常站立高度
    target_height = self.wall_height_ahead + 0.35
    height_error = torch.abs(self.root_states[:, 2] - target_height)
    return torch.exp(-height_error / 0.1) * wall_nearby.float()
```

### 4.2 越墙感知辅助函数

```python
def _detect_wall(self):
    """从 measured_heights 中检测前方墙体。
    墙体特征：前方高度扫描点突然升高。
    返回: wall_distance, wall_height, wall_detected"""
    # measured_heights: (num_envs, 187), reshape to (num_envs, 17, 11)
    heights = self.measured_heights.view(self.num_envs, 17, 11)
    # 只看正前方 (x > 0) 的扫描线
    front_heights = heights[:, 8:, :]  # x: 0 to +0.8m
    # 检测最大高度点
    max_height, max_idx = torch.max(front_heights.reshape(self.num_envs, -1), dim=1)
    # 如果前方最大高度 > 阈值 (5cm)，认为检测到墙
    wall_detected = max_height > 0.05
    # 墙距离 = 最大高度点的x坐标
    wall_distance = ...
    return wall_distance, max_height, wall_detected
```

### 4.3 终止条件调整

当前 `trap_static_time > 5s` 对越墙过程太严格。在墙上时机器人可能短暂停顿。

```python
# 在 check_termination() 中修改：
wall_nearby = self._is_wall_nearby()
# 靠近墙时，trap_static 容忍时间从5s增加到10s
trap_threshold = torch.where(wall_nearby, 10.0, 5.0)
long_time_trap = self.trap_static_time > trap_threshold
```

### 4.4 命令课程调整

在墙体区域，应该倾向于让机器人前进（面向墙体），而不是横向移动或转向。

```python
# 在 _resample_commands 中，靠近墙时偏置命令为前进
def _resample_commands(self, env_ids):
    # ... existing code ...
    # 墙体偏置：靠近墙时更倾向于前进命令
    if hasattr(self, 'wall_nearby'):
        bias_mask = self.wall_nearby[env_ids]
        # 对有墙的env，提高前进速度命令
        self.commands[env_ids[bias_mask], 0] = torch.clip(
            self.commands[env_ids[bias_mask], 0], 0.3, 1.5  # 正向前进
        )
        self.commands[env_ids[bias_mask], 1] *= 0.3  # 减少横向命令
```

### 4.5 奖励惩罚松弛

越墙时需要更大的力矩和身体倾斜，应该暂时放宽限制：

```python
def _reward_torques(self):
    # 在墙附近降低力矩惩罚
    wall_relief = torch.where(self._is_wall_nearby(), 0.3, 1.0)
    return torch.sum(torch.square(self.torques[:,~self.wheel_joint_indices]), dim=1) * wall_relief

def _reward_orientation(self):
    # 在墙附近降低朝向惩罚（允许更多倾斜）
    wall_relief = torch.where(self._is_wall_nearby(), 0.3, 1.0)
    return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1) * wall_relief
```

### 4.6 课程策略优化

当前配置 `low_wall_height_min=0.10, max=0.45` 是正确的。建议调整：

```python
# go2w_config.py
low_wall_height_min = 0.05   # 从5cm开始，先学会"感觉到墙"
low_wall_height_max = 0.35   # 最高35cm（含30cm目标）

# 课程升级条件：变为基于 "越墙成功率" 而非 "前进距离"
# 因为越墙场景中前进距离不总是衡量成功的标准
```

### 4.7 推荐配置权重汇总

```python
# go2w_config.py rewards.scales 新增:
wall_progress = 0.5          # 越墙进度奖励
wall_front_lift = 1.0        # 前轮抬起奖励（重要）
wall_crossed = 5.0           # 成功越墙一次性大奖励
wall_height_gain = 0.3       # 高度增益奖励

# 修改现有权重（越墙时动态松弛通过代码实现，不需要改配置）:
tracking_lin_vel = 2.0       # 保持
tracking_ang_vel = 0.5       # 降低（越墙时转向不重要）
```

---

## 五、实施步骤

### 第一步：添加墙体感知函数

在 `legged_robot.py` 中添加 `_detect_wall()` / `_is_wall_nearby()` 等辅助函数，从已有的 `measured_heights` 中提取墙体信息。

### 第二步：添加越墙奖励函数

添加 `_reward_wall_progress`、`_reward_wall_front_lift`、`_reward_wall_crossed`、`_reward_wall_height_gain`。

### 第三步：修改终止条件

放宽墙附近的 trap-static 时间阈值。

### 第四步：修改命令采样

在墙体环境中偏置为前进命令。

### 第五步：调整域随机化

增加墙体摩擦随机化、墙体高度随机抖动（±2cm），提升 sim-to-real 鲁棒性。

### 第六步：训练

- Phase 1（teacher-student 蒸馏）：5000-10000 iterations
- Phase 2（student reinforcing）：5000+ iterations
- 预期在 phase 1 中期（~3000 iter）机器人就能在低墙上出现攀爬行为，phase 1 后期（~8000 iter）应能稳定跨越 30cm 墙

---

## 六、预期难点与应对

| 难点 | 应对方案 |
|------|----------|
| 5cm 窄墙，轮子容易滑落 | 增加墙体顶面摩擦随机化（训练范围 0.2-2.0），policy 学会稳健放置 |
| 前轮上台后身体大幅倾斜触发终止 | 放宽 projected_gravity 终止阈值（从 -0.1 到 0.2）或添加墙体豁免 |
| 越墙行为发现慢 | 稀疏奖励 `wall_crossed` + 密集奖励 `wall_front_lift` 组合引导 |
| Sim-to-real gap | 已有完善的域随机化框架，额外增加墙体高度/位置随机扰动 |

---

## 总结

你的代码基础设施已经支持这个任务了。核心改动量不大——大约需要在 `legged_robot.py` 中新增 150-200 行代码（越墙感知 + 奖励函数），在 `go2w_config.py` 中新增约 20 行配置。不需要改变 teacher-student 架构、不需要改 terrain 生成、不需要改网络结构。
import gym
from gym import spaces
import numpy as np
from utils.data_generator import generate_task_instance
import logging


class USVSchedulingEnv(gym.Env):
    """
    优化后的USV调度环境 - 以Makespan优化为核心
    主要改进：
    1. 简化奖励函数为makespan差值
    2. 添加辅助引导机制
    3. 优化动作选择策略
    """

    def __init__(self, num_usvs=3, num_tasks=30, area_size_x=(0, 500), area_size_y=(0, 500),
                 processing_time_range=(45, 90), battery_capacity=100, speed_range=(1, 3),
                 charge_time=5):
        super().__init__()

        # 环境参数
        self.num_usvs = num_usvs
        self.num_tasks = num_tasks
        self.area_size_x = area_size_x
        self.area_size_y = area_size_y
        self.processing_time_range = processing_time_range
        self.battery_capacity = battery_capacity
        self.speed_range = speed_range
        self.charge_time = charge_time
        self.debug_mode = True
        self.charging_penalty = 0

        # 定义观测空间和动作空间
        self.observation_space = spaces.Dict({
            'usv_features': spaces.Box(low=0, high=1, shape=(num_usvs, 4)),
            'task_features': spaces.Box(low=0, high=100, shape=(num_tasks, 6)),
            'edge_features': spaces.Box(low=0, high=100, shape=(num_usvs, num_tasks)),
            'action_mask': spaces.Box(low=0, high=1, shape=(num_usvs * num_tasks,), dtype=np.bool_)
        })

        self.action_space = spaces.Discrete(num_usvs * num_tasks)

        # 初始化状态变量
        self.tasks = None
        self.usvs = None
        self._initialize_state_variables()

        # 新增：makespan跟踪变量
        self.last_makespan = 0.0
        self.current_makespan = 0.0
        self.initial_makespan_estimate = 0.0

    def _initialize_state_variables(self):
        """初始化状态变量"""
        self.scheduled_tasks = []
        self.current_time = 0.0
        self.usv_positions = np.zeros((self.num_usvs, 2))
        self.usv_batteries = np.full(self.num_usvs, self.battery_capacity)
        self.usv_speeds = np.zeros(self.num_usvs)
        self.usv_next_available_time = np.zeros(self.num_usvs)
        self.makespan_batch = np.zeros(self.num_tasks)
        self.task_assignment = np.full(self.num_tasks, -1, dtype=int)
        self.task_schedule_details = {}

        # 重置makespan跟踪
        self.last_makespan = 0.0
        self.current_makespan = 0.0

    def reset(self):
        """重置环境为初始状态"""
        default_tasks, default_usvs = generate_task_instance(
            num_tasks=self.num_tasks,
            num_usvs=self.num_usvs,
            area_size_x=self.area_size_x,
            area_size_y=self.area_size_y,
            processing_time_range=self.processing_time_range,
            battery_capacity=self.battery_capacity,
            speed_range=self.speed_range,
            charge_time=self.charge_time
        )
        return self.reset_with_instances(default_tasks, default_usvs)

    def reset_with_instances(self, tasks, usvs):
        """使用给定的任务和USV状态重置环境"""
        self.tasks = tasks
        self.usvs = usvs

        # 重置所有状态变量
        self._initialize_state_variables()

        # 设置USV速度
        self._setup_usv_speeds(usvs['speed'])

        # 估算初始makespan（用于归一化）
        self._estimate_initial_makespan()

        return self._get_observation()

    def _setup_usv_speeds(self, speed_data):
        """设置USV速度"""
        speed_array = np.array(speed_data)

        if len(speed_array) >= self.num_usvs:
            self.usv_speeds = speed_array[:self.num_usvs]
        else:
            default_speed = np.mean(self.speed_range)
            self.usv_speeds = np.pad(
                speed_array,
                (0, self.num_usvs - len(speed_array)),
                constant_values=default_speed
            )

    def _estimate_initial_makespan(self):
        """估算理论最小makespan用于归一化"""
        # 计算所有任务的总处理时间
        total_processing_time = 0
        for task_idx in range(self.num_tasks):
            proc_time = self._get_processing_time(task_idx)
            total_processing_time += proc_time

        # 理论最小makespan（完美均衡情况）
        avg_speed = np.mean(self.usv_speeds)
        avg_distance = np.sqrt((self.area_size_x[1] ** 2 + self.area_size_y[1] ** 2)) / 4  # 估算平均距离
        avg_travel_time = avg_distance / avg_speed

        # 每个USV的理想工作时间
        ideal_work_per_usv = (total_processing_time + self.num_tasks * avg_travel_time) / self.num_usvs
        self.initial_makespan_estimate = ideal_work_per_usv

    def step(self, action):
        """执行动作并更新环境状态"""
        # 解析动作
        usv_idx, task_idx = self._parse_action(action)

        # 验证动作有效性
        if not self._is_valid_action(action):
            return self._handle_invalid_action(action)

        # 执行调度
        reward = self._execute_scheduling(usv_idx, task_idx)

        # 检查是否完成
        done = len(self.scheduled_tasks) >= self.num_tasks

        # 生成信息字典
        info = self._generate_info(usv_idx, task_idx, done)

        return self._get_observation(), reward, done, info

    def _parse_action(self, action):
        """解析动作为USV索引和任务索引"""
        usv_idx = action // self.num_tasks
        task_idx = action % self.num_tasks
        return usv_idx, task_idx

    def _is_valid_action(self, action):
        """检查动作是否有效"""
        usv_idx, task_idx = self._parse_action(action)

        if usv_idx >= self.num_usvs or task_idx >= self.num_tasks:
            return False

        if task_idx in self.scheduled_tasks:
            return False

        return True

    def _handle_invalid_action(self, action):
        """处理无效动作"""
        logging.warning(f"无效动作: {action}")
        # 无效动作给予强惩罚
        return self._get_observation(), -1000.0, False, {'invalid_action': True}

    def _execute_scheduling(self, usv_idx, task_idx):
        """执行调度逻辑"""
        # 保存上一步的makespan
        self.last_makespan = self.current_makespan

        # 获取位置和时间信息
        usv_pos = self.usv_positions[usv_idx]
        task_pos = self.tasks['coords'][task_idx]

        # 计算处理时间
        processing_time = self._get_processing_time(task_idx)

        # 计算距离和旅行时间
        distance = np.linalg.norm(usv_pos - task_pos)
        travel_time = distance / self.usv_speeds[usv_idx]

        # 更新时间计算
        travel_start_time = self.usv_next_available_time[usv_idx]
        processing_start_time = travel_start_time + travel_time
        processing_end_time = processing_start_time + processing_time

        # 更新USV状态
        self._update_usv_state(usv_idx, task_pos, distance, processing_end_time)

        # 记录任务分配
        self._record_task_assignment(task_idx, usv_idx, travel_start_time,
                                     travel_time, processing_start_time, processing_time)

        # 更新当前makespan
        self.current_makespan = np.max(self.usv_next_available_time)

        # 计算奖励
        reward = self._calculate_reward(usv_idx, task_idx)

        return reward

    def _get_processing_time(self, task_idx):
        """获取任务处理时间"""
        proc_time_data = self.tasks['processing_time'][task_idx]
        if isinstance(proc_time_data, (list, tuple, np.ndarray)):
            return np.mean(proc_time_data)
        return proc_time_data

    def _update_usv_state(self, usv_idx, new_position, distance, end_time):
        """更新USV状态"""
        # 更新位置
        self.usv_positions[usv_idx] = new_position

        # 简化的电量消耗模型
        battery_consumption = distance * 0.1
        new_battery = self.usv_batteries[usv_idx] - battery_consumption

        # 电池管理
        if new_battery < 20:
            charge_time = (self.battery_capacity - new_battery) * self.charge_time / 100
            end_time += charge_time
            self.usv_batteries[usv_idx] = self.battery_capacity
        else:
            self.usv_batteries[usv_idx] = new_battery

        # 更新可用时间
        self.usv_next_available_time[usv_idx] = end_time

        # 更新全局时间
        self.current_time = np.max(self.usv_next_available_time)

    def _record_task_assignment(self, task_idx, usv_idx, travel_start, travel_time,
                                processing_start, processing_time):
        """记录任务分配详情"""
        self.task_assignment[task_idx] = usv_idx
        self.scheduled_tasks.append(task_idx)

        completion_time = processing_start + processing_time
        self.makespan_batch[task_idx] = completion_time

        self.task_schedule_details[task_idx] = {
            'task_idx': task_idx,
            'usv_idx': usv_idx,
            'travel_start_time': travel_start,
            'travel_time': travel_time,
            'processing_start_time': processing_start,
            'processing_time': processing_time
        }

    def _calculate_reward(self, usv_idx, task_idx):
        """
        简化的奖励函数 - 以makespan优化为核心
        """
        # ========== 核心奖励：Makespan改进 ==========
        makespan_improvement = self.last_makespan - self.current_makespan

        # 基础奖励：makespan差值
        if len(self.scheduled_tasks) == 1:
            # 第一个任务，给予小奖励鼓励开始
            base_reward = 10.0
        else:
            # makespan改进奖励（放大信号）
            if makespan_improvement > 0:
                # Makespan减少（理论上不应该发生，但如果发生给予大奖励）
                base_reward = 100.0 * makespan_improvement
            elif makespan_improvement == 0:
                # Makespan没有增加（选择了当前最空闲的USV）
                base_reward = 50.0
            else:
                # Makespan增加（根据增加量给予惩罚）
                makespan_increase = -makespan_improvement
                if makespan_increase < 50:
                    base_reward = -0.5 * makespan_increase  # 小幅增加，轻微惩罚
                elif makespan_increase < 100:
                    base_reward = -1.0 * makespan_increase  # 中等增加，中等惩罚
                else:
                    base_reward = -2.0 * makespan_increase  # 大幅增加，重度惩罚

        # ========== 辅助奖励1：选择最优USV ==========
        # 鼓励选择当前完成时间最早的USV
        usv_completion_times = self.usv_next_available_time.copy()
        min_completion_usv = np.argmin(usv_completion_times)

        if usv_idx == min_completion_usv:
            usv_selection_bonus = 30.0  # 选择了最优USV
        else:
            # 根据选择的USV排名给予惩罚
            sorted_indices = np.argsort(usv_completion_times)
            rank = np.where(sorted_indices == usv_idx)[0][0]
            usv_selection_bonus = -10.0 * rank  # 排名越靠后，惩罚越大

        # ========== 辅助奖励2：负载均衡 ==========
        # 只在后期考虑负载均衡
        progress = len(self.scheduled_tasks) / self.num_tasks

        if progress > 0.7:  # 后期才考虑均衡
            task_counts = np.bincount(
                self.task_assignment[self.task_assignment != -1],
                minlength=self.num_usvs
            )
            task_std = np.std(task_counts)

            if task_std < 1.5:
                balance_bonus = 20.0
            elif task_std < 2.5:
                balance_bonus = 0.0
            else:
                balance_bonus = -20.0 * (task_std - 2.5)
        else:
            balance_bonus = 0.0

        # ========== 辅助奖励3：效率奖励 ==========
        # 距离效率（鼓励选择近的任务）
        all_distances = []
        for i in range(self.num_usvs):
            for j in range(self.num_tasks):
                if j not in self.scheduled_tasks:
                    dist = np.linalg.norm(self.usv_positions[i] - self.tasks['coords'][j])
                    all_distances.append(dist)

        if all_distances:
            current_distance = np.linalg.norm(self.usv_positions[usv_idx] - self.tasks['coords'][task_idx])
            min_distance = min(all_distances)

            if current_distance <= min_distance * 1.2:  # 选择了较近的任务
                distance_bonus = 10.0
            else:
                distance_bonus = -5.0 * (current_distance / min_distance - 1.2)
        else:
            distance_bonus = 0.0

        # ========== 最终任务奖励 ==========
        if len(self.scheduled_tasks) == self.num_tasks:
            # 完成所有任务
            final_makespan = self.current_makespan

            if final_makespan < self.initial_makespan_estimate * 1.5:
                completion_bonus = 500.0  # 优秀完成
            elif final_makespan < self.initial_makespan_estimate * 2.0:
                completion_bonus = 200.0  # 良好完成
            else:
                completion_bonus = 50.0  # 一般完成

            # 负载均衡奖励
            final_task_counts = np.bincount(
                self.task_assignment[self.task_assignment != -1],
                minlength=self.num_usvs
            )
            final_std = np.std(final_task_counts)

            if final_std < 1.0:
                completion_bonus += 200.0  # 极好的均衡
            elif final_std < 2.0:
                completion_bonus += 100.0  # 良好的均衡
        else:
            completion_bonus = 0.0

        # ========== 计算总奖励 ==========
        total_reward = base_reward + usv_selection_bonus + balance_bonus + distance_bonus + completion_bonus

        # 调试输出（每5个任务输出一次）
        if self.debug_mode and len(self.scheduled_tasks) % 5 == 0:
            print(f"\n📊 任务{task_idx} -> USV{usv_idx} (第{len(self.scheduled_tasks)}个任务)")
            print(f"  Makespan: {self.last_makespan:.1f} -> {self.current_makespan:.1f}")
            print(f"  基础奖励: {base_reward:.1f}")
            print(f"  USV选择: {usv_selection_bonus:.1f}")
            print(f"  均衡奖励: {balance_bonus:.1f}")
            print(f"  距离奖励: {distance_bonus:.1f}")
            print(f"  总奖励: {total_reward:.1f}")
            print(f"  USV完成时间: {self.usv_next_available_time}")

        return total_reward

    def _generate_info(self, usv_idx, task_idx, done):
        """生成信息字典"""
        info = {
            'makespan': self.current_makespan,
            'scheduled_tasks_count': len(self.scheduled_tasks),
            'usv_battery': self.usv_batteries[usv_idx],
            'action_taken': usv_idx * self.num_tasks + task_idx,
            'task_completion_rate': len(self.scheduled_tasks) / self.num_tasks
        }

        if done:
            info['final_makespan'] = np.max(self.makespan_batch)
            info['avg_completion_time'] = np.mean(self.makespan_batch[self.scheduled_tasks])

        return info

    def _get_observation(self):
        """生成环境观测值"""
        usv_features = self._get_usv_features()
        task_features = self._get_task_features()
        edge_features = self._get_edge_features()
        action_mask = self._get_action_mask()

        return {
            'usv_features': usv_features.astype(np.float32),
            'task_features': task_features.astype(np.float32),
            'edge_features': edge_features.astype(np.float32),
            'action_mask': action_mask
        }

    def _get_usv_features(self):
        """获取USV特征 - 增加时间特征"""
        # 添加完成时间作为重要特征
        max_time = max(np.max(self.usv_next_available_time), 1.0)

        return np.column_stack([
            self.usv_positions / 500.0,
            self.usv_batteries / self.battery_capacity,
            self.usv_speeds / np.max(self.speed_range),
            self.usv_next_available_time / max_time  # 归一化的完成时间
        ])

    def _get_task_features(self):
        """获取任务特征"""
        task_features = np.zeros((self.num_tasks, 6))

        task_features[:, :2] = self.tasks['coords'] / 500.0

        proc_times = self.tasks['processing_time']
        if isinstance(proc_times[0], (list, np.ndarray)):
            task_features[:, 2:5] = proc_times / 100.0
        else:
            task_features[:, 2:5] = np.column_stack([proc_times, proc_times, proc_times]) / 100.0

        for task_idx in range(self.num_tasks):
            if task_idx not in self.scheduled_tasks:
                task_features[task_idx, 5] = 1.0

        return task_features

    def _get_edge_features(self):
        """获取边特征（距离矩阵）"""
        distances = np.zeros((self.num_usvs, self.num_tasks))

        for i in range(self.num_usvs):
            for j in range(self.num_tasks):
                if j not in self.scheduled_tasks:
                    distances[i, j] = np.linalg.norm(
                        self.usv_positions[i] - self.tasks['coords'][j]
                    )

        return distances

    def _get_action_mask(self):
        """获取动作掩码"""
        action_mask = np.zeros(self.num_usvs * self.num_tasks, dtype=np.bool_)

        for task_idx in range(self.num_tasks):
            if task_idx not in self.scheduled_tasks:
                for usv_idx in range(self.num_usvs):
                    action_idx = usv_idx * self.num_tasks + task_idx
                    action_mask[action_idx] = True

        return action_mask

    def get_valid_actions(self):
        """获取当前有效的动作列表"""
        valid_actions = []
        for task_idx in range(self.num_tasks):
            if task_idx not in self.scheduled_tasks:
                for usv_idx in range(self.num_usvs):
                    action = usv_idx * self.num_tasks + task_idx
                    valid_actions.append((action, usv_idx, task_idx))
        return valid_actions

    def render(self, mode='human'):
        """可视化环境状态"""
        if mode == 'human':
            print(f"\n{'=' * 50}")
            print(f"环境状态 - 时间: {self.current_time:.2f}")
            print(f"{'=' * 50}")
            print(f"已调度任务: {len(self.scheduled_tasks)}/{self.num_tasks}")
            print(f"当前makespan: {self.current_makespan:.2f}")
            print(f"Makespan变化: {self.last_makespan:.2f} -> {self.current_makespan:.2f}")

            print(f"\nUSV状态:")
            for i in range(self.num_usvs):
                print(f"  USV {i}: 位置{self.usv_positions[i]}, "
                      f"电量{self.usv_batteries[i]:.1f}, "
                      f"下次可用{self.usv_next_available_time[i]:.2f}")

            print(f"\n任务分配统计:")
            from collections import Counter
            assignments = Counter(self.task_assignment[self.task_assignment != -1])
            for usv_idx in range(self.num_usvs):
                count = assignments.get(usv_idx, 0)
                print(f"  USV {usv_idx}: {count} 个任务")

    def close(self):
        """清理资源"""
        pass
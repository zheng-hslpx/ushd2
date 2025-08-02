import gym
from gym import spaces
import numpy as np
from utils.data_generator import generate_task_instance
import logging


class USVSchedulingEnv(gym.Env):
    """
    优化后的USV调度环境
    主要改进：
    1. 简化奖励函数
    2. 优化动作掩码逻辑
    3. 清理冗余代码
    4. 增强错误处理
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

        # 定义观测空间和动作空间
        self.observation_space = spaces.Dict({
            'usv_features': spaces.Box(low=0, high=1, shape=(num_usvs, 4)),  # [x, y, battery, speed]
            'task_features': spaces.Box(low=0, high=100, shape=(num_tasks, 6)),  # [x, y, t1, t2, t3, is_pending]
            'edge_features': spaces.Box(low=0, high=100, shape=(num_usvs, num_tasks)),  # 距离矩阵
            'action_mask': spaces.Box(low=0, high=1, shape=(num_usvs * num_tasks,), dtype=np.bool_)
        })

        self.action_space = spaces.Discrete(num_usvs * num_tasks)

        # 奖励参数 - 简化为makespan奖励
        self.reward_config = {
            'use_simple_makespan_reward': True,  # 使用简单的makespan奖励
            # 备用：如果需要可以添加其他奖励组件
            # 'completion_bonus': 10.0,
            # 'efficiency_weight': 0.1,
        }

        # 初始化状态变量
        self.tasks = None  # 将由reset或reset_with_instances设置
        self.usvs = None  # 将由reset或reset_with_instances设置
        self._initialize_state_variables()

    def _initialize_state_variables(self):
        """初始化状态变量（不包括tasks和usvs，它们由外部设置）"""
        # 注意：不重置 self.tasks 和 self.usvs，它们由reset_with_instances设置
        self.scheduled_tasks = []
        self.current_time = 0.0
        self.usv_positions = np.zeros((self.num_usvs, 2))
        self.usv_batteries = np.full(self.num_usvs, self.battery_capacity)
        self.usv_speeds = np.zeros(self.num_usvs)
        self.usv_next_available_time = np.zeros(self.num_usvs)
        self.makespan_batch = np.zeros(self.num_tasks)
        self.task_assignment = np.full(self.num_tasks, -1, dtype=int)
        self.task_schedule_details = {}
        self.last_makespan = 0.0

    def reset(self):
        """重置环境为初始状态"""
        # 生成默认任务和USV数据
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

        return self._get_observation()

    def _setup_usv_speeds(self, speed_data):
        """设置USV速度，处理长度不匹配的情况"""
        speed_array = np.array(speed_data)

        if len(speed_array) >= self.num_usvs:
            self.usv_speeds = speed_array[:self.num_usvs]
        else:
            # 用平均速度填充不足的部分
            default_speed = np.mean(self.speed_range)
            self.usv_speeds = np.pad(
                speed_array,
                (0, self.num_usvs - len(speed_array)),
                constant_values=default_speed
            )

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

        # 检查索引范围
        if usv_idx >= self.num_usvs or task_idx >= self.num_tasks:
            return False

        # 检查任务是否已调度
        if task_idx in self.scheduled_tasks:
            return False

        return True

    def _handle_invalid_action(self, action):
        """处理无效动作"""
        logging.warning(f"无效动作: {action}")
        # 返回惩罚奖励，不改变状态
        return self._get_observation(), -100.0, False, {'invalid_action': True}

    def _execute_scheduling(self, usv_idx, task_idx):
        """执行调度逻辑"""
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

        # 计算奖励
        reward = self._calculate_reward(usv_idx, task_idx, distance, processing_time)

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

        # 更新电量
        battery_consumption = distance * 0.1  # 电量消耗系数
        self.usv_batteries[usv_idx] = max(0, self.usv_batteries[usv_idx] - battery_consumption)

        # 更新可用时间
        self.usv_next_available_time[usv_idx] = end_time

        # 更新全局时间
        self.current_time = np.max(self.usv_next_available_time)

    def _record_task_assignment(self, task_idx, usv_idx, travel_start, travel_time,
                                processing_start, processing_time):
        """记录任务分配详情"""
        # 更新分配记录
        self.task_assignment[task_idx] = usv_idx
        self.scheduled_tasks.append(task_idx)

        # 更新完成时间
        completion_time = processing_start + processing_time
        self.makespan_batch[task_idx] = completion_time

        # 记录详细信息
        self.task_schedule_details[task_idx] = {
            'task_idx': task_idx,
            'usv_idx': usv_idx,
            'travel_start_time': travel_start,
            'travel_time': travel_time,
            'processing_start_time': processing_start,
            'processing_time': processing_time
        }

        logging.debug(f"任务 {task_idx} 分配给 USV {usv_idx}, 完成时间: {completion_time:.2f}")

    def _calculate_reward(self, usv_idx, task_idx, distance, processing_time):
        """
        计算奖励 - 基于makespan变化的简单奖励
        reward = last_makespan - current_makespan

        这样保证：
        - makespan减少时，reward > 0 (好的)
        - makespan增加时，reward < 0 (坏的)
        - makespan不变时，reward = 0 (中性的)
        - reward和makespan幅度一致且为相反数关系
        """
        # 计算当前makespan
        current_makespan = np.max(self.usv_next_available_time)

        # 核心奖励逻辑：makespan改进量
        reward = self.last_makespan - current_makespan

        # 更新last_makespan用于下次计算
        self.last_makespan = current_makespan

        return reward

    def _generate_info(self, usv_idx, task_idx, done):
        """生成信息字典"""
        current_makespan = np.max(self.usv_next_available_time)

        info = {
            'makespan': current_makespan,
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
        # USV特征
        usv_features = self._get_usv_features()

        # 任务特征
        task_features = self._get_task_features()

        # 距离特征
        edge_features = self._get_edge_features()

        # 动作掩码
        action_mask = self._get_action_mask()

        return {
            'usv_features': usv_features.astype(np.float32),
            'task_features': task_features.astype(np.float32),
            'edge_features': edge_features.astype(np.float32),
            'action_mask': action_mask
        }

    def _get_usv_features(self):
        """获取USV特征"""
        return np.column_stack([
            self.usv_positions / 500.0,  # 归一化位置
            self.usv_batteries / self.battery_capacity,  # 归一化电量
            self.usv_speeds / np.max(self.speed_range)  # 归一化速度
        ])

    def _get_task_features(self):
        """获取任务特征"""
        task_features = np.zeros((self.num_tasks, 6))

        # 位置特征
        task_features[:, :2] = self.tasks['coords'] / 500.0

        # 处理时间特征
        proc_times = self.tasks['processing_time']
        if isinstance(proc_times[0], (list, np.ndarray)):
            task_features[:, 2:5] = proc_times / 100.0
        else:
            task_features[:, 2:5] = np.column_stack([proc_times, proc_times, proc_times]) / 100.0

        # 待处理标记
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

        # 只有未调度的任务对应的动作才有效
        for task_idx in range(self.num_tasks):
            if task_idx not in self.scheduled_tasks:
                for usv_idx in range(self.num_usvs):
                    action_idx = usv_idx * self.num_tasks + task_idx
                    action_mask[action_idx] = True

        return action_mask

    def get_valid_actions(self):
        """获取当前有效的动作列表（调试用）"""
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
            print(f"当前makespan: {np.max(self.usv_next_available_time):.2f}")

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

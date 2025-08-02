import numpy as np
from scipy.spatial.distance import cdist
import pickle


def generate_task_instance(num_tasks, num_usvs, area_size_x=(0, 500), area_size_y=(0, 500),
                           processing_time_range=(45, 90), battery_capacity=100, speed_range=(1, 3), charge_time=5):
    """生成固定数量的任务和USV初始状态"""
    # 生成固定数量的任务坐标和处理时间（三角模糊时间）
    tasks = {
        'coords': np.column_stack([
            np.random.uniform(area_size_x[0], area_size_x[1], num_tasks),
            np.random.uniform(area_size_y[0], area_size_y[1], num_tasks)
        ]),  # 固定形状：(num_tasks, 2)
        'processing_time': np.random.randint(processing_time_range[0], processing_time_range[1],
                                             size=(num_tasks, 3))  # 固定形状：(num_tasks, 3) [t1, t2, t3]
    }

    # 生成固定数量的USV初始状态
    usvs = {
        'coords': np.zeros((num_usvs, 2)),  # 固定形状：(num_usvs, 2)
        'battery': np.full(num_usvs, battery_capacity),  # 固定形状：(num_usvs,)
        'speed': np.random.uniform(speed_range[0], speed_range[1], num_usvs),  # 固定形状：(num_usvs,)
        'charge_time': charge_time
    }
    return tasks, usvs


def generate_batch_instances(num_instances, fixed_tasks=30, fixed_usvs=3, area_size_x=(0, 500), area_size_y=(0, 500),
                             processing_time_range=(45, 90), battery_capacity=100, speed_range=(1, 3), charge_time=5):
    """生成批量算例，任务和USV数量固定"""
    instances = []
    for _ in range(num_instances):
        # 强制使用固定数量的任务和USV（与环境参数匹配）
        tasks, usvs = generate_task_instance(
            num_tasks=fixed_tasks,
            num_usvs=fixed_usvs,
            area_size_x=area_size_x,
            area_size_y=area_size_y,
            processing_time_range=processing_time_range,
            battery_capacity=battery_capacity,
            speed_range=speed_range,
            charge_time=charge_time
        )
        instances.append((tasks, usvs))
    return instances


def save_instances_to_file(instances, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(instances, f)


def load_instances_from_file(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

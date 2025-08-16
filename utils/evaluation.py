import numpy as np
from collections import defaultdict

def evaluate_scheduling_result(env):
    """
    基于环境最终状态评估调度结果

    Args:
        env: 训练后的USV调度环境

    Returns:
        dict: 包含各种评估指标的字典
    """
    if not env.scheduled_tasks:
        return {
            'total_makespan': float('inf'),
            'avg_completion_time': float('inf'),
            'task_completion_rate': 0.0,
            'usv_utilization': {},
            'load_balance_std': float('inf'),
            'efficiency_score': 0.0
        }

    # 1. 基本时间指标
    completed_times = env.makespan_batch[env.scheduled_tasks]
    total_makespan = np.max(completed_times)
    avg_completion_time = np.mean(completed_times)

    # 2. 任务完成率
    task_completion_rate = len(env.scheduled_tasks) / env.num_tasks

    # 3. USV利用率分析
    usv_utilization = {}
    usv_task_counts = defaultdict(int)
    usv_work_times = defaultdict(float)

    # 分析每个USV的工作负载
    for task_idx in env.scheduled_tasks:
        usv_idx = env.task_assignment[task_idx]
        if usv_idx != -1:
            usv_task_counts[usv_idx] += 1
            if task_idx in env.task_schedule_details:
                details = env.task_schedule_details[task_idx]
                work_time = details['travel_time'] + details['processing_time']
                usv_work_times[usv_idx] += work_time

    # 计算每个USV的利用率
    for usv_idx in range(env.num_usvs):
        total_work_time = usv_work_times.get(usv_idx, 0)
        utilization_rate = total_work_time / total_makespan if total_makespan > 0 else 0
        usv_utilization[f'USV_{usv_idx}'] = {
            'task_count': usv_task_counts.get(usv_idx, 0),
            'work_time': total_work_time,
            'utilization_rate': utilization_rate
        }

    # 4. 负载均衡度（标准差越小越好）
    task_counts = [usv_task_counts.get(i, 0) for i in range(env.num_usvs)]
    work_times = [usv_work_times.get(i, 0) for i in range(env.num_usvs)]

    load_balance_std = {
        'task_count_std': np.std(task_counts),
        'work_time_std': np.std(work_times)
    }

    # 5. 效率得分（综合指标）
    # 考虑完成率、时间效率、负载均衡
    time_efficiency = 1.0 / (1.0 + total_makespan / 1000.0)  # 归一化时间效率
    balance_score = 1.0 / (1.0 + np.std(work_times))  # 负载均衡得分
    efficiency_score = task_completion_rate * 0.5 + time_efficiency * 0.3 + balance_score * 0.2

    return {
        'total_makespan': total_makespan,
        'avg_completion_time': avg_completion_time,
        'task_completion_rate': task_completion_rate,
        'usv_utilization': usv_utilization,
        'load_balance_std': load_balance_std,
        'efficiency_score': efficiency_score,
        'detailed_metrics': {
            'min_completion_time': np.min(completed_times),
            'max_completion_time': np.max(completed_times),
            'completion_time_std': np.std(completed_times),
            'total_travel_distance': _calculate_total_travel_distance(env),
            'avg_usv_battery_remaining': np.mean(env.usv_batteries)
        }
    }


def _calculate_total_travel_distance(env):
    """计算总旅行距离"""
    total_distance = 0.0

    for task_idx in env.scheduled_tasks:
        if task_idx in env.task_schedule_details:
            details = env.task_schedule_details[task_idx]
            usv_idx = details['usv_idx']
            travel_time = details['travel_time']
            # 距离 = 旅行时间 * 速度
            distance = travel_time * env.usv_speeds[usv_idx]
            total_distance += distance

    return total_distance


def compare_scheduling_methods(results_dict):
    """
    比较不同调度方法的性能

    Args:
        results_dict: {method_name: evaluation_result, ...}

    Returns:
        dict: 排序后的比较结果
    """
    if not results_dict:
        return {}

    # 定义比较指标（越小越好的指标用负号）
    comparison_metrics = {
        'total_makespan': 'min',  # 越小越好
        'task_completion_rate': 'max',  # 越大越好
        'efficiency_score': 'max',  # 越大越好
        'load_balance_std.work_time_std': 'min'  # 越小越好
    }

    rankings = {}

    for metric, direction in comparison_metrics.items():
        method_scores = []

        for method_name, result in results_dict.items():
            # 处理嵌套指标
            if '.' in metric:
                keys = metric.split('.')
                value = result
                for key in keys:
                    value = value.get(key, float('inf') if direction == 'min' else 0)
            else:
                value = result.get(metric, float('inf') if direction == 'min' else 0)

            method_scores.append((method_name, value))

        # 排序
        reverse = (direction == 'max')
        method_scores.sort(key=lambda x: x[1], reverse=reverse)
        rankings[metric] = method_scores

    return rankings


def print_evaluation_report(evaluation_result, method_name="Current Method"):
    """打印详细的评估报告"""
    print(f"\n{'=' * 60}")
    print(f"调度方法评估报告: {method_name}")
    print(f"{'=' * 60}")

    print(f"\n📊 基本性能指标:")
    print(f"  总完成时间 (Makespan): {evaluation_result['total_makespan']:.2f}")
    print(f"  平均完成时间: {evaluation_result['avg_completion_time']:.2f}")
    print(f"  任务完成率: {evaluation_result['task_completion_rate']:.1%}")
    print(f"  效率得分: {evaluation_result['efficiency_score']:.3f}")

    print(f"\n🚢 USV利用率分析:")
    for usv_name, util_data in evaluation_result['usv_utilization'].items():
        print(f"  {usv_name}:")
        print(f"    任务数量: {util_data['task_count']}")
        print(f"    工作时间: {util_data['work_time']:.2f}")
        print(f"    利用率: {util_data['utilization_rate']:.1%}")

    print(f"\n⚖️ 负载均衡:")
    balance = evaluation_result['load_balance_std']
    print(f"  任务数量标准差: {balance['task_count_std']:.2f}")
    print(f"  工作时间标准差: {balance['work_time_std']:.2f}")

    print(f"\n📈 详细指标:")
    details = evaluation_result['detailed_metrics']
    print(f"  最短完成时间: {details['min_completion_time']:.2f}")
    print(f"  最长完成时间: {details['max_completion_time']:.2f}")
    print(f"  完成时间标准差: {details['completion_time_std']:.2f}")
    print(f"  总旅行距离: {details['total_travel_distance']:.2f}")
    print(f"  平均剩余电量: {details['avg_usv_battery_remaining']:.1f}")


# 使用示例
def evaluate_trained_model(env_after_episode):
    """
    评估训练后的模型性能
    在训练脚本的episode结束后调用
    """
    result = evaluate_scheduling_result(env_after_episode)
    print_evaluation_report(result, "PPO Trained Model")
    return result
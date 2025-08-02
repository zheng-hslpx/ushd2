import os
import time
import random
import torch
import numpy as np
from tqdm import tqdm  # 用于显示训练进度条
from torch.utils.tensorboard import SummaryWriter
from env.usv_env import USVSchedulingEnv  # 导入修改后的环境类
from env.state_representation import build_heterogeneous_graph, calculate_usv_task_distances
from graph.hgnn import USVHeteroGNN  
from PPO_model import PPO, Memory  
import visdom
import utils.data_generator as data_generator
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
import logging

# 配置根 logger
logging.basicConfig(
    level=logging.INFO,  # 修改：从DEBUG改为INFO减少日志输出
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# 环境参数
num_usvs = 3  # USV数量
num_tasks = 30  # 任务数量
num_instances = 10  # 算例数量

# 训练参数配置
TRAINING_CONFIG = {
    'max_episodes': 1000,  # 最大训练回合数
    'max_steps_per_episode': num_tasks * 10,  # 每个训练回合的最大步数
    'early_stop_patience': 1000,  # 早停耐心值
    'seed': 42,  # 随机种子
    'eta': 2,  # 构建异构图时考虑的最近邻任务数量
}

# HGNN (异构图神经网络) 参数配置
HGNN_CONFIG = {
    'hidden_dim': 128,  # HGNN中节点和边的隐藏层特征维度
    'n_heads': 4,  # GAT 层中的注意力头数量
    'num_layers': 2,  # HGNN中GAT层的数量
    'dropout': 0.1,  # Dropout率
}

# PPO (近端策略优化) 算法参数配置
PPO_CONFIG = {
    'lr': 3e-4,  # 学习率
    'gamma': 0.98,  # 折扣因子
    'eps_clip': 0.2,  # PPO裁剪参数
    'K_epochs': 10,  # 每次策略更新时的优化轮数
    'entropy_coef': 0.02,  # 熵正则化系数
    'value_coef': 0.5,  # 价值函数损失的系数
}


def setup_seed(seed):
    """设置随机种子确保结果可复现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# ========== 修复2: 移除重复的函数定义 ==========
def get_next_model_number(model_dir):
    """获取下一个可用的模型编号"""
    if not os.path.exists(model_dir):
        return 0

    existing_models = [f for f in os.listdir(model_dir) if f.startswith('ppo_best_') and f.endswith('.pt')]
    numbers = []
    for model in existing_models:
        try:
            number = int(model.split('_')[-1].split('.')[0])
            numbers.append(number)
        except ValueError:
            continue
    if numbers:
        return max(numbers) + 1
    return 0


def generate_gantt_chart(env):
    """生成并显示甘特图，为每个调度的任务添加任务编号标识"""
    if not env.scheduled_tasks:
        print("警告：没有已调度的任务，无法生成甘特图。")
        return

    fig, ax = plt.subplots(figsize=(15, 8))

    # 为每个 USV 生成唯一颜色
    num_usvs = env.num_usvs
    hues = np.linspace(0, 1, num_usvs, endpoint=False)
    usv_colors_list = [mcolors.hsv_to_rgb((h, 0.8, 0.8)) for h in hues]

    # 用于存储每个 USV 的任务及其时间信息
    usv_task_data = {i: [] for i in range(env.num_usvs)}

    # 检查是否有任务调度详情
    if not hasattr(env, 'task_schedule_details') or not env.task_schedule_details:
        print("警告：环境未存储任务调度详情，甘特图将不包含航行时间。")
        return

    # 填充 usv_task_data
    for task_idx in env.scheduled_tasks:
        if task_idx in env.task_schedule_details:
            details = env.task_schedule_details[task_idx]
            try:
                task_idx = details['task_idx']
                usv_idx = details['usv_idx']
                usv_task_data[usv_idx].append(details)
            except KeyError as e:
                print(f"Warning: Task {task_idx} details missing key {e}, skipping in Gantt chart.")
        else:
            print(f"Warning: Task {task_idx} details not found in env.task_schedule_details, skipping in Gantt chart.")

    # 对每个 USV 上的任务按处理开始时间排序
    for usv_idx in usv_task_data:
        if usv_task_data[usv_idx]:
            usv_task_data[usv_idx].sort(key=lambda x: x['processing_start_time'])
        else:
            print(f"Warning: No tasks found for USV {usv_idx}, skipping sorting.")

    # 绘制每个任务的条形图
    y_labels = []
    y_positions = []
    y_spacing = 1.5
    bar_height = 0.4

    for usv_idx, tasks_data in usv_task_data.items():
        y_pos = usv_idx * y_spacing
        y_labels.append(f'USV {usv_idx}')
        y_positions.append(y_pos)

        if not tasks_data:
            print(f"Warning: No task data to plot for USV {usv_idx}.")
            continue

        current_usv_color = usv_colors_list[usv_idx]

        for task_data in tasks_data:
            try:
                task_idx = task_data['task_idx']
                processing_start_time = task_data['processing_start_time']
                processing_time = task_data['processing_time']
                travel_start_time = task_data['travel_start_time']
                travel_time = task_data['travel_time']

                # 绘制航行时间条
                if travel_time > 0:
                    ax.barh(y_pos, travel_time, left=travel_start_time, height=bar_height,
                            color='gray', label='Navigation' if usv_idx == 0 and task_idx == usv_task_data[usv_idx][0][
                            'task_idx'] else "")

                # 绘制处理时间条
                ax.barh(y_pos, processing_time, left=processing_start_time, height=bar_height, color=current_usv_color)

                # 添加任务编号
                ax.text(processing_start_time + processing_time / 2, y_pos, f'{task_idx}',
                        ha='center', va='center', fontsize=9, color='black', weight='bold')
            except KeyError as e:
                print(f"Error plotting task for USV {usv_idx}: Missing key {e} in task_data {task_data}")
            except Exception as e:
                print(f"Unexpected error plotting task for USV {usv_idx}: {e}")

    # 设置坐标轴
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel('Time')
    ax.set_ylabel('USV')
    ax.set_title('Gantt Chart of USV Task Scheduling')

    # 添加图例
    handles, labels = ax.get_legend_handles_labels()
    if 'Navigation' in labels:
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='gray', label='Navigation')]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))

    fig.tight_layout()
    plt.show()
    print("甘特图已生成并显示。")


def main():
    """主训练函数"""
    # 设置随机种子
    setup_seed(TRAINING_CONFIG['seed'])

    # 设备配置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | {'GPU加速已启用' if device.type == 'cuda' else '使用CPU'}")

    # 数据准备
    print("准备训练数据...")
    instances = data_generator.generate_batch_instances(
        num_instances=num_instances,
        fixed_tasks=num_tasks,
        fixed_usvs=num_usvs
    )
    file_path = "data/fixed_instances.pkl"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    data_generator.save_instances_to_file(instances, file_path)
    print(f"生成并保存了{len(instances)}个训练算例")

    # 初始化环境
    print("初始化环境...")
    env = USVSchedulingEnv(num_usvs=num_usvs, num_tasks=num_tasks)

    # 模型初始化
    print("初始化模型...")
    # 获取特征维度
    state = env.reset()
    usv_feats = state['usv_features']
    task_feats = state['task_features']
    distances = calculate_usv_task_distances(usv_feats[:, :2], task_feats[:, :2])
    graph = build_heterogeneous_graph(usv_feats, task_feats, distances, eta=TRAINING_CONFIG['eta'])
    graph = graph.to(device)

    print(f"图中节点类型: {graph.ntypes}")
    print(f"图中完整边类型三元组: {graph.canonical_etypes}")
    for etype in graph.canonical_etypes:
        src_type, edge_name, dst_type = etype
        print(f"边类型 '{edge_name}': {src_type} → {dst_type}, 边数量: {graph.num_edges(etype)}")

    # 初始化HGNN模型
    hgnn = USVHeteroGNN(
        usv_feat_dim=usv_feats.shape[1],
        task_feat_dim=task_feats.shape[1],
        hidden_dim=HGNN_CONFIG['hidden_dim'],
        n_heads=HGNN_CONFIG['n_heads'],
        num_layers=HGNN_CONFIG['num_layers'],
        dropout=HGNN_CONFIG['dropout']
    ).to(device)

    print(f"HGNN初始化完成 - USV特征维度: {usv_feats.shape[1]}, 任务特征维度: {task_feats.shape[1]}")

    # 初始化PPO代理
    action_dim = num_usvs * num_tasks
    ppo = PPO(
        hgnn=hgnn,
        action_dim=action_dim,
        lr=PPO_CONFIG['lr'],
        gamma=PPO_CONFIG['gamma'],
        eps_clip=PPO_CONFIG['eps_clip'],
        K_epochs=PPO_CONFIG['K_epochs'],
        device=device,
        entropy_coef=PPO_CONFIG['entropy_coef'],
        value_coef=PPO_CONFIG['value_coef']
    )

    print(f"PPO初始化完成 - 动作维度: {action_dim}")

    # 日志和可视化设置
    print("设置日志和可视化...")
    # 创建模型保存目录
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    next_model_number = get_next_model_number(model_dir)

    # 初始化TensorBoard
    writer = SummaryWriter("runs/usv_scheduling")

    # 初始化visdom（可选）
    vis = visdom.Visdom()
    if not vis.check_connection():
        print("Warning: Visdom server not connected. Plots will not be displayed.")
        vis = None

    reward_window = None
    makespan_window = None
    policy_loss_window = None
    value_loss_window = None

    if vis:
        try:
            reward_window = vis.line(
                Y=torch.zeros((1)).cpu(),
                X=torch.zeros((1)).cpu(),
                opts=dict(
                    xlabel='Episode',
                    ylabel='Reward',
                    title='Training Reward',
                    ylim=[-3200, 0],
                    ytickmarks=list(range(-3200, 1, 50))
                )
            )
            makespan_window = vis.line(
                Y=torch.zeros((1)).cpu(),
                X=torch.zeros((1)).cpu(),
                opts=dict(
                    xlabel='Episode',
                    ylabel='Makespan',
                    title='Training Makespan'
                )
            )
            policy_loss_window = vis.line(
                Y=torch.zeros((1)).cpu(),
                X=torch.zeros((1)).cpu(),
                opts=dict(
                    xlabel='Episode',
                    ylabel='Loss',
                    title='Policy Loss'
                )
            )
            value_loss_window = vis.line(
                Y=torch.zeros((1)).cpu(),
                X=torch.zeros((1)).cpu(),
                opts=dict(
                    xlabel='Episode',
                    ylabel='Loss',
                    title='Value Loss'
                )
            )
            print("Visdom可视化初始化成功")
        except Exception as e:
            print(f"Warning: Failed to create Visdom windows: {e}")
            vis = None

    # ========== 修复3: 正确的缩进层级 ==========
    # 训练统计
    best_makespan = float('inf')
    no_improve_count = 0
    start_time = time.time()
    instance_index = 0

    print("开始训练...")
    # 主训练循环
    pbar = tqdm(range(TRAINING_CONFIG['max_episodes']), desc="训练进度", unit="episode")
    for episode in pbar:
        # 重置环境
        tasks, usvs = instances[instance_index % num_instances]
        instance_index += 1
        state = env.reset_with_instances(tasks, usvs)

        memory = Memory()  # 使用新的Memory类
        done = False
        total_reward = 0
        steps = 0
        episode_makespan = 0

        while not done and steps < TRAINING_CONFIG['max_steps_per_episode']:
            # 构建异构图状态
            usv_feats = state['usv_features']
            task_feats = state['task_features']
            distances = calculate_usv_task_distances(usv_feats[:, :2], task_feats[:, :2])
            graph = build_heterogeneous_graph(usv_feats, task_feats, distances, eta=TRAINING_CONFIG['eta'])
            graph = graph.to(device)
            graph.action_mask = state['action_mask']

            # 选择动作 - 使用新的PPO接口
            action, log_prob, state_value = ppo.select_action(graph)

            # 执行动作
            next_state, reward, done, info = env.step(action)

            # 处理episode结束时的奖励
            if done and 'final_makespan' in info:
                episode_makespan = info['final_makespan']
                sparse_reward = -episode_makespan * 0.1
                reward += sparse_reward
            elif not done:
                episode_makespan = info.get("makespan", episode_makespan)

            # 存储经验
            memory.states.append(graph)
            memory.actions.append(action)
            memory.logprobs.append(log_prob)
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            memory.state_values.append(state_value)

            # 更新状态和统计
            state = next_state
            total_reward += reward
            steps += 1

        # 更新PPO策略 - 使用新的返回值
        policy_loss_avg, value_loss_avg, entropy_loss_avg = ppo.update(memory)

        # 性能统计
        makespan = episode_makespan if episode_makespan > 0 else float('inf')

        # 早停检查与模型保存
        if makespan < best_makespan:
            best_makespan = makespan
            no_improve_count = 0
            model_name = f"ppo_best_{next_model_number}.pt"
            model_path = os.path.join(model_dir, model_name)
            torch.save(ppo.policy.state_dict(), model_path)
            pbar.write(f"Episode {episode}: 新的最佳完成时间 {best_makespan:.4f}，已保存模型至 {model_path}")
        else:
            no_improve_count += 1
            if no_improve_count >= TRAINING_CONFIG['early_stop_patience']:
                pbar.write(f"早停触发：连续{TRAINING_CONFIG['early_stop_patience']}轮未改进最佳完成时间")
                break

        # 记录日志
        writer.add_scalar("Reward/Episode", total_reward, episode)
        writer.add_scalar("Makespan/Episode", makespan, episode)
        writer.add_scalar("Steps/Episode", steps, episode)
        writer.add_scalar("Policy_Loss/Episode", policy_loss_avg, episode)
        writer.add_scalar("Value_Loss/Episode", value_loss_avg, episode)
        writer.add_scalar("Entropy_Loss/Episode", entropy_loss_avg, episode)

        # 更新可视化
        if vis and all([reward_window, makespan_window, policy_loss_window, value_loss_window]):
            try:
                vis.line(Y=torch.tensor([total_reward]).cpu(), X=torch.tensor([episode]).cpu(),
                         win=reward_window, update='append')
                vis.line(Y=torch.tensor([makespan]).cpu(), X=torch.tensor([episode]).cpu(),
                         win=makespan_window, update='append')
                vis.line(Y=torch.tensor([policy_loss_avg]).cpu(), X=torch.tensor([episode]).cpu(),
                         win=policy_loss_window, update='append', name='Policy Loss')
                vis.line(Y=torch.tensor([value_loss_avg]).cpu(), X=torch.tensor([episode]).cpu(),
                         win=value_loss_window, update='append', name='Value Loss')
            except Exception as e:
                print(f"Warning: Failed to update Visdom plots: {e}")

        # 更新进度条
        pbar.set_postfix({
            "奖励": f"{total_reward:.2f}",
            "完成时间": f"{makespan:.2f}",
            "最佳时间": f"{best_makespan:.2f}",
            "耗时": f"{time.time() - start_time:.1f}s"
        })

    # 训练结束
    writer.close()
    print(f"训练完成！最佳完成时间: {best_makespan:.4f}")

    # 生成甘特图
    try:
        generate_gantt_chart(env)
    except Exception as e:
        print(f"生成甘特图时出错: {e}")


# ========== 修复4: 正确的main函数调用 ==========
if __name__ == "__main__":
    main()

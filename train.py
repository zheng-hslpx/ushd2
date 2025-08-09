import os
import time
import random
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from env.usv_env import USVSchedulingEnv
from env.state_representation import build_heterogeneous_graph, calculate_usv_task_distances
from graph.hgnn import StableUSVHeteroGNN
from PPO_model import PPO, Memory
from utils.evaluation import evaluate_scheduling_result, print_evaluation_report
import visdom
import utils.data_generator as data_generator
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# ========== 改进的训练配置 ==========

# 环境参数
num_usvs = 3
num_tasks = 30
num_instances = 100  # 增加训练数据多样性

# 训练参数配置
TRAINING_CONFIG = {
    'max_episodes': 100,  # 继续训练
    'max_steps_per_episode': num_tasks,
    'early_stop_patience': 300,
    'seed': 42,
    'eta': 7,  # 增加图连接度
    'warmup_episodes': 200,
    'eval_frequency': 50,
    'batch_episodes': 15,  # 增加批次大小
    'save_frequency': 100,

    # 新增：课程学习参数
    'curriculum_stages': [
        {'episodes': 500, 'focus': 'balance'},  # 阶段1：重视均衡
        {'episodes': 1000, 'focus': 'mixed'},   # 阶段2：平衡
        {'episodes': 1500, 'focus': 'makespan'} # 阶段3：重视效率
    ]
}

# HGNN参数配置
HGNN_CONFIG = {
    'hidden_dim': 256,
    'n_heads': 8,
    'num_layers': 3,
    'dropout': 0.1,  # 降低dropout
}

# PPO参数配置（改进版）
PPO_CONFIG = {
    'lr_actor': 2e-4,  # 稍微降低学习率
    'lr_critic': 8e-4,
    'gamma': 0.995,  # 增加远见
    'eps_clip': 0.15,  # 更保守的更新
    'K_epochs': 12,
    'entropy_coef': 0.008,  # 降低探索
    'value_coef': 1.2,
    'gae_lambda': 0.97,
}


def setup_seed(seed):
    """设置随机种子确保结果可复现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def adaptive_exploration(action, action_mask, episode, usv_completion_times):
    """基于USV完成时间的自适应探索"""
    explore_prob = max(0.02, 0.2 * (0.997 ** episode))

    if np.random.random() < explore_prob:
        valid_actions = np.where(action_mask)[0]
        if len(valid_actions) == 0:
            return action

        # 基于USV当前完成时间分配概率
        action_probs = np.ones(len(valid_actions))

        for i, valid_action in enumerate(valid_actions):
            usv_idx = valid_action // 30
            # 完成时间越早的USV，被选择概率越高
            completion_time_ranks = np.argsort(usv_completion_times)
            rank = np.where(completion_time_ranks == usv_idx)[0][0]
            action_probs[i] = 1.0 / (1.0 + rank)

        action_probs = action_probs / np.sum(action_probs)
        return np.random.choice(valid_actions, p=action_probs)

    return action


def collect_batch_episodes(env, ppo, instances, batch_size, device, config, episode_num):
    """
    批量收集多个episode的经验
    """
    batch_memory = Memory()
    batch_rewards = []
    batch_makespans = []
    batch_balances = []

    for _ in range(batch_size):
        # 随机选择一个实例
        instance_idx = np.random.randint(len(instances))
        tasks, usvs = instances[instance_idx]

        # 重置环境
        state = env.reset_with_instances(tasks, usvs)

        episode_reward = 0
        steps = 0
        done = False

        # 跟踪USV任务分配
        usv_task_counts = np.zeros(env.num_usvs)

        while not done and steps < config['max_steps_per_episode']:
            # 构建图
            usv_feats = state['usv_features']
            task_feats = state['task_features']
            distances = calculate_usv_task_distances(usv_feats[:, :2], task_feats[:, :2])
            graph = build_heterogeneous_graph(
                usv_feats, task_feats, distances,
                eta=config['eta'], device=device
            )
            graph.action_mask = state['action_mask']

            # 选择动作
            action, log_prob, state_value = ppo.select_action(graph)

            # 智能探索（考虑负载均衡）
            action = adaptive_exploration(
                action, state['action_mask'],
                episode_num, usv_task_counts
            )

            # 执行动作
            next_state, reward, done, info = env.step(action)

            # 更新USV任务计数
            usv_idx = action // num_tasks
            usv_task_counts[usv_idx] += 1

            # 存储经验
            batch_memory.states.append(graph)
            batch_memory.actions.append(action)
            batch_memory.logprobs.append(log_prob)
            batch_memory.rewards.append(reward)
            batch_memory.is_terminals.append(done)
            batch_memory.state_values.append(state_value)

            state = next_state
            episode_reward += reward
            steps += 1

        # 计算负载均衡度
        final_task_counts = np.bincount(
            env.task_assignment[env.task_assignment != -1],
            minlength=env.num_usvs
        )
        balance_std = np.std(final_task_counts)

        batch_rewards.append(episode_reward)
        batch_makespans.append(info.get('final_makespan', float('inf')))
        batch_balances.append(balance_std)

    return batch_memory, batch_rewards, batch_makespans, batch_balances


def evaluate_model(ppo, env, test_instances, device, config, episode):
    """
    评估模型性能
    """
    total_rewards = []
    total_makespans = []
    load_balances = []

    ppo.policy_old.eval()  # 设置为评估模式

    with torch.no_grad():
        for tasks, usvs in test_instances[:10]:  # 评估10个实例
            state = env.reset_with_instances(tasks, usvs)

            episode_reward = 0
            done = False
            steps = 0

            while not done and steps < config['max_steps_per_episode']:
                # 构建图
                usv_feats = state['usv_features']
                task_feats = state['task_features']
                distances = calculate_usv_task_distances(usv_feats[:, :2], task_feats[:, :2])
                graph = build_heterogeneous_graph(
                    usv_feats, task_feats, distances,
                    eta=config['eta'], device=device
                )
                graph.action_mask = state['action_mask']

                # 选择动作
                action, log_prob, state_value = ppo.select_action(graph)

                # 获取USV完成时间用于探索
                usv_completion_times = env.usv_next_available_time.copy()

                # 智能探索（基于USV完成时间）
                action = adaptive_exploration(
                    action, state['action_mask'],
                    episode, usv_completion_times
                )

                # 执行动作
                next_state, reward, done, info = env.step(action)

            # 计算负载均衡
            task_counts = np.bincount(
                env.task_assignment[env.task_assignment != -1],
                minlength=env.num_usvs
            )

            total_rewards.append(episode_reward)
            total_makespans.append(info.get('final_makespan', float('inf')))
            load_balances.append(np.std(task_counts))

    ppo.policy_old.train()  # 恢复训练模式

    return np.mean(total_rewards), np.mean(total_makespans), np.mean(load_balances)


def generate_gantt_chart(env, save_path=None):
    """生成并显示甘特图"""
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

    # 填充 usv_task_data
    for task_idx in env.scheduled_tasks:
        if task_idx in env.task_schedule_details:
            details = env.task_schedule_details[task_idx]
            usv_idx = details['usv_idx']
            usv_task_data[usv_idx].append(details)

    # 对每个 USV 上的任务按处理开始时间排序
    for usv_idx in usv_task_data:
        if usv_task_data[usv_idx]:
            usv_task_data[usv_idx].sort(key=lambda x: x['processing_start_time'])

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
            continue

        current_usv_color = usv_colors_list[usv_idx]

        for task_data in tasks_data:
            task_idx = task_data['task_idx']
            processing_start_time = task_data['processing_start_time']
            processing_time = task_data['processing_time']
            travel_start_time = task_data['travel_start_time']
            travel_time = task_data['travel_time']

            # 绘制航行时间条
            if travel_time > 0:
                ax.barh(y_pos, travel_time, left=travel_start_time, height=bar_height,
                        color='gray', alpha=0.5)

            # 绘制处理时间条
            ax.barh(y_pos, processing_time, left=processing_start_time, height=bar_height,
                    color=current_usv_color)

            # 添加任务编号
            ax.text(processing_start_time + processing_time / 2, y_pos, f'{task_idx}',
                    ha='center', va='center', fontsize=9, color='black', weight='bold')

    # 设置坐标轴
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel('Time')
    ax.set_ylabel('USV')
    ax.set_title('Gantt Chart of USV Task Scheduling')

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='gray', alpha=0.5, label='Navigation')]
    ax.legend(handles=legend_elements, loc='upper right')

    # 添加统计信息
    task_counts = np.bincount(
        env.task_assignment[env.task_assignment != -1],
        minlength=env.num_usvs
    )
    stats_text = f"Task Distribution: {task_counts} | Std: {np.std(task_counts):.2f}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"甘特图已保存到: {save_path}")

    plt.show()


def main():
    """主训练函数"""
    # 设置随机种子
    setup_seed(TRAINING_CONFIG['seed'])

    # 设备配置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建目录
    model_dir = "model"
    log_dir = "logs"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 初始化环境
    print("初始化环境和模型...")
    env = USVSchedulingEnv(num_usvs=num_usvs, num_tasks=num_tasks)

    # 获取特征维度
    temp_state = env.reset()
    usv_feat_dim = temp_state['usv_features'].shape[1]
    task_feat_dim = temp_state['task_features'].shape[1]
    action_dim = num_usvs * num_tasks

    print(f"特征维度 - USV: {usv_feat_dim}, Task: {task_feat_dim}, Action: {action_dim}")

    # 初始化HGNN模型
    hgnn = StableUSVHeteroGNN(
        usv_feat_dim=usv_feat_dim,
        task_feat_dim=task_feat_dim,
        hidden_dim=HGNN_CONFIG['hidden_dim'],
        n_heads=HGNN_CONFIG['n_heads'],
        num_layers=HGNN_CONFIG['num_layers'],
        dropout=HGNN_CONFIG['dropout']
    ).to(device)

    # 初始化PPO
    # 注意：如果您的PPO类支持分离的学习率，请使用以下参数
    # 如果不支持，请使用单一学习率
    try:
        ppo = PPO(
            hgnn=hgnn,
            action_dim=action_dim,
            lr_actor=PPO_CONFIG['lr_actor'],
            lr_critic=PPO_CONFIG['lr_critic'],
            gamma=PPO_CONFIG['gamma'],
            eps_clip=PPO_CONFIG['eps_clip'],
            K_epochs=PPO_CONFIG['K_epochs'],
            device=device,
            entropy_coef=PPO_CONFIG['entropy_coef'],
            value_coef=PPO_CONFIG['value_coef'],
            gae_lambda=PPO_CONFIG.get('gae_lambda', 0.95)
        )
    except TypeError:
        # 如果不支持分离学习率，使用单一学习率
        print("使用单一学习率配置")
        ppo = PPO(
            hgnn=hgnn,
            action_dim=action_dim,
            lr=PPO_CONFIG['lr_actor'],
            gamma=PPO_CONFIG['gamma'],
            eps_clip=PPO_CONFIG['eps_clip'],
            K_epochs=PPO_CONFIG['K_epochs'],
            device=device,
            entropy_coef=PPO_CONFIG['entropy_coef'],
            value_coef=PPO_CONFIG['value_coef']
        )

    # 初始化TensorBoard
    writer = SummaryWriter(f"runs/usv_scheduling_{time.strftime('%Y%m%d_%H%M%S')}")

    # 初始化Visdom（可选）
    vis = None
    try:
        vis = visdom.Visdom()
        if vis.check_connection():
            print("Visdom连接成功")
            # 初始化可视化窗口
            vis_windows = {
                'reward': vis.line(Y=torch.zeros((1)).cpu(), X=torch.zeros((1)).cpu(),
                                   opts=dict(xlabel='Episode', ylabel='Reward', title='Training Reward')),
                'makespan': vis.line(Y=torch.zeros((1)).cpu(), X=torch.zeros((1)).cpu(),
                                     opts=dict(xlabel='Episode', ylabel='Makespan', title='Training Makespan')),
                'balance': vis.line(Y=torch.zeros((1)).cpu(), X=torch.zeros((1)).cpu(),
                                    opts=dict(xlabel='Episode', ylabel='Std', title='Load Balance')),
                'losses': vis.line(Y=torch.zeros((1, 2)).cpu(), X=torch.zeros((1)).cpu(),
                                   opts=dict(xlabel='Episode', ylabel='Loss', title='Losses',
                                             legend=['Policy', 'Value']))
            }
        else:
            print("Visdom未连接，将跳过可视化")
            vis = None
    except Exception as e:
        print(f"Visdom初始化失败: {e}")
        vis = None

    # 生成训练数据
    print(f"生成{num_instances}个训练实例...")
    train_instances = data_generator.generate_batch_instances(
        num_instances=num_instances,
        fixed_tasks=num_tasks,
        fixed_usvs=num_usvs
    )

    # 训练统计
    all_rewards = []
    all_makespans = []
    all_balances = []
    best_avg_reward = -float('inf')
    best_avg_makespan = float('inf')
    best_balance = float('inf')
    no_improve_count = 0

    print("\n" + "=" * 60)
    print("🚀 开始训练")
    print("=" * 60)

    episode = 0
    pbar = tqdm(total=TRAINING_CONFIG['max_episodes'], desc="训练进度")

    while episode < TRAINING_CONFIG['max_episodes']:
        # 批量收集经验
        batch_memory, batch_rewards, batch_makespans, batch_balances = collect_batch_episodes(
            env, ppo, train_instances,
            TRAINING_CONFIG['batch_episodes'],
            device, TRAINING_CONFIG, episode
        )
        episode += TRAINING_CONFIG['batch_episodes']

        # 更新PPO
        if len(batch_memory.states) > 0:
            # 根据您的PPO实现选择正确的返回值
            losses = ppo.update(batch_memory)
            if isinstance(losses, tuple) and len(losses) >= 2:
                policy_loss, value_loss = losses[0], losses[1]
                entropy_loss = losses[2] if len(losses) > 2 else 0
            else:
                policy_loss = value_loss = entropy_loss = 0
        else:
            policy_loss = value_loss = entropy_loss = 0

        # 更新统计
        all_rewards.extend(batch_rewards)
        all_makespans.extend(batch_makespans)
        all_balances.extend(batch_balances)
        episode += TRAINING_CONFIG['batch_episodes']

        # 限制历史记录长度
        if len(all_rewards) > 1000:
            all_rewards = all_rewards[-1000:]
            all_makespans = all_makespans[-1000:]
            all_balances = all_balances[-1000:]

        # 计算平均指标
        avg_reward = np.mean(batch_rewards)
        avg_makespan = np.mean(batch_makespans)
        avg_balance = np.mean(batch_balances)

        # 记录到TensorBoard
        writer.add_scalar("Batch/AvgReward", avg_reward, episode)
        writer.add_scalar("Batch/AvgMakespan", avg_makespan, episode)
        writer.add_scalar("Batch/AvgBalance", avg_balance, episode)
        writer.add_scalar("Loss/Policy", policy_loss, episode)
        writer.add_scalar("Loss/Value", value_loss, episode)
        writer.add_scalar("Loss/Entropy", entropy_loss, episode)

        # 更新Visdom
        if vis and 'reward' in vis_windows:
            try:
                vis.line(Y=torch.tensor([avg_reward]).cpu(), X=torch.tensor([episode]).cpu(),
                         win=vis_windows['reward'], update='append')
                vis.line(Y=torch.tensor([avg_makespan]).cpu(), X=torch.tensor([episode]).cpu(),
                         win=vis_windows['makespan'], update='append')
                vis.line(Y=torch.tensor([avg_balance]).cpu(), X=torch.tensor([episode]).cpu(),
                         win=vis_windows['balance'], update='append')
                vis.line(Y=torch.tensor([[policy_loss, value_loss]]).cpu(),
                         X=torch.tensor([episode]).cpu(),
                         win=vis_windows['losses'], update='append')
            except:
                pass

        # 更新进度条
        pbar.update(TRAINING_CONFIG['batch_episodes'])
        pbar.set_postfix({
            'R': f"{avg_reward:.1f}",
            'M': f"{avg_makespan:.1f}",
            'B': f"{avg_balance:.2f}",
            'PL': f"{policy_loss:.3f}",
            'VL': f"{value_loss:.3f}"
        })

        # 定期评估
        if episode % TRAINING_CONFIG['eval_frequency'] == 0 and episode > 0:
            eval_reward, eval_makespan, eval_balance = evaluate_model(
                ppo, env, train_instances, device, TRAINING_CONFIG
            )

            print(f"\n📊 Episode {episode} 评估结果:")
            print(f"  平均奖励: {eval_reward:.2f}")
            print(f"  平均Makespan: {eval_makespan:.2f}")
            print(f"  负载均衡度(std): {eval_balance:.2f}")

            # 保存最佳模型
            if eval_reward > best_avg_reward and eval_balance < 3.0:  # 确保负载均衡
                best_avg_reward = eval_reward
                best_avg_makespan = eval_makespan
                best_balance = eval_balance
                no_improve_count = 0

                torch.save({
                    'model_state_dict': ppo.policy.state_dict(),
                    'optimizer_state_dict': ppo.optimizer.state_dict() if hasattr(ppo, 'optimizer') else None,
                    'episode': episode,
                    'avg_reward': best_avg_reward,
                    'avg_makespan': best_avg_makespan,
                    'balance': best_balance,
                    'config': {
                        'hgnn': HGNN_CONFIG,
                        'ppo': PPO_CONFIG,
                        'training': TRAINING_CONFIG
                    }
                }, f'{model_dir}/best_model.pt')
                print(f"  ✅ 保存最佳模型")
            else:
                no_improve_count += 1

        # 定期保存检查点
        if episode % TRAINING_CONFIG['save_frequency'] == 0 and episode > 0:
            torch.save({
                'model_state_dict': ppo.policy.state_dict(),
                'optimizer_state_dict': ppo.optimizer.state_dict() if hasattr(ppo, 'optimizer') else None,
                'episode': episode,
                'all_rewards': all_rewards[-100:],
                'all_makespans': all_makespans[-100:],
                'all_balances': all_balances[-100:]
            }, f'{model_dir}/checkpoint_ep{episode}.pt')

        # 早停检查
        if no_improve_count >= TRAINING_CONFIG['early_stop_patience']:
            print(f"\n⚠️ 早停: 连续{no_improve_count}次评估无改进")
            break

    pbar.close()

    # 训练结束
    print("\n" + "=" * 60)
    print("✅ 训练完成!")
    print(f"最佳平均奖励: {best_avg_reward:.2f}")
    print(f"最佳平均Makespan: {best_avg_makespan:.2f}")
    print(f"最佳负载均衡度: {best_balance:.2f}")
    print("=" * 60)

    # 生成最终甘特图
    print("\n生成最终调度甘特图...")
    try:
        # 加载最佳模型
        checkpoint = torch.load(f'{model_dir}/best_model.pt')
        ppo.policy.load_state_dict(checkpoint['model_state_dict'])

        # 运行一个完整的episode
        tasks, usvs = train_instances[0]
        state = env.reset_with_instances(tasks, usvs)
        done = False
        steps = 0

        while not done and steps < num_tasks:
            usv_feats = state['usv_features']
            task_feats = state['task_features']
            distances = calculate_usv_task_distances(usv_feats[:, :2], task_feats[:, :2])
            graph = build_heterogeneous_graph(
                usv_feats, task_feats, distances,
                eta=TRAINING_CONFIG['eta'], device=device
            )
            graph.action_mask = state['action_mask']

            with torch.no_grad():
                action, _, _ = ppo.policy_old.act(graph, device, state['action_mask'])

            state, _, done, _ = env.step(action)
            steps += 1

        # 生成甘特图
        generate_gantt_chart(env, save_path=f'{model_dir}/final_gantt.png')

        # 打印最终评估报告
        eval_result = evaluate_scheduling_result(env)
        print_evaluation_report(eval_result, "Final PPO Model")

    except Exception as e:
        print(f"生成甘特图时出错: {e}")

    # 关闭资源
    writer.close()

    print("\n训练脚本执行完毕！")


if __name__ == "__main__":
    main()

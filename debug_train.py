import os
import time
import random
import torch
import numpy as np
import sys

print("=== 训练脚本开始执行 ===")
print(f"Python版本: {sys.version}")
print(f"工作目录: {os.getcwd()}")


def debug_print(msg):
    """带时间戳的调试输出"""
    print(f"[DEBUG {time.strftime('%H:%M:%S')}] {msg}")
    sys.stdout.flush()  # 强制刷新输出


try:
    debug_print("开始导入模块...")

    from tqdm import tqdm

    debug_print("✅ tqdm导入成功")

    from torch.utils.tensorboard import SummaryWriter

    debug_print("✅ tensorboard导入成功")

    from env.usv_env import USVSchedulingEnv

    debug_print("✅ USVSchedulingEnv导入成功")

    from env.state_representation import build_heterogeneous_graph, calculate_usv_task_distances

    debug_print("✅ state_representation导入成功")

    from graph.hgnn import USVHeteroGNN

    debug_print("✅ USVHeteroGNN导入成功")

    from PPO_model import PPO, Memory

    debug_print("✅ PPO模块导入成功")

    import visdom

    debug_print("✅ visdom导入成功")

    import utils.data_generator as data_generator

    debug_print("✅ data_generator导入成功")

    import matplotlib.pyplot as plt

    debug_print("✅ matplotlib导入成功")

    debug_print("所有模块导入完成！")

except Exception as e:
    debug_print(f"❌ 导入模块时出错: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# 简化的超参数配置
num_usvs = 3
num_tasks = 30
num_instances = 10

TRAINING_CONFIG = {
    'max_episodes': 5,  # 先用很少的episode测试
    'max_steps_per_episode': 50,  # 减少步数
    'early_stop_patience': 200,
    'seed': 42,
    'eta': 2,
}

HGNN_CONFIG = {
    'hidden_dim': 64,  # 减小以加快速度
    'n_heads': 2,
    'num_layers': 1,
    'dropout': 0.1,
}

PPO_CONFIG = {
    'lr': 3e-4,
    'gamma': 0.98,
    'eps_clip': 0.2,
    'K_epochs': 4,  # 减少
    'entropy_coef': 0.02,
    'value_coef': 0.5,
}


def setup_seed(seed):
    """设置随机种子确保结果可复现"""
    debug_print(f"设置随机种子: {seed}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    """主函数"""
    debug_print("=== 进入main函数 ===")

    try:
        # 设置随机种子
        debug_print("步骤1: 设置随机种子")
        setup_seed(TRAINING_CONFIG['seed'])

        # 设备配置
        debug_print("步骤2: 配置设备")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        debug_print(f"使用设备: {device}")

        # 数据准备
        debug_print("步骤3: 准备数据")
        instances = data_generator.generate_batch_instances(
            num_instances=num_instances,
            fixed_tasks=num_tasks,
            fixed_usvs=num_usvs
        )
        debug_print(f"生成了{len(instances)}个算例")

        # 保存数据
        debug_print("步骤4: 保存数据")
        file_path = "data/fixed_instances.pkl"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        data_generator.save_instances_to_file(instances, file_path)
        debug_print("数据保存完成")

        # 初始化环境
        debug_print("步骤5: 初始化环境")
        env = USVSchedulingEnv(num_usvs=num_usvs, num_tasks=num_tasks)
        debug_print("环境创建成功")

        # 获取特征维度
        debug_print("步骤6: 获取特征维度")
        state = env.reset()
        usv_feats = state['usv_features']
        task_feats = state['task_features']
        debug_print(f"USV特征维度: {usv_feats.shape}, 任务特征维度: {task_feats.shape}")

        # 测试图构建
        debug_print("步骤7: 测试图构建")
        distances = calculate_usv_task_distances(usv_feats[:, :2], task_feats[:, :2])
        graph = build_heterogeneous_graph(usv_feats, task_feats, distances, eta=TRAINING_CONFIG['eta'])
        graph = graph.to(device)
        debug_print(f"图构建成功，节点类型: {graph.ntypes}")

        # 初始化HGNN模型
        debug_print("步骤8: 初始化HGNN")
        hgnn = USVHeteroGNN(
            usv_feat_dim=usv_feats.shape[1],
            task_feat_dim=task_feats.shape[1],
            hidden_dim=HGNN_CONFIG['hidden_dim'],
            n_heads=HGNN_CONFIG['n_heads'],
            num_layers=HGNN_CONFIG['num_layers'],
            dropout=HGNN_CONFIG['dropout']
        ).to(device)
        debug_print("HGNN初始化成功")

        # 初始化PPO
        debug_print("步骤9: 初始化PPO")
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
        debug_print("PPO初始化成功")

        # 创建日志目录
        debug_print("步骤10: 创建日志目录")
        model_dir = "model"
        os.makedirs(model_dir, exist_ok=True)
        debug_print("模型目录创建成功")

        # 初始化TensorBoard（可能出问题的地方）
        debug_print("步骤11: 初始化TensorBoard")
        try:
            writer = SummaryWriter("runs/usv_scheduling")
            debug_print("TensorBoard初始化成功")
        except Exception as e:
            debug_print(f"TensorBoard初始化失败: {e}")
            writer = None

        # 初始化visdom（可能出问题的地方）
        debug_print("步骤12: 初始化Visdom")
        try:
            vis = visdom.Visdom()
            if vis.check_connection():
                debug_print("Visdom连接成功")
            else:
                debug_print("Visdom连接失败，跳过可视化")
                vis = None
        except Exception as e:
            debug_print(f"Visdom初始化出错: {e}")
            vis = None

        # 训练统计
        debug_print("步骤13: 初始化训练统计")
        best_makespan = float('inf')
        no_improve_count = 0
        start_time = time.time()
        instance_index = 0

        # 开始训练循环
        debug_print("步骤14: 开始训练循环")
        debug_print(f"计划训练{TRAINING_CONFIG['max_episodes']}个episode")

        for episode in range(TRAINING_CONFIG['max_episodes']):
            debug_print(f"\n--- Episode {episode + 1}/{TRAINING_CONFIG['max_episodes']} ---")

            try:
                # 重置环境
                debug_print(f"Episode {episode}: 重置环境")
                tasks, usvs = instances[instance_index % num_instances]
                instance_index += 1
                state = env.reset_with_instances(tasks, usvs)
                debug_print(f"Episode {episode}: 环境重置成功")

                memory = Memory()
                done = False
                total_reward = 0
                steps = 0
                episode_makespan = 0

                debug_print(f"Episode {episode}: 开始episode内循环")

                # Episode内循环
                step_count = 0
                while not done and steps < TRAINING_CONFIG['max_steps_per_episode']:
                    step_count += 1
                    if step_count % 10 == 0:
                        debug_print(f"Episode {episode}: 执行第{step_count}步")

                    # 构建图状态
                    usv_feats = state['usv_features']
                    task_feats = state['task_features']
                    distances = calculate_usv_task_distances(usv_feats[:, :2], task_feats[:, :2])
                    graph = build_heterogeneous_graph(usv_feats, task_feats, distances, eta=TRAINING_CONFIG['eta'])
                    graph = graph.to(device)
                    graph.action_mask = state['action_mask']

                    # 选择动作
                    action, log_prob, state_value = ppo.select_action(graph)

                    # 执行动作
                    next_state, reward, done, info = env.step(action)

                    # 处理episode结束
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

                debug_print(f"Episode {episode}: episode内循环完成，总步数: {steps}")

                # 更新PPO策略
                debug_print(f"Episode {episode}: 更新PPO策略")
                if len(memory.states) > 0:
                    policy_loss_avg, value_loss_avg, entropy_loss_avg = ppo.update(memory)
                    debug_print(f"Episode {episode}: PPO更新完成")
                else:
                    debug_print(f"Episode {episode}: 内存为空，跳过PPO更新")

                # 性能统计
                makespan = episode_makespan if episode_makespan > 0 else float('inf')

                debug_print(f"Episode {episode}: 奖励={total_reward:.2f}, makespan={makespan:.2f}")

                # 记录日志
                if writer:
                    try:
                        writer.add_scalar("Reward/Episode", total_reward, episode)
                        writer.add_scalar("Makespan/Episode", makespan, episode)
                        debug_print(f"Episode {episode}: TensorBoard日志记录成功")
                    except Exception as e:
                        debug_print(f"Episode {episode}: TensorBoard记录失败: {e}")

            except Exception as e:
                debug_print(f"❌ Episode {episode} 执行出错: {e}")
                import traceback
                traceback.print_exc()
                break

        debug_print("=== 训练循环完成 ===")

        # 清理资源
        debug_print("步骤15: 清理资源")
        if writer:
            writer.close()
            debug_print("TensorBoard writer关闭")

        debug_print("🎉 训练脚本执行完成！")

    except Exception as e:
        debug_print(f"❌ main函数执行出错: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    debug_print("=== 脚本开始执行 ===")
    try:
        success = main()
        if success:
            debug_print("✅ 程序正常结束")
        else:
            debug_print("❌ 程序异常结束")
    except Exception as e:
        debug_print(f"❌ 顶层异常: {e}")
        import traceback

        traceback.print_exc()

    debug_print("=== 脚本执行结束 ===")
    input("按Enter键退出...")  # 防止窗口立即关闭

import sys

print("Python版本:", sys.version)
print("当前工作目录:", __file__)


def test_imports():
    """测试所有导入"""
    print("\n=== 测试导入 ===")

    try:
        import os
        print("✅ os")
        import torch
        print("✅ torch", torch.__version__)
        import numpy as np
        print("✅ numpy", np.__version__)
        import dgl
        print("✅ dgl", dgl.__version__)
    except Exception as e:
        print(f"❌ 基础库导入失败: {e}")
        return False

    try:
        # 测试自定义模块导入
        import utils.data_generator as data_generator
        print("✅ data_generator")
    except Exception as e:
        print(f"❌ data_generator导入失败: {e}")
        return False

    try:
        from env.usv_env import USVSchedulingEnv
        print("✅ USVSchedulingEnv")
    except Exception as e:
        print(f"❌ USVSchedulingEnv导入失败: {e}")
        return False

    try:
        from env.state_representation import build_heterogeneous_graph
        print("✅ state_representation")
    except Exception as e:
        print(f"❌ state_representation导入失败: {e}")
        return False

    try:
        from graph.hgnn import USVHeteroGNN
        print("✅ USVHeteroGNN")
    except Exception as e:
        print(f"❌ USVHeteroGNN导入失败: {e}")
        return False

    try:
        from PPO_model import PPO, Memory
        print("✅ PPO模块")
    except Exception as e:
        print(f"❌ PPO模块导入失败: {e}")
        return False

    return True


def test_environment():
    """测试环境创建"""
    print("\n=== 测试环境创建 ===")

    try:
        from env.usv_env import USVSchedulingEnv
        env = USVSchedulingEnv(num_usvs=3, num_tasks=30)
        print("✅ 环境创建成功")

        state = env.reset()
        print("✅ 环境重置成功")
        print(f"状态键: {list(state.keys())}")

        return True
    except Exception as e:
        print(f"❌ 环境测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_models():
    """测试模型创建"""
    print("\n=== 测试模型创建 ===")

    try:
        import torch
        from graph.hgnn import USVHeteroGNN
        from PPO_model import PPO

        # 使用GPU如果可用
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"✅ 设备: {device}")

        # 创建HGNN
        hgnn = USVHeteroGNN(
            usv_feat_dim=4,
            task_feat_dim=6,
            hidden_dim=128,  # 恢复正常维度
            n_heads=4,
            num_layers=2,
            dropout=0.1  # 现在应该能正常工作
        ).to(device)
        print("✅ HGNN创建成功")

        # 创建PPO
        ppo = PPO(
            hgnn=hgnn,
            action_dim=90,  # 3*30
            device=device
        )
        print("✅ PPO创建成功")

        return True
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_generation():
    """测试数据生成"""
    print("\n=== 测试数据生成 ===")

    try:
        import utils.data_generator as data_generator

        # 测试更多算例
        instances = data_generator.generate_batch_instances(
            num_instances=10,  # 增加到10个算例
            fixed_tasks=30,
            fixed_usvs=3
        )
        print(f"✅ 生成了{len(instances)}个算例")

        return True
    except Exception as e:
        print(f"❌ 数据生成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("=== 开始诊断训练脚本问题 ===")

    # 逐步测试各个组件
    if not test_imports():
        print("❌ 导入测试失败，请检查模块路径和依赖")
        return

    if not test_data_generation():
        print("❌ 数据生成测试失败")
        return

    if not test_environment():
        print("❌ 环境测试失败")
        return

    if not test_models():
        print("❌ 模型测试失败")
        return

    print("\n🎉 所有组件测试通过！")
    print("现在可以尝试运行完整的训练脚本")


if __name__ == "__main__":
    main()

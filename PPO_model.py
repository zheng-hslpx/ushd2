import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from graph.hgnn import StableUSVHeteroGNN


# ============================================================
# 经验回放存储类，用于在一个 batch 内存储 PPO 所需的所有轨迹数据
# ============================================================
class Memory:
    def __init__(self):
        self.actions = []        # 存储动作
        self.states = []         # 存储状态（图数据）
        self.logprobs = []       # 存储动作的 log 概率
        self.rewards = []        # 存储奖励
        self.is_terminals = []   # 存储是否终止
        self.state_values = []   # 存储状态价值
        self.returns = []        # 存储计算好的 returns
        self.advantages = []     # 存储计算好的优势值

    def clear_memory(self):
        """清空缓冲区"""
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.state_values[:]
        del self.returns[:]
        del self.advantages[:]


# ============================================================
# Actor-Critic 网络
# - 这里直接内部实例化 StableUSVHeteroGNN，而不从外部传入
# ============================================================
class ActorCritic(nn.Module):
    """Actor-Critic 网络结构"""

    def __init__(self, action_dim, usv_feat_dim, task_feat_dim,
                 hidden_dim=256, n_heads=4, num_layers=2, dropout=0.1, device='cpu'):
        """
        参数:
            action_dim: 动作空间维度（usv 数量 * task 数量）
            usv_feat_dim: USV 节点特征维度
            task_feat_dim: Task 节点特征维度
            hidden_dim: GNN 隐藏层维度
            n_heads: GAT 多头注意力头数
            num_layers: GNN 层数
            dropout: Dropout 概率
            device: 计算设备（'cpu' or 'cuda'）
        """
        super().__init__()

        # 在这里直接创建 HGNN，避免在外部单独实例化
        self.hgnn = StableUSVHeteroGNN(
            usv_feat_dim=usv_feat_dim,
            task_feat_dim=task_feat_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            num_layers=num_layers,
            dropout=dropout
        ).to(device)

        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # ------------------- Actor 网络 -------------------
        self.actor_feature = nn.Sequential(
            nn.Linear(self.hgnn.hidden_dim, hidden_dim),  # 从 GNN 输出到隐藏层
            nn.LayerNorm(hidden_dim),                     # LayerNorm 提升稳定性
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)              # 输出动作 logits
        )

        # ------------------- Critic 网络 -------------------
        self.critic_feature = nn.Sequential(
            nn.Linear(self.hgnn.hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)                       # 输出状态价值
        )

        # 参数初始化
        self.apply(self._init_weights)
        nn.init.orthogonal_(self.actor_head[-1].weight, gain=0.01)  # actor 最后一层小权重
        nn.init.orthogonal_(self.critic_head[-1].weight, gain=1.0)  # critic 最后一层大权重

    def _init_weights(self, module):
        """权重初始化方法：正交初始化"""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    # ---------------------------------------------------------
    # 动作选择函数（用于采样）
    # ---------------------------------------------------------
    def act(self, state, device, action_mask=None):
        """
        参数:
            state: 图数据（DGLGraph）
            device: 运行设备
            action_mask: 动作掩码（True 表示可选，False 表示不可选）
        返回:
            action.item(): 选择的动作
            action_logprob: 动作的 log 概率
            state_val: 当前状态的价值
        """
        with torch.no_grad():
            # 1. 通过 HGNN 提取图嵌入
            graph_embedding = self.hgnn(state).to(device)

            # 2. Actor 分支
            actor_features = self.actor_feature(graph_embedding)
            action_logits = self.actor_head(actor_features)

            # 3. Critic 分支
            critic_features = self.critic_feature(graph_embedding)
            state_val = self.critic_head(critic_features)

            # 4. 应用动作掩码（不可选的动作 logits 设为 -1e10）
            if action_mask is not None:
                action_mask_tensor = torch.from_numpy(action_mask).to(device)
                action_logits = action_logits.masked_fill(~action_mask_tensor, -1e10)

            # 5. Softmax 转为概率
            action_probs = torch.softmax(action_logits, dim=-1)

            # 6. 5% 概率添加随机噪声，增加探索
            if torch.rand(1).item() < 0.05:
                noise = torch.randn_like(action_probs) * 0.1
                action_probs = torch.softmax(action_logits + noise, dim=-1)

            # 7. 按概率采样动作
            dist = Categorical(action_probs)
            action = dist.sample()
            action_logprob = dist.log_prob(action)

        return action.item(), action_logprob, state_val

    # ---------------------------------------------------------
    # 评估函数（用于计算损失）
    # ---------------------------------------------------------
    def evaluate(self, states, actions, device):
        """
        参数:
            states: 图数据列表
            actions: 动作张量
            device: 运行设备
        返回:
            action_logprobs: 动作 log 概率
            state_values: 状态价值
            dist_entropy: 策略分布熵（鼓励探索）
        """
        graph_embeddings = []
        for s in states:
            emb = self.hgnn(s.to(device))
            graph_embeddings.append(emb)
        graph_embeddings = torch.stack(graph_embeddings)

        actor_features = self.actor_feature(graph_embeddings)
        action_logits = self.actor_head(actor_features)

        critic_features = self.critic_feature(graph_embeddings)
        state_values = self.critic_head(critic_features)

        action_probs = torch.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()

        return action_logprobs, state_values.squeeze(-1), dist_entropy


# ============================================================
# PPO 算法封装
# ============================================================
class PPO:
    def __init__(self, action_dim, usv_feat_dim, task_feat_dim,
                 hidden_dim=256, n_heads=4, num_layers=2, dropout=0.1,
                 lr_actor=3e-4, lr_critic=1e-3,
                 gamma=0.99, eps_clip=0.2, K_epochs=10, device='cpu',
                 entropy_coef=0.01, value_coef=1.0, gae_lambda=0.95):

        # --------- PPO 参数 ---------
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.gae_lambda = gae_lambda

        # --------- 策略网络（ActorCritic）---------
        # policy（新策略）
        self.policy = ActorCritic(
            action_dim=action_dim,
            usv_feat_dim=usv_feat_dim,
            task_feat_dim=task_feat_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            num_layers=num_layers,
            dropout=dropout,
            device=device
        ).to(device)

        # policy_old（旧策略，用于计算 PPO 损失）
        self.policy_old = ActorCritic(
            action_dim=action_dim,
            usv_feat_dim=usv_feat_dim,
            task_feat_dim=task_feat_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            num_layers=num_layers,
            dropout=dropout,
            device=device
        ).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        # --------- 优化器 ---------
        self.actor_optimizer = torch.optim.Adam(
            list(self.policy.actor_feature.parameters()) +
            list(self.policy.actor_head.parameters()),
            lr=lr_actor,
            eps=1e-5
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.policy.critic_feature.parameters()) +
            list(self.policy.critic_head.parameters()),
            lr=lr_critic,
            eps=1e-5
        )

        # 学习率调度（余弦退火）
        self.actor_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=1000, eta_min=1e-5)
        self.critic_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=1000, eta_min=1e-4)

        # Critic 损失函数
        self.mse_loss = nn.MSELoss()

    # ---------------------------------------------------------
    # 使用旧策略选择动作（交互阶段）
    # ---------------------------------------------------------
    def select_action(self, state):
        graph = state
        action_mask = getattr(graph, 'action_mask', None)
        action, action_logprob, state_value = self.policy_old.act(
            graph, self.device, action_mask
        )
        return action, action_logprob.cpu(), state_value.cpu()

    # ---------------------------------------------------------
    # 计算 GAE（广义优势估计）
    # ---------------------------------------------------------
    def compute_gae(self, rewards, values, is_terminals):
        advantages = []
        returns = []
        gae = 0
        next_value = 0

        for t in reversed(range(len(rewards))):
            if is_terminals[t]:
                next_value = 0
                gae = 0
            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
            next_value = values[t]

        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages

    # ---------------------------------------------------------
    # 更新 PPO 策略
    # ---------------------------------------------------------
    def update(self, memory):
        old_states = memory.states
        old_actions = torch.tensor(memory.actions, dtype=torch.long).to(self.device)
        old_logprobs = torch.stack(memory.logprobs).to(self.device).detach()
        old_state_values = torch.stack(memory.state_values).squeeze().to(self.device).detach()

        returns, advantages = self.compute_gae(
            memory.rewards,
            old_state_values.cpu().numpy(),
            memory.is_terminals
        )

        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0

        # 多轮优化
        for epoch in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions, self.device
            )

            ratios = torch.exp(logprobs - old_logprobs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * dist_entropy.mean()
            critic_loss = self.mse_loss(state_values, returns)

            # 更新 Actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(
                list(self.policy.actor_feature.parameters()) +
                list(self.policy.actor_head.parameters()),
                max_norm=0.5
            )
            self.actor_optimizer.step()

            # 更新 Critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.policy.critic_feature.parameters()) +
                list(self.policy.critic_head.parameters()),
                max_norm=0.5
            )
            self.critic_optimizer.step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy += dist_entropy.mean().item()

        # 学习率衰减
        self.actor_scheduler.step()
        self.critic_scheduler.step()

        # 更新旧策略参数
        self.policy_old.load_state_dict(self.policy.state_dict())
        memory.clear_memory()

        return (
            total_actor_loss / self.K_epochs,
            total_critic_loss / self.K_epochs,
            total_entropy / self.K_epochs
        )

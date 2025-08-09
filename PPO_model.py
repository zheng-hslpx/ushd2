import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR


class Memory:
    """改进的经验回放缓冲区"""

    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.state_values = []
        self.returns = []  # 新增：存储计算好的returns
        self.advantages = []  # 新增：存储计算好的advantages

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


class ActorCritic(nn.Module):
    """改进的Actor-Critic网络"""

    def __init__(self, hgnn, action_dim, hidden_dim=256):
        super().__init__()
        self.hgnn = hgnn
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # 改进1: 分离Actor和Critic的特征提取
        # Actor网络
        self.actor_feature = nn.Sequential(
            nn.Linear(hgnn.hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # 使用LayerNorm替代BatchNorm
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # Critic网络 - 独立的特征提取
        self.critic_feature = nn.Sequential(
            nn.Linear(hgnn.hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # 改进2: 更好的初始化
        self.apply(self._init_weights)

        # 特别处理最后一层
        nn.init.orthogonal_(self.actor_head[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head[-1].weight, gain=1.0)

    def _init_weights(self, module):
        """改进的权重初始化"""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def act(self, state, device, action_mask=None):
        """选择动作"""
        with torch.no_grad():
            # 通过HGNN获取图嵌入
            graph_embedding = self.hgnn(state).to(device)

            # Actor分支
            actor_features = self.actor_feature(graph_embedding)
            action_logits = self.actor_head(actor_features)

            # Critic分支
            critic_features = self.critic_feature(graph_embedding)
            state_val = self.critic_head(critic_features)

            # 应用action mask
            if action_mask is not None:
                action_mask_tensor = torch.from_numpy(action_mask).to(device)
                # 使用更稳定的mask值
                action_logits = action_logits.masked_fill(~action_mask_tensor, -1e10)

            # 计算动作概率和采样
            action_probs = torch.softmax(action_logits, dim=-1)

            # 添加噪声以增加探索
            if torch.rand(1).item() < 0.05:  # 5%概率添加噪声
                noise = torch.randn_like(action_probs) * 0.1
                action_probs = torch.softmax(action_logits + noise, dim=-1)

            dist = Categorical(action_probs)
            action = dist.sample()
            action_logprob = dist.log_prob(action)

        return action.item(), action_logprob, state_val

    def evaluate(self, states, actions, device):
        """评估状态-动作对"""
        graph_embeddings = []
        for s in states:
            emb = self.hgnn(s.to(device))
            graph_embeddings.append(emb)

        graph_embeddings = torch.stack(graph_embeddings)

        # Actor分支
        actor_features = self.actor_feature(graph_embeddings)
        action_logits = self.actor_head(actor_features)

        # Critic分支
        critic_features = self.critic_feature(graph_embeddings)
        state_values = self.critic_head(critic_features)

        # 计算动作概率
        action_probs = torch.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)

        # 计算log概率和熵
        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()

        return action_logprobs, state_values.squeeze(-1), dist_entropy


class PPO:
    """改进的PPO算法实现"""

    def __init__(self, hgnn, action_dim, lr_actor=3e-4, lr_critic=1e-3,
                 gamma=0.99, eps_clip=0.2, K_epochs=10, device='cpu',
                 entropy_coef=0.01, value_coef=1.0, gae_lambda=0.95):

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.gae_lambda = gae_lambda  # GAE参数

        # 创建网络
        self.policy = ActorCritic(hgnn, action_dim).to(device)
        self.policy_old = ActorCritic(hgnn, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        # 改进3: 分离Actor和Critic的优化器
        self.actor_optimizer = torch.optim.Adam(
            list(self.policy.actor_feature.parameters()) +
            list(self.policy.actor_head.parameters()),
            lr=lr_actor,
            eps=1e-5
        )

        self.critic_optimizer = torch.optim.Adam(
            list(self.policy.critic_feature.parameters()) +
            list(self.policy.critic_head.parameters()),
            lr=lr_critic,  # Critic使用更高的学习率
            eps=1e-5
        )

        # 改进4: 使用余弦退火学习率调度
        self.actor_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=1000, eta_min=1e-5)
        self.critic_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=1000, eta_min=1e-4)

        # 损失函数
        self.mse_loss = nn.MSELoss()

        # 统计信息
        self.update_count = 0

    def select_action(self, state):
        """使用旧策略选择动作"""
        graph = state
        action_mask = getattr(graph, 'action_mask', None)

        action, action_logprob, state_value = self.policy_old.act(
            graph, self.device, action_mask
        )

        return action, action_logprob.cpu(), state_value.cpu()

    def compute_gae(self, rewards, values, is_terminals):
        """计算Generalized Advantage Estimation (GAE)"""
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

        # 标准化advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns, advantages

    def update(self, memory):
        """改进的PPO更新"""
        # 准备数据
        old_states = memory.states
        old_actions = torch.tensor(memory.actions, dtype=torch.long).to(self.device)
        old_logprobs = torch.stack(memory.logprobs).to(self.device).detach()
        old_state_values = torch.stack(memory.state_values).squeeze().to(self.device).detach()

        # 计算GAE
        returns, advantages = self.compute_gae(
            memory.rewards,
            old_state_values.cpu().numpy(),
            memory.is_terminals
        )

        # 保存初始损失用于监控
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0

        # 多轮优化
        for epoch in range(self.K_epochs):
            # 评估旧动作
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions, self.device
            )

            # 计算比率
            ratios = torch.exp(logprobs - old_logprobs)

            # Actor损失
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * dist_entropy.mean()

            # Critic损失 - 使用MSE
            critic_loss = self.mse_loss(state_values, returns)

            # 改进5: 分别更新Actor和Critic
            # 更新Actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(
                list(self.policy.actor_feature.parameters()) +
                list(self.policy.actor_head.parameters()),
                max_norm=0.5
            )
            self.actor_optimizer.step()

            # 更新Critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.policy.critic_feature.parameters()) +
                list(self.policy.critic_head.parameters()),
                max_norm=0.5
            )
            self.critic_optimizer.step()

            # 累计损失
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy += dist_entropy.mean().item()

        # 更新学习率
        self.actor_scheduler.step()
        self.critic_scheduler.step()

        # 更新旧策略
        self.policy_old.load_state_dict(self.policy.state_dict())

        # 清理内存
        memory.clear_memory()

        self.update_count += 1

        return (
            total_actor_loss / self.K_epochs,
            total_critic_loss / self.K_epochs,
            total_entropy / self.K_epochs
        )

import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Memory:
    """经验回放缓冲区"""
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.state_values = [] # 新增：存储旧的状态价值

    def clear_memory(self):
        """清空缓冲区"""
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.state_values[:] # 新增：清空旧的状态价值

class ActorCritic(nn.Module):
    """Actor-Critic网络"""
    def __init__(self, hgnn, action_dim, hidden_dim=128):
        super().__init__()
        self.hgnn = hgnn
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # 共享特征层
        self.shared_net = nn.Sequential(
            nn.Linear(hgnn.hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Actor网络 - 添加残差连接
        self.actor_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # Critic网络
        self.critic_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """权重初始化"""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=0.01)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def act(self, state, device, action_mask=None):
        """选择动作"""
        # 通过HGNN获取图嵌入
        graph_embedding = self.hgnn(state).to(device)
        shared_features = self.shared_net(graph_embedding)

        # 计算动作logits
        action_logits = self.actor_net(shared_features)

        # 应用action mask(动作掩码)
        if action_mask is not None:
            action_mask_tensor = torch.from_numpy(action_mask).to(device)
            action_logits = action_logits.masked_fill(~action_mask_tensor, -1e8)

        # 计算动作概率和采样
        action_probs = torch.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        # 计算状态价值
        state_val = self.critic_net(shared_features)

        return action.detach().cpu().item(), action_logprob.detach(), state_val.detach()

    def evaluate(self, states, actions):
        """评估状态-动作对"""
        graph_embeddings = []
        for s in states:
            emb = self.hgnn(s)
            graph_embeddings.append(emb)

        graph_embeddings = torch.stack(graph_embeddings)
        shared_features = self.shared_net(graph_embeddings)

        # 计算动作概率
        action_logits = self.actor_net(shared_features)
        action_probs = torch.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)

        # 计算log概率、熵和状态价值
        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        state_values = self.critic_net(shared_features)

        return action_logprobs, torch.squeeze(state_values, -1), dist_entropy


class PPO:
    """PPO算法实现"""
    def __init__(self, hgnn, action_dim, lr=1e-4, gamma=0.99, eps_clip=0.15,
                 K_epochs=8, device='cpu', entropy_coef=0.02, value_coef=0.5):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

        # 创建当前策略和旧策略网络
        self.policy = ActorCritic(hgnn, action_dim).to(device)
        self.policy_old = ActorCritic(hgnn, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        # 使用AdamW优化器，更好的权重衰减
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=lr,
            weight_decay=1e-4,
            eps=1e-5
        )

        # 学习率调度器
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.8,
            patience=50,
            verbose=True
        )

        # 损失函数
        self.value_loss_fn = nn.SmoothL1Loss()

        # 统计信息
        self.update_count = 0

    def select_action(self, state):
        """使用旧策略选择动作"""
        graph = state
        action_mask = getattr(graph, 'action_mask', None)

        with torch.no_grad():
            action, action_logprob, state_value = self.policy_old.act(
                graph, self.device, action_mask
            )

        return action, action_logprob.cpu(), state_value.cpu()

    def update(self, memory):
        """更新PPO策略"""
        # 计算折扣奖励
        rewards = []
        discounted_reward = 0

        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = reward
            else:
                discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # 奖励标准化 - 使用更稳定的方法
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        if len(rewards) > 1:
            reward_mean = rewards.mean()
            reward_std = rewards.std() + 1e-8
            rewards = (rewards - reward_mean) / reward_std

        # 准备数据
        old_states = memory.states
        old_actions = torch.tensor(memory.actions, dtype=torch.long).to(self.device)
        old_logprobs = torch.stack(memory.logprobs).to(self.device).detach()
        old_state_values = torch.stack(memory.state_values).to(self.device).detach()

        # 计算优势
        advantages = rewards - old_state_values.squeeze()

        # 优势标准化
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 多轮优化
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0

        for epoch in range(self.K_epochs):
            # 评估旧动作
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # 计算重要性采样比率
            ratios = torch.exp(logprobs - old_logprobs)

            # 计算代理损失
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # 损失组件
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = self.value_loss_fn(state_values, rewards)
            entropy_loss = -self.entropy_coef * dist_entropy.mean()

            # 总损失
            total_loss = policy_loss + self.value_coef * value_loss + entropy_loss

            # 梯度更新
            self.optimizer.zero_grad()
            total_loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)

            self.optimizer.step()

            # 累计损失
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy_loss += entropy_loss.item()

        # 更新学习率
        avg_value_loss = total_value_loss / self.K_epochs
        self.scheduler.step(avg_value_loss)

        # 更新旧策略
        self.policy_old.load_state_dict(self.policy.state_dict())

        # 清理内存
        memory.clear_memory()

        self.update_count += 1

        return (total_policy_loss / self.K_epochs,
                total_value_loss / self.K_epochs,
                total_entropy_loss / self.K_epochs)


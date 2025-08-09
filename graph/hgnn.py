import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn


class StableUSVHeteroGNN(nn.Module):
    """
    更稳定的异构图神经网络
    主要改进：
    1. 简化架构，减少不必要的复杂性
    2. 使用更稳定的归一化方法
    3. 改进的注意力机制
    """

    def __init__(self, usv_feat_dim, task_feat_dim, hidden_dim=256,
                 n_heads=4, num_layers=2, dropout=0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.num_layers = num_layers

        # 特征编码器 - 使用LayerNorm替代BatchNorm
        self.usv_encoder = nn.Sequential(
            nn.Linear(usv_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.task_encoder = nn.Sequential(
            nn.Linear(task_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 简化的边类型
        self.etypes = [('task', 'to', 'task'), ('usv', 'to', 'task')]

        # GAT层
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            conv_dict = {}
            for etype in self.etypes:
                conv_dict[etype] = dglnn.GATConv(
                    in_feats=hidden_dim,
                    out_feats=hidden_dim // n_heads,
                    num_heads=n_heads,
                    feat_drop=dropout,
                    attn_drop=dropout,
                    residual=True,  # 使用内置的残差连接
                    activation=F.elu,
                    allow_zero_in_degree=True
                )
            self.gat_layers.append(dglnn.HeteroGraphConv(conv_dict, aggregate='mean'))

        # 节点级别的输出投影
        self.node_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 改进的全局池化
        self.global_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1, bias=False)
        )

        # 最终输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Xavier初始化"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, graph):

        # --- 添加调试信息 ---
        print(f"Debug: Graph node types: {graph.ntypes}")
        print(f"Debug: Graph expected 'usv'? {'usv' in graph.ntypes}")
        print(f"Debug: Graph expected 'task'? {'task' in graph.ntypes}")
        if 'usv' in graph.ntypes:
            print(f"Debug: Number of 'usv' nodes: {graph.num_nodes('usv')}")
            print(f"Debug: 'usv' node features exist? {'feat' in graph.nodes['usv'].data}")
        if 'task' in graph.ntypes:
            print(f"Debug: Number of 'task' nodes: {graph.num_nodes('task')}")
            print(f"Debug: 'task' node features exist? {'feat' in graph.nodes['task'].data}")
        # --------------------

        """前向传播"""
        # 特征编码
        h = {
            'usv': self.usv_encoder(graph.nodes['usv'].data['feat']),
            'task': self.task_encoder(graph.nodes['task'].data['feat'])
        }

        # GAT传播
        for i, gat_layer in enumerate(self.gat_layers):
            h_new = gat_layer(graph, h)
        #     # 处理多头输出
        #     for ntype in h_new:
        #         if h_new[ntype].dim() == 3:
        #             h_new[ntype] = h_new[ntype].flatten

        # --- 修改开始 ---
        for ntype in h_new:  # 只处理 h_new 中存在的类型
            if h_new[ntype].dim() == 3:
                h_new[ntype] = h_new[ntype].flatten(1)
        # 确保所有原始节点类型都在 h_new 中
        # 如果原始 h 中的类型在 h_new 中没有，则复制原始的（未更新的）特征
        # 这解决了 GAT 层可能不更新某些节点类型（如 'usv'）的问题
        for ntype in h:  # 遍历原始 h 中的所有类型
            if ntype not in h_new:  # 如果 h_new 中缺失
                h_new[ntype] = h[ntype]  # 从 h 复制过来
            # --- 修改结束 ---

            # 更新特征
            h = h_new

        # 节点投影
        h['usv'] = self.node_projector(h['usv'])
        h['task'] = self.node_projector(h['task'])

        # 全局池化
        usv_global = self._attention_pool(h['usv'])
        task_global = self._attention_pool(h['task'])

        # 合并全局特征
        global_feat = torch.cat([usv_global, task_global], dim=-1)

        # 输出
        output = self.output_layer(global_feat)

        return output

    def _attention_pool(self, features):
        """注意力池化"""
        if features.dim() == 1:
            return features

        # 计算注意力权重
        attention_scores = self.global_attention(features)
        attention_weights = F.softmax(attention_scores, dim=0)

        # 加权平均
        pooled = torch.sum(features * attention_weights, dim=0)

        return pooled


# class LightweightHGNN(nn.Module):
#     """
#     轻量级HGNN（备选方案）
#     适用于快速训练和推理
#     """
#
#     def __init__(self, usv_feat_dim, task_feat_dim, hidden_dim=128):
#         super().__init__()
#
#         self.hidden_dim = hidden_dim
#
#         # 简单的特征变换
#         self.usv_transform = nn.Linear(usv_feat_dim, hidden_dim)
#         self.task_transform = nn.Linear(task_feat_dim, hidden_dim)
#
#         # 消息传递层
#         self.message_layer = dglnn.GraphConv(hidden_dim, hidden_dim,
#                                              norm='both',
#                                              weight=True,
#                                              bias=True,
#                                              activation=F.relu)
#
#         # 输出层
#         self.output = nn.Sequential(
#             nn.Linear(hidden_dim * 2, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim)
#         )
#
#     def forward(self, graph):
#         """前向传播"""
#         # 特征变换
#         usv_h = F.relu(self.usv_transform(graph.nodes['usv'].data['feat']))
#         task_h = F.relu(self.task_transform(graph.nodes['task'].data['feat']))
#
#         # 创建同构图进行消息传递
#         num_usv = usv_h.shape[0]
#         num_task = task_h.shape[0]
#
#         # 合并特征
#         all_h = torch.cat([usv_h, task_h], dim=0)
#
#         # 构建同构图的边（简化处理）
#         src = []
#         dst = []
#
#         # USV到任务的边
#         for i in range(num_usv):
#             for j in range(num_task):
#                 src.append(i)
#                 dst.append(num_usv + j)
#
#         # 任务之间的边（可选）
#         for i in range(num_task):
#             for j in range(min(3, num_task)):  # 每个任务连接3个最近邻
#                 if i != j:
#                     src.append(num_usv + i)
#                     dst.append(num_usv + j)
#
#         # 创建DGL图
#         homo_graph = dgl.graph((src, dst))
#         homo_graph = homo_graph.to(graph.device)
#
#         # 消息传递
#         h_updated = self.message_layer(homo_graph, all_h)
#
#         # 分离USV和任务特征
#         usv_h_updated = h_updated[:num_usv]
#         task_h_updated = h_updated[num_usv:]
#
#         # 全局池化
#         usv_global = torch.mean(usv_h_updated, dim=0)
#         task_global = torch.mean(task_h_updated, dim=0)
#
#         # 输出
#         global_feat = torch.cat([usv_global, task_global], dim=-1)
#         output = self.output(global_feat)
#
#         return output

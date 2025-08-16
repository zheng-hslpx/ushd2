import numpy as np
import torch
import dgl

def build_heterogeneous_graph(usv_features, task_features, usv_task_distances, eta=2, device=None):
    """
    构建异构图，任务节点仅与η个最近邻任务聚合

    Args:
        usv_features: USV特征矩阵 [num_usvs, usv_feat_dim]
        task_features: 任务特征矩阵 [num_tasks, task_feat_dim]
        usv_task_distances: USV-任务距离矩阵 [num_usvs, num_tasks]
        eta: 最近邻任务数量
        device: PyTorch设备（可选）

    Returns:
        dgl.DGLGraph: 构建的异构图
    """

    # --- 添加调试信息 ---
    # print(f"Debug: usv_features shape: {usv_features.shape}")
    # print(f"Debug: task_features shape: {task_features.shape}")
    # --------------------

    num_tasks = len(task_features)
    num_usvs = len(usv_features)

    # 1. 构建任务-任务边（基于空间最近邻）
    task_edges = []
    if eta > 0 and num_tasks > 1:
        # 计算任务间距离矩阵
        task_positions = task_features[:, :2]  # 假设前两维是坐标
        task_distances = np.sqrt(((task_positions[:, np.newaxis] - task_positions) ** 2).sum(axis=2))

        # 对每个任务，选择η个最近邻任务（排除自身）
        for i in range(num_tasks):
            # 获取最近邻索引，排除自身
            neighbors = np.argsort(task_distances[i])[1:min(eta + 1, num_tasks)]
            for j in neighbors:
                task_edges.append((i, j))

    # 2. 构建USV-任务边（基于可达性和距离约束）
    usv_task_edges = []
    max_reasonable_distance = np.percentile(usv_task_distances[usv_task_distances < np.inf], 90)

    for usv_idx in range(num_usvs):
        for task_idx in range(num_tasks):
            distance = usv_task_distances[usv_idx, task_idx]
            # 只连接距离合理的USV-任务对
            if distance < np.inf and distance <= max_reasonable_distance:
                usv_task_edges.append((usv_idx, task_idx))

    # 3. 处理边为空的情况
    if not task_edges:
        # 如果没有任务-任务边，创建一个自环避免图为空
        if num_tasks > 0:
            task_edges = [(0, 0)]

    if not usv_task_edges:
        # 如果没有USV-任务边，为每个USV连接最近的任务
        for usv_idx in range(num_usvs):
            nearest_task = np.argmin(usv_task_distances[usv_idx])
            usv_task_edges.append((usv_idx, nearest_task))

    # 4. 构建异构图
    try:
        graph_data = {
            ('usv', 'to', 'task'): usv_task_edges,
            ('task', 'to', 'task'): task_edges
        }
        graph = dgl.heterograph(graph_data)

        # 5. 设置节点特征
        graph.nodes['usv'].data['feat'] = torch.tensor(usv_features, dtype=torch.float32)
        graph.nodes['task'].data['feat'] = torch.tensor(task_features, dtype=torch.float32)

        # 6. 移动到指定设备
        if device is not None:
            graph = graph.to(device)

        return graph

    except Exception as e:
        print(f"构建异构图时出错: {e}")
        print(f"USV数量: {num_usvs}, 任务数量: {num_tasks}")
        print(f"USV-任务边数量: {len(usv_task_edges)}, 任务-任务边数量: {len(task_edges)}")
        raise


def calculate_usv_task_distances(usv_positions, task_positions, max_distance=None):
    """
    计算USV与任务之间的距离矩阵

    Args:
        usv_positions: USV位置 [num_usvs, 2]
        task_positions: 任务位置 [num_tasks, 2]
        max_distance: 最大距离限制，超过此距离设为inf

    Returns:
        np.ndarray: 距离矩阵 [num_usvs, num_tasks]
    """
    # 确保输入是numpy数组
    usv_positions = np.asarray(usv_positions)
    task_positions = np.asarray(task_positions)

    # 计算欧几里得距离
    distances = np.sqrt(((usv_positions[:, np.newaxis] - task_positions) ** 2).sum(axis=2))

    # 应用距离限制
    if max_distance is not None:
        distances[distances > max_distance] = np.inf

    return distances


def add_graph_features(graph, additional_features=None):
    """
    为图添加额外的特征

    Args:
        graph: DGL异构图
        additional_features: 额外特征字典

    Returns:
        dgl.DGLGraph: 增强后的图
    """
    if additional_features is None:
        return graph

    # 添加USV额外特征
    if 'usv' in additional_features:
        for feat_name, feat_data in additional_features['usv'].items():
            graph.nodes['usv'].data[feat_name] = torch.tensor(feat_data, dtype=torch.float32)

    # 添加任务额外特征
    if 'task' in additional_features:
        for feat_name, feat_data in additional_features['task'].items():
            graph.nodes['task'].data[feat_name] = torch.tensor(feat_data, dtype=torch.float32)

    # 添加边特征
    if 'edges' in additional_features:
        edge_features = additional_features['edges']
        if 'usv_to_task' in edge_features:
            graph.edges['to'].data['weight'] = torch.tensor(
                edge_features['usv_to_task'], dtype=torch.float32
            )

    return graph


def validate_graph_structure(graph):
    """
    验证图结构的合理性

    Args:
        graph: DGL异构图

    Returns:
        dict: 验证结果
    """
    validation_result = {
        'is_valid': True,
        'warnings': [],
        'errors': [],
        'stats': {}
    }

    try:
        # 检查节点数量
        num_usvs = graph.num_nodes('usv')
        num_tasks = graph.num_nodes('task')

        validation_result['stats']['num_usvs'] = num_usvs
        validation_result['stats']['num_tasks'] = num_tasks

        # 检查边数量
        usv_task_edges = graph.num_edges(('usv', 'to', 'task'))
        task_task_edges = graph.num_edges(('task', 'to', 'task'))

        validation_result['stats']['usv_task_edges'] = usv_task_edges
        validation_result['stats']['task_task_edges'] = task_task_edges

        # 验证规则
        if num_usvs == 0:
            validation_result['errors'].append("没有USV节点")
            validation_result['is_valid'] = False

        if num_tasks == 0:
            validation_result['errors'].append("没有任务节点")
            validation_result['is_valid'] = False

        if usv_task_edges == 0:
            validation_result['warnings'].append("没有USV-任务边，可能影响训练")

        if task_task_edges == 0:
            validation_result['warnings'].append("没有任务-任务边，任务间无连接")

        # 检查特征维度
        if 'feat' in graph.nodes['usv'].data:
            usv_feat_dim = graph.nodes['usv'].data['feat'].shape[1]
            validation_result['stats']['usv_feat_dim'] = usv_feat_dim

        if 'feat' in graph.nodes['task'].data:
            task_feat_dim = graph.nodes['task'].data['feat'].shape[1]
            validation_result['stats']['task_feat_dim'] = task_feat_dim

    except Exception as e:
        validation_result['errors'].append(f"验证过程出错: {str(e)}")
        validation_result['is_valid'] = False

    return validation_result
import argparse
import os
import json
import pickle
import numpy as np
import networkx as nx
from grakel import Graph
from grakel.kernels import WeisfeilerLehman, VertexHistogram, EdgeHistogram, ShortestPath
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE

fragment_id_list = ['7_28_1 R21', '8_10_1 R18', '8_10_2 R19', '8_11_1 R20']
ego_id_dict = {
    '7_28_1 R21': [1, 9, 11, 13, 26, 31, 79, 141, 144, 148, 162, 167, 170, 181],
    '8_10_1 R18': [13, 70, 76, 157],
    '8_10_2 R19': [75, 112, 126, 178],
    '8_11_1 R20': [4, 9, 37, 57, 60, 80, 84, 87, 93, 109, 159, 160, 161, 175, 216, 219, 289, 295, 316, 333, 372, 385, 390, 400, 479]
}
head2tail_types = ['following', 'diverging', 'converging', 'crossing conflict: same cross type', 'left turn and straight cross conflict: same side']
head2head_types = ['left turn and straight cross conflict: opposite side', 
                   'right turn and straight cross conflict: start side', 
                   'right turn and straight cross conflict: end side', 
                   'left turn and right turn conflict: start side', 
                   'left turn and right turn conflict: end side']

def load_cgs(save_dir):
    cgs_dict = {}
    for fragment_id in fragment_id_list:
        for ego_id in ego_id_dict[fragment_id]:
            save_path = os.path.join(save_dir, f"{fragment_id}_{ego_id}")
            json_file = os.path.join(save_path, f"cg_{fragment_id}_{ego_id}.json")
            
            if not os.path.exists(json_file):
                print(f"因果图文件未找到: {json_file}")
                
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    serializable_cg = json.load(f)
                    
                # 转回原始格式
                cg = {}
                for parent_id, edges in serializable_cg.items():
                    parent_id = int(parent_id) if parent_id.isdigit() else parent_id
                    cg[parent_id] = []
                    for edge in edges:
                        cg[parent_id].append((
                            int(edge['child_id']) if edge['child_id'].isdigit() else edge['child_id'],
                            edge['edge_attributes']
                        ))
                cgs_dict[(fragment_id, ego_id)] = cg
                        
                print(f"成功加载因果图: {json_file}")
                
            except Exception as e:
                print(f"加载因果图时出错: {e}")

    return cgs_dict

def convert_cg_to_grakel(cg, fragment_id):
    """
    将因果图转换为GraKeL图格式
    
    Returns:
        GraKeL图对象
    """
    if not cg:
        print("警告：因果图为空")
        return None
        
    G = nx.DiGraph()
    
    # 收集所有节点
    all_nodes = set()
    for parent_id, edges in cg.items():
        all_nodes.add(parent_id)
        for child_id, _ in edges:
            all_nodes.add(child_id)
    
    # 添加节点及其属性
    for node in all_nodes:
        # 获取节点信息
        node_info = get_agent_info(fragment_id, node)
        if node_info:
            agent_type, agent_class, cross_type, signal_violation, retrograde_type, cardinal_direction = node_info
            
            # 将类别信息编码为数值
            # 将agent_type编码为数值
            type_code = 0
            if agent_type == 'mv':
                type_code = 1
            elif agent_type == 'nmv':
                type_code = 2
            else:
                type_code = 0
            
            # 将cross_type编码为数值
            cross_type_code = 0
            if cross_type:
                if 'Left' in str(cross_type):
                    cross_type_code = 1
                elif 'Right' in str(cross_type):
                    cross_type_code = 2
                elif 'Straight' in str(cross_type):
                    cross_type_code = 3
                elif 'U-Turn' in str(cross_type):
                    cross_type_code = 4

            sv_code = 0
            if signal_violation:
                if 'yellow' in signal_violation:
                    sv_code = 1
                elif 'red' in signal_violation:
                    sv_code = 2

            rt_code = 0
            if retrograde_type:
                if 'front' in retrograde_type:
                    rt_code = 1
                elif 'rear' in retrograde_type:
                    rt_code = 2
                elif 'full' in retrograde_type:
                    rt_code = 3
        
            G.add_node(node, type=type_code, cross_type=cross_type_code, signal_violation=sv_code, retrograde_type=rt_code)
        else:
            G.add_node(node, type=0, cross_type=0, signal_violation=0, retrograde_type=0)
    
    # 添加边及其属性
    for parent_id, edges in cg.items():
        for child_id, edge_attrs in edges:
            # 为边属性创建适当的数值编码
            conflict_type = 0
            is_reverse = False
            is_signal_violation = False
            
            # 遍历边属性列表，查找关键词并编码
            for attr in edge_attrs:
                if 'following' in attr.lower():
                    conflict_type = 1
                elif 'crossing' in attr.lower():
                    conflict_type = 2
                elif 'turn' in attr.lower():
                    conflict_type = 3
                elif 'converging' in attr.lower():
                    conflict_type = 4
                elif 'diverging' in attr.lower():
                    conflict_type = 5
                elif 'opposite' in attr.lower():
                    conflict_type = 6
                elif 'start' in attr.lower():
                    conflict_type = 7
                elif 'end' in attr.lower():
                    conflict_type = 8
                elif 'retrograde' in attr.lower() or 'reverse' in attr.lower():
                    is_reverse = True
                elif 'running' in attr.lower():
                    is_signal_violation = True
            
            # 添加边及其属性
            G.add_edge(parent_id, child_id, conflict_type=conflict_type, is_reverse=is_reverse, is_signal_violation=is_signal_violation)
    
    node_labels = {}
    for node, attrs in G.nodes(data=True):
        node_labels[node] = (attrs.get('type', 0), attrs.get('cross_type', 0), attrs.get('signal_violation', 0), attrs.get('retrograde_type', 0))
    
    edge_labels = {}
    for u, v, attrs in G.edges(data=True):
        edge_labels[(u, v)] = (attrs.get('conflict_type', 0), int(attrs.get('is_reverse', False)), int(attrs.get('is_signal_violation', False)))
    
    grakel_graph = Graph(G.edges(), 
                            node_labels=node_labels,
                            edge_labels=edge_labels)
    
    return grakel_graph

def get_agent_info(fragment_id, tp_id):
    """
    Get the information of the agent
    
    Args:
        fragment_id: Scenario ID
        tp_id: Target vehicle ID
    """
    track = tp_info[fragment_id].get(tp_id, None)
    if track is None:
        return None
    agent_type = track['Type']
    agent_class = track['Class']
    cross_type = track['CrossType']
    signal_violation = track['Signal_Violation_Behavior']
    retrograde_type = track.get('retrograde_type', None)
    cardinal_direction = track.get('cardinal direction', None)
    return (agent_type, agent_class, cross_type, signal_violation, retrograde_type, cardinal_direction)

def convert_cgs_to_grakel_graphs(cgs_dict):
    """
    将多个因果图转换为GraKeL图列表
    
    Args:
        cgs_dict: 字典，键为(fragment_id, ego_id)，值为对应的因果图dict
        
    Returns:
        grakel_graphs: GraKeL图列表
        scene_ids: 与grakel_graphs对应的场景ID列表
    """
    grakel_graphs = []
    scene_ids = []
    
    for (fragment_id, ego_id), cg in cgs_dict.items():
        grakel_graph = convert_cg_to_grakel(cg, fragment_id)
        
        if grakel_graph is not None:
            grakel_graphs.append(grakel_graph)
            scene_ids.append(fragment_id + '_' + str(ego_id))
    
    return grakel_graphs, scene_ids
    
def cluster_causal_graphs(cgs_dict, kernel_type='wl', n_clusters=None, cluster_method='spectral', save_path=None):
    """
    对多个因果图进行聚类分析
    
    Args:
        cgs_dict: 字典，键为场景标识符，值为对应的因果图dict
        kernel_type: 要使用的图核类型，可选'wl', 'vertex', 'edge', 'sp'
        n_clusters: 聚类数量，如果为None则自动选择
        cluster_method: 聚类方法，可选'kmeans', 'spectral'
        
    Returns:
        labels: 聚类标签
        cluster_stats: 聚类统计信息
        scene_ids: 与labels对应的场景ID列表
    """
    print("步骤1: 将因果图转换为GraKeL格式...")
    grakel_graphs, scene_ids = convert_cgs_to_grakel_graphs(cgs_dict)
    
    if not grakel_graphs:
        print("错误: 没有有效的因果图用于聚类")
        return None, None, None
    
    print(f"步骤2: 使用{kernel_type}核计算图间相似度...")

    if kernel_type == 'wl':
        # Weisfeiler-Lehman核考虑图的结构和节点标签
        kernel = WeisfeilerLehman(n_iter=5)
    elif kernel_type == 'vertex':
        # 仅考虑节点标签分布
        kernel = VertexHistogram()
    elif kernel_type == 'edge':
        # 仅考虑边标签分布
        kernel = EdgeHistogram()
    elif kernel_type == 'sp':
        # 最短路径核
        kernel = ShortestPath()
    else:
        raise ValueError(f"不支持的核类型: {kernel_type}")
    
    # 计算核矩阵
    kernel_matrix = kernel.fit_transform(grakel_graphs)
    
    if n_clusters is None:
        print("步骤3: 确定最佳聚类数量...")
        silhouette_scores = []
        
        max_clusters = min(10, len(grakel_graphs) - 1)
        if max_clusters < 2:
            n_clusters = 1
            print("图数量过少，设置聚类数为1")
        else:
            for k in range(2, max_clusters + 1):
                try:
                    if cluster_method == 'kmeans':
                        # 使用K-means聚类
                        # 确保核矩阵是半正定的
                        kernel_matrix_adjusted = kernel_matrix.astype(np.float64)
                        min_eig = np.min(np.linalg.eigvals(kernel_matrix_adjusted))
                        if min_eig < 0:
                            kernel_matrix_adjusted -= min_eig * np.eye(kernel_matrix_adjusted.shape[0])
                        
                        clusterer = KMeans(n_clusters=k, random_state=42)
                        labels = clusterer.fit_predict(kernel_matrix_adjusted)
                    else:
                        # 使用谱聚类
                        clusterer = SpectralClustering(
                            n_clusters=k, 
                            affinity='precomputed',
                            random_state=42
                        )
                        labels = clusterer.fit_predict(kernel_matrix)
                    
                    # 需要从相似度矩阵转为距离矩阵以计算轮廓系数
                    distance_matrix = 1 - kernel_matrix / np.max(kernel_matrix)
                    # 将对角线元素设置为0，避免计算轮廓系数时出错
                    np.fill_diagonal(distance_matrix, 0)
                    score = silhouette_score(distance_matrix, labels, metric='precomputed')
                    silhouette_scores.append((k, score))
                    print(f"聚类数 {k}, 轮廓系数: {score:.4f}")
                except Exception as e:
                    print(f"计算聚类数 {k} 的轮廓系数时出错: {e}")
            
            if silhouette_scores:
                n_clusters = max(silhouette_scores, key=lambda x: x[1])[0]
            else:
                n_clusters = 2
            
            print(f"最佳聚类数量: {n_clusters}")
    
    print(f"步骤4: 使用{cluster_method}方法进行聚类...")
    if cluster_method == 'kmeans':
        kernel_matrix_adjusted = kernel_matrix.astype(np.float64)
        min_eig = np.min(np.linalg.eigvals(kernel_matrix_adjusted))
        if min_eig < 0:
            kernel_matrix_adjusted -= min_eig * np.eye(kernel_matrix_adjusted.shape[0])
        
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        labels = clusterer.fit_predict(kernel_matrix_adjusted)
    elif cluster_method == 'spectral':
        clusterer = SpectralClustering(
            n_clusters=n_clusters, 
            affinity='precomputed',
            random_state=42
        )
        labels = clusterer.fit_predict(kernel_matrix)
    else:
        raise ValueError(f"不支持的聚类方法: {cluster_method}")
    
    print("步骤5: 分析聚类结果...")
    cluster_stats = {}
    for i in range(n_clusters):
        cluster_indices = [j for j, label in enumerate(labels) if label == i]
        cluster_size = len(cluster_indices)
        cluster_scene_ids = [scene_ids[j] for j in cluster_indices]
        
        cluster_stats[f"Cluster_{i}"] = {
            "size": cluster_size,
            "percentage": cluster_size / len(grakel_graphs) * 100,
            "scene_ids": cluster_scene_ids
        }
        
        print(f"聚类 {i}: 包含 {cluster_size} 个场景 ({cluster_size / len(grakel_graphs) * 100:.2f}%)")
        print(f"  场景IDs: {cluster_scene_ids}")
    
    print("步骤6: 可视化聚类结果...")
    visualize_graph_clusters(kernel_matrix, labels, scene_ids, save_path)
    
    return labels, cluster_stats, scene_ids

def visualize_graph_clusters(kernel_matrix, labels, ids=None, save_path=None):
    """
    可视化图聚类结果
    
    Args:
        kernel_matrix: 图间的核相似度矩阵
        labels: 聚类标签
        ids: 与标签对应的标识符列表
    """
    
    # # 获取样本数量
    # n_samples = len(labels)
    
    # # 计算合适的perplexity值，确保小于样本数量
    # # 一般建议perplexity在5-50之间，但必须小于样本数
    # perplexity = min(30, max(5, n_samples // 5))

    # 创建简单的散点图
    plt.figure(figsize=(10, 6))
    for i, label in enumerate(labels):
        plt.scatter(i, 0, c=[label], cmap='viridis', s=100, alpha=0.8)
        if ids:
            plt.annotate(str(ids[i]), (i, 0.2), fontsize=9, ha='center')
        else:
            plt.annotate(str(i), (i, 0.2), fontsize=9, ha='center')
    plt.title('Causal Graph Clustering Results')
    plt.xlabel('Graph Index')
    plt.yticks([])
    plt.colorbar(label='Cluster')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, 'graph_cluster.png'))
        print(f"聚类结果已保存到: {save_path}")
    
    # # 如果样本数过少，无法使用t-SNE
    # if n_samples <= 10:
    #     print(f"警告：样本数量过少({n_samples})，无法使用t-SNE进行可视化。")
    #     # 创建简单的散点图
    #     plt.figure(figsize=(10, 6))
    #     for i, label in enumerate(labels):
    #         plt.scatter(i, 0, c=[label], cmap='viridis', s=100, alpha=0.8)
    #         if ids:
    #             plt.annotate(str(ids[i]), (i, 0.2), fontsize=9, ha='center')
    #         else:
    #             plt.annotate(str(i), (i, 0.2), fontsize=9, ha='center')
    #     plt.title('Causal Graph Clustering Results')
    #     plt.xlabel('Graph Index')
    #     plt.yticks([])
    #     plt.colorbar(label='Cluster')
    #     plt.grid(True, linestyle='--', alpha=0.7)
    #     plt.tight_layout()
    #     save_path = os.path.join('output/tj/dep2_noparallel', "causal_graph_clusters.png")
    #     plt.savefig(save_path)
    #     plt.show()
    #     print(f"聚类结果已保存到: {save_path}")
    #     return
    
    # # t-SNE降维，使用自适应的perplexity
    # tsne = TSNE(n_components=2, 
    #             random_state=42, 
    #             metric='precomputed',
    #             perplexity=perplexity)
    
    # # 将相似度矩阵转换为距离矩阵
    # distance_matrix = 1 - kernel_matrix / np.max(kernel_matrix)
    # 将对角线元素设置为0
    # np.fill_diagonal(distance_matrix, 0)
    
    # # 应用t-SNE
    # embeddings = tsne.fit_transform(distance_matrix)
    
    # # 绘制结果
    # plt.figure(figsize=(12, 10))
    # scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap='viridis', s=100, alpha=0.8)
    # plt.colorbar(scatter, label='Cluster')
    # plt.title(f'Causal Graph Clustering Visualization (t-SNE, perplexity={perplexity})')
    # plt.xlabel('t-SNE Dimension 1')
    # plt.ylabel('t-SNE Dimension 2')
    
    # for i, (x, y) in enumerate(embeddings):
    #     if ids:
    #         plt.annotate(str(ids[i]), (x, y), fontsize=9)
    #     else:
    #         plt.annotate(str(i), (x, y), fontsize=9)
            
    # plt.grid(True, linestyle='--', alpha=0.7)
    # plt.tight_layout()
    
    # save_path = os.path.join('output/tj/dep2_noparallel', "causal_graph_clusters.png")
    # plt.savefig(save_path)
    # plt.show()
    # print(f"聚类结果已保存到: {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/tj")
    parser.add_argument("--save_dir", type=str, default="output/tj/dep2_long")
    args = parser.parse_args()

    data_dir = args.data_dir
    save_dir = args.save_dir
    track_change, tp_info, frame_data, frame_data_processed = None, None, None, None
    # with open(os.path.join(data_dir, "track_change_tj.pkl"), "rb") as f:
    #     track_change = pickle.load(f)
    with open(os.path.join(data_dir, "tp_info_tj.pkl"), "rb") as f:
        tp_info = pickle.load(f)
    # with open(os.path.join(data_dir, "frame_data_tj.pkl"), "rb") as f:
    #     frame_data = pickle.load(f)
    # with open(os.path.join(data_dir, "frame_data_tj_processed.pkl"), "rb") as f:
    #     frame_data_processed = pickle.load(f)

    cgs_dict = load_cgs(save_dir)

    cluster_causal_graphs(cgs_dict, n_clusters=4, cluster_method='spectral', save_path=save_dir)
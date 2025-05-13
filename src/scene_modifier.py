"""
场景修改器：用于读取因果图并添加新的节点和边
"""
import os
import sys
import yaml
import pickle
import json
import random
from typing import Dict, List, Tuple, Set
import matplotlib.pyplot as plt
from utils.visualization_utils import create_gif_from_scenario
from utils import reverse_cardinal_direction
from src.agent import Agent

class SceneModifier:
    def __init__(self, data_dir: str, output_dir: str = None, conflict_type_config: str = None):
        """
        初始化场景修改器
        
        Args:
            data_dir: 数据目录路径
            output_dir: 输出目录路径
        """
        self.data_dir = data_dir
        self.output_dir = output_dir if output_dir else os.path.join(data_dir, 'modified_scenes')
        os.makedirs(self.output_dir, exist_ok=True)
        self.conflict_type_config = conflict_type_config if conflict_type_config else "./configs/conflict_type.yaml"

        self.risk_events = None
        self.cg = None
        self.fragment_id = None
        self.ego_id = None
        self.frame_data = None
        self.frame_data_processed = None
        self.conflict_types = {}

    def load_all(self, fragment_id: str, ego_id: int):
        self.load_data()
        self.load_conflict_types()
        self.load_risk_events(fragment_id, ego_id)
        self.load_cg(fragment_id, ego_id)

    def load_conflict_types(self):
        yaml_file_path = self.conflict_type_config
        try:
            if not os.path.exists(yaml_file_path):
                print(f"Configuration file not found: {yaml_file_path}")
                self.conflict_types = {}
                return
            
            with open(yaml_file_path, 'r') as file:
                yaml_data = yaml.safe_load(file)
            
            self.conflict_types = yaml_data
    
        except Exception as e:
            print(f"Error reading YAML file: {e}")
        
    def load_data(self):
        """加载原始数据"""
        with open(os.path.join(self.data_dir, "track_change_tj.pkl"), "rb") as f:
            self.track_change = pickle.load(f)
        with open(os.path.join(self.data_dir, "tp_info_tj.pkl"), "rb") as f:
            self.tp_info = pickle.load(f)
        with open(os.path.join(self.data_dir, "frame_data_tj.pkl"), "rb") as f:
            self.frame_data = pickle.load(f)
        with open(os.path.join(self.data_dir, "frame_data_tj_processed.pkl"), "rb") as f:
            self.frame_data_processed = pickle.load(f)
        
    def load_risk_events(self, fragment_id: str, ego_id: int) -> bool:
        """
        读取保存的风险数据
        
        Args:
            fragment_id: 片段ID
            ego_id: 自车ID
            
        Returns:
            bool: 是否成功加载数据
        """
        save_path = os.path.join(self.output_dir, f"{fragment_id}_{ego_id}")
        risk_events_file = os.path.join(save_path, f"risk_events_data_{fragment_id}_{ego_id}.pkl")
        
        if not os.path.exists(risk_events_file):
            print(f"风险事件文件未找到: {risk_events_file}")
            return False
        
        try:
            with open(risk_events_file, "rb") as f:
                self.risk_events = pickle.load(f)
            self.fragment_id = fragment_id
            self.ego_id = ego_id
            print(f"成功加载风险事件: {risk_events_file}")
            return True
        except Exception as e:
            print(f"加载风险事件时出错: {e}")
            return False
        
    def load_cg(self, fragment_id: str, ego_id: int) -> bool:
        """
        从JSON文件加载因果图
        
        Args:
            fragment_id: 片段ID
            ego_id: 自车ID
            
        Returns:
            bool: 是否成功加载数据
        """
        save_path = os.path.join(self.output_dir, f"{fragment_id}_{ego_id}")
        json_file = os.path.join(save_path, f"cg_{fragment_id}_{ego_id}.json")
        
        if not os.path.exists(json_file):
            print(f"因果图文件未找到: {json_file}")
            return False
            
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                serializable_cg = json.load(f)
                
            # 将JSON数据转换回原始格式
            self.cg = {}
            for parent_id, edges in serializable_cg.items():
                parent_id = int(parent_id) if parent_id.isdigit() else parent_id
                self.cg[parent_id] = []
                for edge in edges:
                    self.cg[parent_id].append((
                        int(edge['child_id']) if edge['child_id'].isdigit() else edge['child_id'],
                        edge['edge_attributes']
                    ))
                    
            self.fragment_id = fragment_id
            self.ego_id = ego_id
            print(f"成功加载因果图: {json_file}")
            return True
            
        except Exception as e:
            print(f"加载因果图时出错: {e}")
            return False
        
    def generate_agent(self, agent_id: int, edge: list) -> Agent:
        agent_info = {'id': agent_id}
        agent_info['agent_type'] = 'mv'
        agent_info['agent_class'] = 'car'
        # TODO
        agent_info['signal_violation'] = 'No violation of traffic lights'

        for (child_id, edge_attr_list) in edge:
            for edge_attr in edge_attr_list:
                # Without signal lights data, we cannot generate scenes with signal violations (unless with extra assumptions)
                # if edge_attr in self.conflict_types['signal_violation']:
                #     agent_info['signal_violation'] = edge_attr
                if edge_attr in self.conflict_types['retrograde']:
                    agent_info['retrograde_type'] = edge_attr
                
                if not agent_info:
                    # Determine cross_type
                    if edge_attr in self.conflict_types['head2tail_types']:
                        child_info = self.get_agent_info(child_id)
                        if edge_attr == 'following':
                            agent_info['cross_type'] = child_info[2]
                            agent_info['cardinal_direction'] = child_info[5]
                        # TODO: determine the cardinal direction and cross type under diverging and converging conditions
                        # For now, cross_type remains the same as child node.
                        elif edge_attr == 'diverging':
                            agent_info['cross_type'] = child_info[2]
                            agent_info['cardinal_direction'] = 'NaN_NaN'
                        elif edge_attr == 'converging':
                            pass
                        else:
                            agent_info['cross_type'] = child_info[2]
                            agent_info['cardinal_direction'] = "NaN_NaN"

                else:
                    # This is an easy implementation
                    agent_info['cross_type'] = child_info[2]
                    agent_info['cardinal_direction'] = reverse_cardinal_direction(child_info[5])
                    agent_info['agent_type'] = 'nmv'
                    agent_info['agent_type'] = 'mv'

        return Agent.from_dict(agent_info)
                        
    def generate_conflict(self):
        if not self.conflict_types:
            print("Please load conflict types first.")

        conflict = {}
        edge_attributes = []
        for k, v in self.conflict_types.items():
            # TODO: head2head_types generation
            if k != 'pedestrian' and k != 'head2head_types' and k != 'unusual_behavior' and k != 'signal_violation':
                conflict[k] = random.choice(v)
                edge_attributes.append(conflict[k])

        return edge_attributes

    def generate_track(self, agent: Agent, edges: list):
        pass
            
    # def get_all_nodes(self) -> Set[str]:
    #     """
    #     获取risk_events中的所有节点
        
    #     Returns:
    #         Set[str]: 所有节点的集合
    #     """
    #     if not self.risk_events:
    #         return set()
            
    #     nodes = set()
    #     for agent, influenced_agents in self.risk_events.items():
    #         nodes.add(agent)
    #         for target_id, _, _ in influenced_agents:
    #             nodes.add(target_id)
    #     return nodes

    def get_agent_info(self, tp_id):
        """
        Get the information of the agent
        
        Args:
            fragment_id: Scenario ID
            tp_id: Target vehicle ID
        """
        fragment_id = self.fragment_id
        track = self.tp_info[fragment_id].get(tp_id, None)
        if track is None:
            return None
        agent_type = track['Type']
        agent_class = track['Class']
        cross_type = track['CrossType']
        signal_violation = track['Signal_Violation_Behavior']
        retrograde_type = track.get('retrograde_type', None)
        cardinal_direction = track.get('cardinal direction', None)
        return (agent_type, agent_class, cross_type, signal_violation, retrograde_type, cardinal_direction)

    def get_all_nodes(self) -> Set[str]:
        """
        获取因果图中的所有节点
        
        Returns:
            Set[str]: 所有节点的集合
        """
        if not self.cg:
            return set()
            
        nodes = set()
        for parent_id, edges in self.cg.items():
            nodes.add(parent_id)
            for child_id, _ in edges:
                nodes.add(child_id)
        return nodes
        
    # def add_random_node_and_edge(self) -> Tuple[str, str, str, List[int]]:
    #     """
    #     随机添加一个新节点和一条指向现有节点的边
        
    #     Returns:
    #         Tuple[str, str, str, List[int]]: (新节点ID, 目标节点ID, 安全指标类型, 关键帧列表)
    #     """
    #     if not self.risk_events:
    #         raise ValueError("请先加载风险事件数据")
            
    #     # 获取所有现有节点
    #     existing_nodes = self.get_all_nodes()
    #     if not existing_nodes:
    #         raise ValueError("因果图中没有现有节点")
            
    #     while True:
    #         new_node_id = random.randint(1000, 9999)
    #         if new_node_id not in existing_nodes:
    #             break
                
    #     # 随机选择一个现有节点作为目标
    #     target_node = random.choice(list(existing_nodes))
        
    #     # 随机选择一个安全指标类型
    #     ssm_types = ['tadv', 'ttc2d', 'act', 'distance']
    #     ssm_type = random.choice(ssm_types)
        
    #     # 生成随机的关键帧列表（模拟10-20帧的连续关键帧）
    #     start_frame = random.randint(1, 100)
    #     num_frames = random.randint(10, 20)
    #     critical_frames = list(range(start_frame, start_frame + num_frames))
        
    #     # 添加新节点和边
    #     if new_node_id not in self.risk_events:
    #         self.risk_events[new_node_id] = []
    #     self.risk_events[new_node_id].append((target_node, ssm_type, critical_frames))
        
    #     return new_node_id, target_node, ssm_type, critical_frames
        
    def save_modified_risk_events(self) -> str:
        """
        保存修改后的风险事件数据
        
        Returns:
            str: 保存的文件路径
        """
        if not self.risk_events or not self.fragment_id or not self.ego_id:
            raise ValueError("请先加载风险事件数据")
            
        save_path = os.path.join(self.output_dir, f"{self.fragment_id}_{self.ego_id}")
        os.makedirs(save_path, exist_ok=True)
        
        # 生成新的文件名，避免覆盖原始文件
        modified_file = os.path.join(save_path, f"modified_risk_events_{self.fragment_id}_{self.ego_id}.pkl")
        
        with open(modified_file, "wb") as f:
            pickle.dump(self.risk_events, f)
            
        print(f"修改后的风险事件已保存到: {modified_file}")
        return modified_file

    def add_node_and_edge_to_cg(self, new_node_id: int, target_node_id: int, edge_attributes: list = []) -> bool:
        """
        向因果图中添加一个新节点和一条指向现有节点的边
        
        Args:
            new_node_id: 新节点的ID
            target_node_id: 目标节点的ID
            edge_attributes: 边的属性字典
            
        Returns:
            bool: 是否成功添加节点和边
        """
        if not self.cg:
            raise ValueError("请先加载因果图数据")
            
        target_exists = False
        for parent_id, edges in self.cg.items():
            if parent_id == target_node_id:
                target_exists = True
                break
            for edge in edges:
                if edge[0] == target_node_id:
                    target_exists = True
                    break
            if target_exists:
                break
                
        if not target_exists:
            print(f"目标节点 {target_node_id} 不存在于因果图中")
            return False
        
        if not edge_attributes:
            edge_attributes = self.generate_conflict()
            
        if new_node_id not in self.cg:
            self.cg[new_node_id] = []
        for edge in self.cg[new_node_id]:
            if edge[0] == target_node_id:
                print(f"从 {new_node_id} 到 {target_node_id} 的边已存在")
                return False
                
        self.cg[new_node_id].append((target_node_id, edge_attributes))
        print(f"成功添加节点 {new_node_id} 和到 {target_node_id} 的边")
        return True

    def filter_and_visualize_scenario(self, frame_id: int = None):
        """
        过滤并可视化因果图中的节点对应的代理轨迹
        
        Args:
            frame_id: 指定要可视化的帧ID，如果为None则使用第一帧
        """
        if not self.cg or not self.fragment_id:
            raise ValueError("请先加载因果图数据")
            
        if not self.frame_data or not self.frame_data_processed:
            self.load_data()
        
        causal_nodes = self.get_all_nodes()
        track = self.track_change[self.fragment_id][self.ego_id]

        if frame_id is None:
            frame_id = min(self.track_change[self.fragment_id][self.ego_id]['track_info']['frame_id'])

        first_frame = frame_id
        last_frame = min(max(self.track_change[self.fragment_id][self.ego_id]['track_info']['frame_id']), first_frame+100)
        
        # fig, ax = plt.subplots(figsize=(12, 8))
        # plot_roads(ax)
        # plot_vehicle_positions(
        #     track_id=self.ego_id,
        #     track_info=track,
        #     frame_info=filtered_frame_data,
        #     scene_id=self.fragment_id,
        #     frame_id=frame_id,
        #     start_frame=frame_id,
        #     end_frame=frame_id
        # )

        frame_data_to_plot = []
        for frame in range(first_frame, last_frame+1):
            frame_data = self.frame_data_processed[self.fragment_id][frame]

            filtered_frame_data = []
            for agent_id, agent_data in frame_data.items():
                if agent_id in causal_nodes:
                    filtered_frame_data.append({
                        'tp_id': agent_id,
                        'vehicle_info': agent_data
                    })
            frame_data_to_plot.append(filtered_frame_data)
            
        create_gif_from_scenario(
            track,
            frame_data_to_plot,
            self.ego_id,
            self.fragment_id,
            os.path.join(self.output_dir, f"{self.fragment_id}_{self.ego_id}", "simplified_scenario"),
            0
        )
        
        # print(f"已保存因果图中的代理可视化结果到: {save_path}")

    def visualize_cg(self, save_pic=True, save_pdf=False):
        """
        使用Graphviz可视化因果图，避免边标签重合和节点重合问题。
        如果两个节点之间已经存在边，则合并边标签而不是添加新边。

        Args:
            causal_graph: 表示因果图的字典，键为代理ID，值为它们影响的代理ID列表。
            fragment_id: 片段ID
            ego_id: 自车ID
            save_pic: 是否保存PNG格式图片
            save_pdf: 是否保存PDF格式图片
        """
        try:
            import graphviz
        except ImportError:
            print("Please install graphviz: pip install graphviz")
            print("Also need to install system-level Graphviz: https://graphviz.org/download/")
            return
        
        if self.risk_events is None:
            print("Causal graph does not exist. Please extract causal graph before visualization.")
            return
        
        fragment_id = self.fragment_id
        ego_id = self.ego_id
        dot = graphviz.Digraph(comment=f'Causal Graph for Fragment {fragment_id}, Ego {ego_id}')
        dot.attr(rankdir='LR', size='12,8', dpi='300', fontname='Arial', 
                 bgcolor='white', concentrate='true')
        
        dot.attr('node', shape='circle', style='filled', color='black', 
                 fillcolor='skyblue', fontname='Arial', fontsize='10', fontcolor='black',
                 width='1.0', height='1.0', penwidth='1.0', fixedsize='true')
        
        dot.attr('edge', color='black', fontname='Arial', fontsize='12', fontcolor='darkred',
                 penwidth='1.0', arrowsize='0.5', arrowhead='normal')
        
        all_nodes = set()
        for agent, influenced_agents in self.cg.items():
            all_nodes.add(agent)
            for influenced_agent, _ in influenced_agents:
                all_nodes.add(influenced_agent)
        
        for node in all_nodes:
            node_info = self.get_agent_info(node)
            if node_info:
                agent_type, agent_class, cross_type, signal_violation, retrograde_type, cardinal_direction = node_info
                node_label = f"{node}\n{agent_class}\n"
                
                # 添加违规信息（如果有）
                if cross_type:
                    for ct in cross_type:
                        if ct != "Normal":
                            node_label += f"{ct}\n"
                if cardinal_direction is not None:
                    node_label += f"{cardinal_direction}\n"
                # if signal_violation:
                    # for sv in signal_violation:
                        # if sv != "No violation of traffic lights":
                            # node_label += f"{sv}\n"
                # if retrograde_type and retrograde_type != "normal" and retrograde_type != "unknown":
                    # node_label += f"\n{retrograde_type}"
            else:
                node_label = f"{node}"
            
            if node == ego_id:
                dot.node(str(node), node_label, fillcolor='lightcoral', fontcolor='white')
            else:
                dot.node(str(node), node_label)
        
        edge_dict = {}
        for agent, influenced_agents in self.cg.items():
            for influenced_agent, edge_attr in influenced_agents:
                edge_key = (str(agent), str(influenced_agent))
                
                if edge_key not in edge_dict:
                    edge_dict[edge_key] = edge_attr
        
        # Merge labels and add edge
        for (src, dst), labels in edge_dict.items():
            if len(labels) > 0:
                combined_label = "\n".join(labels)
            else:
                combined_label = "No obvious reason"
            
            dot.edge(src, dst, label=combined_label, minlen='2')
        
        dot.attr(label=f'Graph (Fragment: {fragment_id}, Ego: {ego_id})')
        dot.attr(fontsize='20')
        dot.attr(labelloc='t')

        save_path = os.path.join(self.output_dir, f"{fragment_id}_{ego_id}")
        os.makedirs(save_path, exist_ok=True)     
        dot_path = os.path.join(save_path, f"modified_cg_{fragment_id}_{ego_id}")
        if save_pic:
            dot.render(dot_path, format='png', cleanup=True)
            print(f"Causal graph saved to {dot_path}.png")
        if save_pdf:
            dot.render(dot_path, format='pdf', cleanup=True)
            print(f"Causal graph saved to {dot_path}.pdf")

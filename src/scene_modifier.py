"""
场景修改器：用于读取因果图并添加新的节点和边
"""
import os
import sys
import yaml
import pickle
import json
import random
import math
import numpy as np
from typing import Dict, List, Tuple, Set
import matplotlib.pyplot as plt
from utils.visualization_utils import create_gif_from_scenario
from src.my_utils import reverse_cardinal_direction, infer_cardinal_direction
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
        self.typical_tracks = {}
        self.typical_tracks_frag = "7_28_1 R21"

    def load_all(self, fragment_id: str, ego_id: int):
        if not self.frame_data:
            self.load_data()
        if not self.conflict_types:
            self.load_conflict_types()
        if not self.typical_tracks:
            self.load_typical_tracks()
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

    def load_typical_tracks(self):
        """
        加载包含交叉类型数据的JSON文件
        
        返回:
            dict: 加载的交叉类型字典
        """
        file_path = os.path.join(self.data_dir, "typical_track.json")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                self.typical_tracks = json.load(f)
            print(f"成功加载文件: {file_path}")
        except FileNotFoundError:
            print(f"错误: 文件 '{file_path}' 不存在")
        except json.JSONDecodeError:
            print(f"错误: 文件 '{file_path}' 不是有效的JSON格式")
        except Exception as e:
            print(f"加载文件时发生错误: {str(e)}")
        
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
                    child_info = self.get_agent_info(child_id)
                    if edge_attr in self.conflict_types['head2tail_types']:
                        if edge_attr == 'following':
                            agent_info['cross_type'] = child_info[2]
                            agent_info['cardinal_direction'] = child_info[5]
                        # TODO: determine the cardinal direction and cross type under diverging and converging conditions
                        # For now, cross_type remains the same as child node.
                        elif edge_attr == 'diverging':
                            agent_info['cross_type'] = child_info[2]
                            [start, end] = child_info[5].split('_')
                            agent_info['cardinal_direction'] = start + '_NaN'
                        elif edge_attr == 'converging':
                            return NotImplementedError
                        else:
                            agent_info['cross_type'] = child_info[2]
                            agent_info['cardinal_direction'] = "NaN_NaN"

                    elif edge_attr in self.conflict_types['head2head_types']:
                        return NotImplementedError
                else:
                    child_info = self.get_agent_info(child_id)
                    # This is an easy implementation
                    agent_info['cross_type'] = child_info[2]
                    agent_info['cardinal_direction'] = reverse_cardinal_direction(child_info[5])

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

    def generate_track(self, agent=None, edges: list = None):
        """
        根据agent信息和边属性生成轨迹数据
        
        Args:
            agent: 代理对象，如果为None则根据edges生成
            edges: 边属性列表，如果为None则随机生成
            
        Returns:
            dict: 生成的轨迹数据
        """
        if not edges:
            edges = self.generate_conflict()
        if not agent:
            agent = self.generate_agent(1001, edges)
        
        if not self.fragment_id or not self.frame_data_processed:
            raise ValueError("请先加载数据")
            
        # 获取原始帧数据范围
        ego_track = self.track_change[self.fragment_id][self.ego_id]
        start_frame = min(ego_track['track_info']['frame_id'])
        end_frame = max(ego_track['track_info']['frame_id'])
        frame_count = end_frame - start_frame + 1
        
        # 创建新轨迹数据结构
        track_data = {
            'track_info': {
                'frame_id': [],
                'x': [],
                'y': [],
                'vx': [],
                'vy': [],
                'ax': [],
                'ay': [],
                'heading_rad': [],
                'width': [],
                'length': []
            },
            'anomalies': np.zeros(frame_count, dtype=bool),
            'Type': agent.agent_type,
            'Class': agent.agent_class,
            'CrossType': agent.cross_type,
            'Signal_Violation_Behavior': agent.signal_violation,
            'retrograde_type': agent.retrograde_type,
            'cardinal direction': agent.cardinal_direction,
            'num': 6  # 默认编号
        }
        
        # 根据边属性和冲突类型生成轨迹
        # 找到目标节点的轨迹作为参考
        target_node_id = None
        edge_attributes = []
        if isinstance(edges, list) and len(edges) > 0:
            edge_attributes = edges
        else:
            for child_id, attrs in edges:
                target_node_id = child_id
                edge_attributes = attrs
                break

        # 根据冲突类型生成轨迹
        conflict_type = None
        for attr in edge_attributes:
            if attr in self.conflict_types['head2tail_types']:
                conflict_type = attr
                break
            elif attr in self.conflict_types['head2head_types']:
                conflict_type = attr
                break
            elif attr in self.conflict_types['unusual_behavior']:
                conflict_type = attr
                break
                
        # 默认为跟随行为
        if not conflict_type:
            conflict_type = 'following'
        
        # 如果没有目标节点，使用ego
        if not target_node_id:
            target_node_id = self.ego_id
            
        # 获取目标节点的轨迹
        target_frames = self.frame_data_processed[self.fragment_id]
        target_track = []
        
        # 收集目标节点在所有帧中的位置数据
        for frame_id in range(start_frame, end_frame + 1):
            if frame_id < len(target_frames) and target_node_id in target_frames[frame_id]:
                target_data = target_frames[frame_id][target_node_id]
                target_track.append({
                    'frame_id': frame_id,
                    'x': target_data['x'],
                    'y': target_data['y'],
                    'vx': target_data['vx'],
                    'vy': target_data['vy'],
                    'heading_rad': target_data.get('heading_rad', 0)
                })
                
        if not target_track:
            raise ValueError(f"无法获取目标节点 {target_node_id} 的轨迹数据")
        
        # get reference track
        reference_track_id = 0
        if conflict_type in self.conflict_types['head2tail_types']:
            cd = agent.cardinal_direction
            ct = agent.cross_type[0]

            for k in self.typical_tracks.keys():
                if k in ct.lower():
                    reference_track_id = self.typical_tracks[k].get(cd, [])

        if reference_track_id:
            # reference_track_id = random.choice(reference_track_id)
            reference_track_id = reference_track_id[1]

        reference_track = []
        
        frame_shift = 20
        aligned_frame_id = start_frame - frame_shift
        ref_start_frame = min(self.tp_info[self.typical_tracks_frag][reference_track_id]['State']['frame_id'])
        ref_end_frame = max(self.tp_info[self.typical_tracks_frag][reference_track_id]['State']['frame_id'])
        for frame_id in range(ref_start_frame, ref_end_frame+1):
            reference_data = self.frame_data_processed[self.typical_tracks_frag][frame_id][reference_track_id]
            reference_track.append({
                    'frame_id': aligned_frame_id,
                    'x': reference_data['x'],
                    'y': reference_data['y'],
                    'vx': reference_data['vx'],
                    'vy': reference_data['vy'],
                    'ax': reference_data['ax'],
                    'ay': reference_data['ay'],
                    'heading_rad': reference_data.get('heading_rad', 0)
                })
            aligned_frame_id += 1
            
        # 生成轨迹点
        offset_x, offset_y = 0, 0
        scale_vx, scale_vy = 1.0, 1.0
        
        # 根据冲突类型调整轨迹生成参数
        if conflict_type == 'following':
            # 跟随行为：在时间轴上向前平移（延迟跟随）
            frame_shift = 30
            
            # 创建时间偏移后的轨迹
            shifted_tracks = []
            if len(target_track) > frame_shift:
                shifted_tracks = target_track[:-frame_shift]  # 截取前面的部分，丢弃最后time_shift帧
                
                # 调整空间位置（仍然保持在目标车辆后方）
                offset_x, offset_y = -1.0, 0
                scale_vx, scale_vy = 1.0, 1.0
            else:
                # 如果轨迹太短，无法进行时间平移，退回到原来的方法
                shifted_tracks = target_track
                offset_x, offset_y = -5, 0
                scale_vx, scale_vy = 1.0, 1.0
                
            target_track = shifted_tracks
        elif conflict_type == 'diverging':
            # 分叉行为：从相同位置开始，然后分开
            offset_x, offset_y = -2, 2  # 稍微错开起始位置
            scale_vx, scale_vy = 1.0, 1.2  # 速度略有不同
        elif conflict_type == 'converging':
            # 汇合行为：从不同位置开始，然后靠近
            offset_x, offset_y = -10, 5  # 开始位置较远
            scale_vx, scale_vy = 1.2, 0.9  # 调整速度使轨迹汇合
        elif 'crossing conflict' in conflict_type:
            # 交叉冲突：轨迹相交
            offset_x, offset_y = 5, -8
            scale_vx, scale_vy = 0.9, 1.1
        elif 'retrograde' in agent.retrograde_type:
            # 逆行行为：速度方向相反
            offset_x, offset_y = 10, 0
            scale_vx, scale_vy = -1.0, -1.0
            
        # 根据参考轨迹和参数生成新轨迹
        for i, ref in enumerate(reference_track):
            # # 加入随机扰动使轨迹更自然
            # noise_x = random.uniform(-0.5, 0.5)
            # noise_y = random.uniform(-0.5, 0.5)
            
            # # 计算新位置
            # new_x = ref['x'] + offset_x + noise_x
            # new_y = ref['y'] + offset_y + noise_y
            new_x = ref['x']
            new_y = ref['y']
            
            # 计算新速度
            new_vx = ref['vx'] * scale_vx
            new_vy = ref['vy'] * scale_vy
            
            # 对于'following'模式，我们将帧ID向后偏移，表示时间轴上的前移
            current_frame_id = ref['frame_id']
            
            # # 计算新朝向（根据速度方向）
            # if new_vx != 0 or new_vy != 0:
            #     new_heading = math.atan2(new_vy, new_vx)
            # else:
            #     new_heading = ref['heading_rad']
            new_heading = ref['heading_rad']
                
            # # 简单计算加速度（如果有连续帧）
            # if i > 0:
            #     prev_vx = track_data['track_info']['vx'][-1]
            #     prev_vy = track_data['track_info']['vy'][-1]
            #     dt = 0.1  # 假设帧率为10Hz
            #     new_ax = (new_vx - prev_vx) / dt
            #     new_ay = (new_vy - prev_vy) / dt
            # else:
            #     new_ax = 0
            #     new_ay = 0
            new_ax = ref['ax']
            new_ay = ref['ay']
                
            # 车辆尺寸（根据agent类型设置）
            if agent.agent_class == 'car':
                width, length = 1.8, 4.5
            elif agent.agent_class == 'bus' or agent.agent_class == 'truck':
                width, length = 2.5, 10.0
            else:
                width, length = 1.5, 3.0
                
            # 添加到轨迹数据中
            track_data['track_info']['frame_id'].append(current_frame_id)
            track_data['track_info']['x'].append(new_x)
            track_data['track_info']['y'].append(new_y)
            track_data['track_info']['vx'].append(new_vx)
            track_data['track_info']['vy'].append(new_vy)
            track_data['track_info']['ax'].append(new_ax)
            track_data['track_info']['ay'].append(new_ay)
            track_data['track_info']['heading_rad'].append(new_heading)
            track_data['track_info']['width'].append(width)
            track_data['track_info']['length'].append(length)
            
        # 转换为numpy数组，与原始数据格式一致
        for key in track_data['track_info']:
            if key != 'frame_id':  # frame_id通常保持为列表
                track_data['track_info'][key] = np.array(track_data['track_info'][key])
        
        return track_data

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

    def filter_and_visualize_scenario(self, frame_id: int = None, new_agents: Dict[int, dict] = None):
        """
        过滤并可视化因果图中的节点对应的代理轨迹，支持显示新添加的代理
        
        Args:
            frame_id: 指定要可视化的帧ID，如果为None则使用第一帧
            new_agents: 新添加的代理轨迹字典，键为代理ID，值为轨迹数据
        """
        if not self.cg or not self.fragment_id:
            raise ValueError("请先加载因果图数据")
            
        if not self.frame_data or not self.frame_data_processed:
            self.load_data()
        
        causal_nodes = self.get_all_nodes()
        track = self.track_change[self.fragment_id][self.ego_id]
        track['num'] = 5

        if frame_id is None:
            frame_id = min(self.track_change[self.fragment_id][self.ego_id]['track_info']['frame_id'])

        first_frame = frame_id
        last_frame = min(max(self.track_change[self.fragment_id][self.ego_id]['track_info']['frame_id']), first_frame+500)

        # 准备原始场景数据
        len_fragment = len(self.frame_data_processed[self.fragment_id])
        frame_data_to_plot = []
        for frame in range(first_frame, last_frame+1):
            if frame >= len_fragment:
                continue
                
            frame_data = self.frame_data_processed[self.fragment_id][frame]

            filtered_frame_data = []
            for agent_id, agent_data in frame_data.items():
                if agent_id in causal_nodes:
                    filtered_frame_data.append({
                        'tp_id': agent_id,
                        'vehicle_info': agent_data
                    })
                    
            # 添加新代理数据（如果有）
            if new_agents:
                for agent_id, track_data in new_agents.items():
                    # 找到当前帧对应的轨迹数据索引
                    if 'track_info' in track_data and 'frame_id' in track_data['track_info']:
                        frame_indices = track_data['track_info']['frame_id']
                        if frame in frame_indices:
                            idx = frame_indices.index(frame)
                            
                            # 构造与原始数据相同格式的单帧数据
                            agent_frame_data = {
                                'x': track_data['track_info']['x'][idx],
                                'y': track_data['track_info']['y'][idx],
                                'vx': track_data['track_info']['vx'][idx],
                                'vy': track_data['track_info']['vy'][idx],
                                'heading_rad': track_data['track_info']['heading_rad'][idx],
                                'width': track_data['track_info']['width'][idx],
                                'length': track_data['track_info']['length'][idx],
                                'frame_id': frame,
                                'agent_type': track_data.get('Type', 'mv'),
                                'agent_class': track_data.get('Class', 'car')
                            }
                            
                            filtered_frame_data.append({
                                'tp_id': agent_id,
                                'vehicle_info': agent_frame_data
                            })
                            
            frame_data_to_plot.append(filtered_frame_data)
        
        # 合并为可视化函数需要的轨迹字典
        tracks_dict = {self.ego_id: track}
        if new_agents:
            for agent_id, track_data in new_agents.items():
                if agent_id not in tracks_dict:
                    tracks_dict[agent_id] = track_data
        
        # 创建输出目录
        output_dir = os.path.join(self.output_dir, f"{self.fragment_id}_{self.ego_id}", "modified_scenario")
        os.makedirs(output_dir, exist_ok=True)
        
        create_gif_from_scenario(
            tracks_dict,
            frame_data_to_plot,
            self.ego_id,
            self.fragment_id,
            output_dir,
            0
        )
        
        print(f"已保存修改后的场景可视化结果到: {output_dir}")

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

    def add_and_visualize_new_agent(self, new_agent: Agent, target_node_id: int = None, edge_attributes: list = None):
        """
        生成新代理，添加到因果图中，并可视化修改后的场景
        
        Args:
            new_agent_id: 新代理的ID
            target_node_id: 目标节点ID，如果为None则随机选择
            edge_attributes: 边属性列表，如果为None则随机生成
            
        Returns:
            tuple: (agent_id, track_data) 生成的代理ID和轨迹数据
        """
        if not self.cg or not self.fragment_id:
            raise ValueError("请先加载因果图数据")
            
        # 获取所有现有节点
        existing_nodes = self.get_all_nodes()
        if not existing_nodes:
            raise ValueError("因果图中没有现有节点")
            
        # 生成新代理ID（如果未提供）
        # if not new_agent:
        #     while True:
        #         new_agent_id = random.randint(1000, 9999)
        #         if new_agent_id not in existing_nodes:
        #             break
                    
        # 选择目标节点（如果未提供）
        if target_node_id is None:
            target_node_id = random.choice(list(existing_nodes))
            
        # 生成边属性（如果未提供）
        if edge_attributes is None:
            edge_attributes = self.generate_conflict()
            
        # 添加节点和边到因果图
        if not self.add_node_and_edge_to_cg(new_agent.id, target_node_id, edge_attributes):
            return None, None
            
        # # 生成代理对象
        # if not new_agent:
        #     agent = self.generate_agent(new_agent_id, [(target_node_id, edge_attributes)])
        # else:
        #     agent = new_agent
        
        # 生成轨迹数据
        track_data = self.generate_track(new_agent, [(target_node_id, edge_attributes)])
        
        # 可视化修改后的场景（包含新代理）
        self.filter_and_visualize_scenario(new_agents={new_agent.id: track_data})
        
        # 更新并保存因果图可视化
        self.visualize_cg()
        
        return new_agent.id, track_data

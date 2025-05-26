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
from utils.visualization_utils import create_gif_from_scenario
from src.my_utils import reverse_cardinal_direction, infer_cardinal_direction
from src.agent import Agent

class ScenarioEditor:
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
        self.tps = None
        self.fragment_id = None
        self.ego_id = None
        self.frame_data = None
        self.frame_data_processed = None
        self.conflict_types = {}
        self.typical_tracks = {}
        self.typical_tracks_nmv = {}
        self.typical_tracks_ped = {}
        self.typical_tracks_frag = "7_28_1 R21"

    def load_all(self, fragment_id: str, ego_id: int):
        if not self.frame_data_processed:
            self.load_data()
        if not self.conflict_types:
            self.load_conflict_types()
        if not self.typical_tracks:
            self.load_typical_tracks()
        # self.load_risk_events(fragment_id, ego_id)
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
        mv_file_path = os.path.join(self.data_dir, "typical_track.json")
        nmv_file_path = os.path.join(self.data_dir, "typical_track_nmv.json")
        ped_file_path = os.path.join(self.data_dir, "typical_track_ped.json")

        try:
            with open(mv_file_path, "r", encoding="utf-8") as f:
                self.typical_tracks = json.load(f)
            with open(nmv_file_path, "r", encoding="utf-8") as f:
                self.typical_tracks_nmv = json.load(f)
            with open(ped_file_path, "r", encoding="utf-8") as f:
                self.typical_tracks_ped = json.load(f)
            print(f"Successfully loaded file: {mv_file_path}")
            print(f"Successfully loaded file: {nmv_file_path}")
            print(f"Successfully loaded file: {ped_file_path}")
        except FileNotFoundError:
            print(f"Error: file not found")
        except json.JSONDecodeError:
            print(f"Error: file is not a valid JSON format")
        except Exception as e:
            print(f"Error: {str(e)}")
        
    def load_data(self):
        """加载原始数据"""
        with open(os.path.join(self.data_dir, "track_change_tj.pkl"), "rb") as f:
            self.track_change = pickle.load(f)
        with open(os.path.join(self.data_dir, "tp_info_tj.pkl"), "rb") as f:
            self.tp_info = pickle.load(f)
        # with open(os.path.join(self.data_dir, "frame_data_tj.pkl"), "rb") as f:
        #     self.frame_data = pickle.load(f)
        with open(os.path.join(self.data_dir, "frame_data_tj_processed.pkl"), "rb") as f:
            self.frame_data_processed = pickle.load(f)
        
    # def load_risk_events(self, fragment_id: str, ego_id: int) -> bool:
    #     """
    #     读取保存的风险数据
        
    #     Args:
    #         fragment_id: 片段ID
    #         ego_id: 自车ID
            
    #     Returns:
    #         bool: 是否成功加载数据
    #     """
    #     save_path = os.path.join(self.output_dir, f"{fragment_id}_{ego_id}")
    #     risk_events_file = os.path.join(save_path, f"risk_events_data_{fragment_id}_{ego_id}.pkl")
        
    #     if not os.path.exists(risk_events_file):
    #         print(f"风险事件文件未找到: {risk_events_file}")
    #         return False
        
    #     try:
    #         with open(risk_events_file, "rb") as f:
    #             self.risk_events = pickle.load(f)
    #         self.fragment_id = fragment_id
    #         self.ego_id = ego_id
    #         print(f"成功加载风险事件: {risk_events_file}")
    #         return True
    #     except Exception as e:
    #         print(f"加载风险事件时出错: {e}")
    #         return False
    def load_graph(self, cg_file: str, ego_id: int):
        """
        从JSON文件加载因果图
        """
        with open(cg_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            serializable_cg = data['causal_graph']
            serializable_agents = data.get('agents', {})
        
        self.cg = {}
        for parent_id, edges in serializable_cg.items():
            parent_id = int(parent_id) if parent_id.isdigit() else parent_id
            self.cg[parent_id] = []
            for edge in edges:
                self.cg[parent_id].append((
                    int(edge['child_id']) if edge['child_id'].isdigit() else edge['child_id'],
                    edge['edge_attributes']
                ))
        
        # 加载agents字典
        self.tps = {}
        for agent_id, agent_dict in serializable_agents.items():
            agent_id = int(agent_id) if agent_id.isdigit() else agent_id
            self.tps[agent_id] = Agent.from_dict(agent_dict)
        
        self.fragment_id = None
        self.ego_id = ego_id
        print(f"成功加载因果图: {cg_file}")
        return True
        
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
                data = json.load(f)
                serializable_cg = data['causal_graph']
                serializable_agents = data.get('agents', {})
                
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
            
            # 加载agents字典
            self.tps = {}
            for agent_id, agent_dict in serializable_agents.items():
                agent_id = int(agent_id) if agent_id.isdigit() else agent_id
                self.tps[agent_id] = Agent.from_dict(agent_dict)
            
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

    def generate_track(self, 
                       agent=None, 
                       edges: list = None, 
                       frame_shift: int = None, 
                       reference_id: int = 1, 
                       offset = None, 
                       start_frame = None):
        """
        根据agent信息和边属性生成轨迹数据
        
        Args:
            agent: 代理对象，如果为None则根据edges生成
            edges: 边属性列表，如果为None则随机生成
            
        Returns:
            dict: 生成的轨迹数据
        """
        # if not edges:
        #     edges = self.generate_conflict()
        if not agent:
            agent = self.generate_agent(1001, edges)
        
        if not self.frame_data_processed:
            raise ValueError("Please load data first.")
            
        # 获取原始帧数据范围
        if start_frame is None:
            ego_track = self.track_change[self.fragment_id][self.ego_id]
            start_frame = min(ego_track['track_info']['frame_id'])
            # end_frame = max(ego_track['track_info']['frame_id'])
            # frame_count = end_frame - start_frame + 1
        
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
            # 'anomalies': np.zeros(frame_count, dtype=bool),
            'Type': agent.agent_type,
            'Class': agent.agent_class,
            'CrossType': agent.cross_type,
            'Signal_Violation_Behavior': agent.signal_violation,
            'retrograde_type': agent.retrograde_type,
            'cardinal direction': agent.cardinal_direction,
            'num': 6
        }
        
        # 根据边属性和冲突类型生成轨迹
        # 找到目标节点的轨迹作为参考
        # target_node_id = None
        # edge_attributes = []
        # if isinstance(edges, list) and len(edges) > 0:
        #     edge_attributes = edges
        # else:
        #     for child_id, attrs in edges:
        #         target_node_id = child_id
        #         edge_attributes = attrs
        #         break

        # # 根据冲突类型生成轨迹
        # conflict_type = None
        # for attr in edge_attributes:
        #     if attr in self.conflict_types['head2tail_types']:
        #         conflict_type = attr
        #         break
        #     elif attr in self.conflict_types['head2head_types']:
        #         conflict_type = attr
        #         break
        #     elif attr in self.conflict_types['unusual_behavior']:
        #         conflict_type = attr
        #         break
                
        # # 默认为跟随行为
        # if not conflict_type:
        #     conflict_type = 'following'
        
        # # 如果没有目标节点，使用ego
        # if not target_node_id:
        #     target_node_id = self.ego_id
            
        # 获取目标节点的轨迹
        # target_frames = self.frame_data_processed[self.fragment_id]
        # target_track = []
        
        # # 收集目标节点在所有帧中的位置数据
        # for frame_id in range(start_frame, end_frame + 1):
        #     if frame_id < len(target_frames) and target_node_id in target_frames[frame_id]:
        #         target_data = target_frames[frame_id][target_node_id]
        #         target_track.append({
        #             'frame_id': frame_id,
        #             'x': target_data['x'],
        #             'y': target_data['y'],
        #             'vx': target_data['vx'],
        #             'vy': target_data['vy'],
        #             'heading_rad': target_data.get('heading_rad', 0)
        #         })
                
        
        # get reference track
        reference_track_id = 0
        type = agent.agent_type
        cd = agent.cardinal_direction

        if type == 'mv':
            ct = agent.cross_type[0]
            for k in self.typical_tracks.keys():
                if k in ct.lower():
                    reference_track_id = self.typical_tracks[k].get(cd, [])
        elif type == 'nmv':
            ct = agent.cross_type[0]
            for k in self.typical_tracks_nmv.keys():
                if k in ct.lower():
                    reference_track_id = self.typical_tracks_nmv[k].get(cd, [])
        elif type == 'ped':
            # Pedestrians always go straight
            reference_track_id = self.typical_tracks_ped['straight']

        if reference_track_id:
            # reference_track_id = random.choice(reference_track_id)
            reference_track_id = reference_track_id[reference_id]

        reference_track = []
        
        if not frame_shift:
            frame_shift = 0
        aligned_frame_id = start_frame + frame_shift
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
                
        #     target_track = shifted_tracks
        # elif conflict_type == 'diverging':
        #     # 分叉行为：从相同位置开始，然后分开
        #     offset_x, offset_y = -2, 2  # 稍微错开起始位置
        #     scale_vx, scale_vy = 1.0, 1.2  # 速度略有不同
        # elif conflict_type == 'converging':
        #     # 汇合行为：从不同位置开始，然后靠近
        #     offset_x, offset_y = -10, 5  # 开始位置较远
        #     scale_vx, scale_vy = 1.2, 0.9  # 调整速度使轨迹汇合
        # elif 'retrograde' in agent.retrograde_type:
        #     # 逆行行为：速度方向相反
        #     offset_x, offset_y = 10, 0
        #     scale_vx, scale_vy = -1.0, -1.0
            
        # 根据参考轨迹和参数生成新轨迹
        for ref in reference_track:
            if offset:
                new_x = ref['x'] + offset[0]
                new_y = ref['y'] + offset[1]
            else:
                new_x = ref['x']
                new_y = ref['y']
            
            new_vx = ref['vx']
            new_vy = ref['vy']
            
            current_frame_id = ref['frame_id']
            new_heading = ref['heading_rad']
                
            new_ax = ref['ax']
            new_ay = ref['ay']
                
            # 车辆尺寸
            if agent.agent_class == 'car':
                width, length = 1.8, 4.5
            elif agent.agent_class == 'bus' or agent.agent_class == 'truck':
                width, length = 2.5, 10.0
            else:
                width, length = 1.5, 3.0
                
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
        
        # # 保存生成的轨迹数据
        # self.save_track(track_data, agent.id)
        
        return track_data

    def save_track(self, track_data: dict, agent_id: int) -> str:
        """
        保存生成的轨迹数据到文件
        
        Args:
            track_data: 轨迹数据字典
            agent_id: 代理ID
            
        Returns:
            str: 保存的文件路径
        """
        if not self.fragment_id or not self.ego_id:
            raise ValueError("请先加载数据")
            
        save_path = os.path.join(self.output_dir, f"{self.fragment_id}_{self.ego_id}", "tracks")
        os.makedirs(save_path, exist_ok=True)
        
        file_path = os.path.join(save_path, f"track_{agent_id}.pkl")
        
        with open(file_path, "wb") as f:
            pickle.dump(track_data, f)
            
        print(f"轨迹数据已保存到: {file_path}")
        return file_path

    def load_track(self, agent_id: int) -> dict:
        """
        加载保存的轨迹数据
        
        Args:
            agent_id: 代理ID
            
        Returns:
            dict: 加载的轨迹数据，如果文件不存在则返回None
        """
        if not self.fragment_id or not self.ego_id:
            raise ValueError("请先加载数据")
            
        file_path = os.path.join(self.output_dir, f"{self.fragment_id}_{self.ego_id}", "tracks", f"track_{agent_id}.pkl")
        
        if not os.path.exists(file_path):
            print(f"轨迹数据文件不存在: {file_path}")
            return None
            
        try:
            with open(file_path, "rb") as f:
                track_data = pickle.load(f)
            print(f"成功加载轨迹数据: {file_path}")
            return track_data
        except Exception as e:
            print(f"加载轨迹数据时出错: {e}")
            return None

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
        
    # def save_modified_risk_events(self) -> str:
    #     """
    #     保存修改后的风险事件数据
        
    #     Returns:
    #         str: 保存的文件路径
    #     """
    #     if not self.risk_events or not self.fragment_id or not self.ego_id:
    #         raise ValueError("请先加载风险事件数据")
            
    #     save_path = os.path.join(self.output_dir, f"{self.fragment_id}_{self.ego_id}")
    #     os.makedirs(save_path, exist_ok=True)
        
    #     # 生成新的文件名，避免覆盖原始文件
    #     modified_file = os.path.join(save_path, f"modified_risk_events_{self.fragment_id}_{self.ego_id}.pkl")
        
    #     with open(modified_file, "wb") as f:
    #         pickle.dump(self.risk_events, f)
            
    #     print(f"修改后的风险事件已保存到: {modified_file}")
    #     return modified_file

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

    def filter_and_visualize_scenario(self, frame_id: int = None, new_agents: Dict[int, dict] = None, output_dir: str = None):
        """
        过滤并可视化因果图中的节点对应的代理轨迹，支持显示新添加的代理
        
        Args:
            frame_id: 指定要可视化的帧ID，如果为None则使用第一帧
            new_agents: 新添加的代理轨迹字典，键为代理ID，值为轨迹数据
        """
        if not self.cg or not self.fragment_id:
            raise ValueError("请先加载因果图数据")
            
        if not self.frame_data_processed:
            self.load_data()
        
        causal_nodes = self.get_all_nodes()
        track = self.track_change[self.fragment_id][self.ego_id]
        track['num'] = 6

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
        
        if output_dir is None:
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

    def visualize_graph(self, save_dir = None, save_pic=True, save_pdf=False):
        """
        使用Graphviz可视化因果图，避免边标签重合和节点重合问题。
        如果两个节点之间已经存在边，则合并边标签而不是添加新边。

        Args:
            save_pic: 是否保存PNG格式图片
            save_pdf: 是否保存PDF格式图片
        """
        try:
            import graphviz
        except ImportError:
            print("Please install graphviz: pip install graphviz")
            print("Also need to install system-level Graphviz: https://graphviz.org/download/")
            return
        
        if self.cg is None:
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
            agent = self.tps.get(node, None)
            if agent:
                node_label = f"{node}\n{agent.agent_class}\n"
                
                # 添加违规信息（如果有）
                if agent.cross_type:
                    for ct in agent.cross_type:
                        if ct != "Normal":
                            node_label += f"{ct}\n"
                if agent.cardinal_direction is not None:
                    node_label += f"{agent.cardinal_direction}\n"
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

        save_path = save_dir if save_dir else os.path.join(self.output_dir, f"{fragment_id}_{ego_id}")
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
        
        # Add new agent to agents dict
        self.tps[new_agent.id] = new_agent
            
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
        self.visualize_graph()
        
        return new_agent.id, track_data

    def add_agents(self, new_agents, new_edges, frame_shift_list):
        """
        不依赖原始场景数据，直接根据给定的因果图和典型轨迹库生成场景。
        轨迹全部从典型轨迹库中获取。

        Args:

        Returns:
            dict: 所有节点的轨迹数据，key为agent_id，value为track_data
        """
        result_tracks = {}
        for (new_agent, new_edge, frame_shift) in zip(new_agents, new_edges, frame_shift_list):
            track_data = self.generate_track(agent=new_agent, edges=new_edge, frame_shift=frame_shift)
            result_tracks[new_agent.id] = track_data

        self.filter_and_visualize_scenario(new_agents=result_tracks)

        return result_tracks

    def merge_graphs(self, cg_file: str) -> dict:
        """
        合并两个因果图
        
        Args:
            cg_file: 因果图文件路径
            
        Returns:
            dict: 合并后的因果图
        """
        try:
            with open(cg_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                cg = data['causal_graph']
                agents = data.get('agents', {})
        except Exception as e:
            print(f"读取第二个因果图文件时出错: {e}")
            return None
            
        # 合并因果图
        merged_cg = self.cg.copy()
        merged_agents = self.tps.copy()
                
        for parent_id, edges in cg.items():
            parent_id = int(parent_id) if parent_id.isdigit() else parent_id
            if parent_id not in merged_cg:
                merged_cg[parent_id] = []
                
            for edge in edges:
                child_id = edge['child_id']
                edge_attr = edge['edge_attributes']
                child_id = int(child_id) if child_id.isdigit() else child_id
                # 检查是否已存在相同的边
                edge_exists = False
                for existing_edge in merged_cg[parent_id]:
                    if existing_edge[0] == child_id:
                        # 合并边属性
                        existing_attrs = set(existing_edge[1])
                        new_attrs = set(edge_attr)
                        merged_attrs = list(existing_attrs.union(new_attrs))
                        merged_cg[parent_id].remove(existing_edge)
                        merged_cg[parent_id].append((child_id, merged_attrs))
                        edge_exists = True
                        break
                        
                if not edge_exists:
                    merged_cg[parent_id].append((child_id, edge_attr))
                    
        # 合并agents信息
        for agent_id, agent_info in agents.items():
            agent_id = int(agent_id) if agent_id.isdigit() else agent_id
            if agent_id not in merged_agents:
                merged_agents[agent_id] = Agent.from_dict(agent_info)
                
        self.cg = merged_cg
        self.tps = merged_agents

    def merge_scenes(self, cg_file: str, merged_fragment_id: str, merged_ego_id: int, frame_shift: int = 0) -> dict:
        """
        合并两个场景，将两个因果图中的所有交通参与者添加到一个场景内
        
        Args:
            cg_file: 第二个因果图文件路径
            frame_shift: 第二个场景的帧偏移量，用于调整时间对齐
            
        Returns:
            dict: 合并后的场景数据，包含所有交通参与者的轨迹
        """

        ego_track = self.track_change[self.fragment_id][self.ego_id]
        start_frame = min(ego_track['track_info']['frame_id'])
        end_frame = max(ego_track['track_info']['frame_id'])
        original_start_frame = min(self.tp_info[merged_fragment_id][merged_ego_id]['State']['frame_id'])
        original_end_frame = max(self.tp_info[merged_fragment_id][merged_ego_id]['State']['frame_id'])

        original_nodes = self.get_all_nodes()
        original_agents = self.tps.copy()

        self.merge_graphs(cg_file)
        
        all_nodes = self.get_all_nodes()
        
        # 准备轨迹数据字典
        merged_tracks = {}
        
        # 获取原始场景的轨迹数据
        for node_id in all_nodes:
            # 检查节点是否在原始场景中
            if node_id not in original_agents:
                ref_start_frame = min(self.tp_info[merged_fragment_id][node_id]['State']['frame_id'])
                ref_end_frame = max(self.tp_info[merged_fragment_id][node_id]['State']['frame_id'])
                aligned_frame_id = start_frame - frame_shift - original_start_frame + ref_start_frame

                agent = self.tps[node_id]
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
                    # 'anomalies': np.zeros(frame_count, dtype=bool),
                    'Type': agent.agent_type,
                    'Class': agent.agent_class,
                    'CrossType': agent.cross_type,
                    'Signal_Violation_Behavior': agent.signal_violation,
                    'retrograde_type': agent.retrograde_type,
                    'cardinal direction': agent.cardinal_direction,
                    'num': 6
                }
                track = []

                for frame_id in range(ref_start_frame, ref_end_frame+1):
                    reference_data = self.frame_data_processed[merged_fragment_id][frame_id][node_id]
                    track.append({
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

                length = self.tp_info[merged_fragment_id][node_id]['Length']
                width = self.tp_info[merged_fragment_id][node_id]['Width']
                for trk in track:
                    track_data['track_info']['frame_id'].append(trk['frame_id'])
                    track_data['track_info']['x'].append(trk['x'])
                    track_data['track_info']['y'].append(trk['y'])
                    track_data['track_info']['vx'].append(trk['vx'])
                    track_data['track_info']['vy'].append(trk['vy'])
                    track_data['track_info']['ax'].append(trk['ax'])
                    track_data['track_info']['ay'].append(trk['ay'])
                    track_data['track_info']['heading_rad'].append(trk['heading_rad'])
                    track_data['track_info']['width'].append(width)
                    track_data['track_info']['length'].append(length)

                for key in track_data['track_info']:
                    if key != 'frame_id':  # frame_id通常保持为列表
                        track_data['track_info'][key] = np.array(track_data['track_info'][key])
                    
                merged_tracks[node_id] = track_data
        
        # 可视化合并后的场景
        self.filter_and_visualize_scenario(new_agents=merged_tracks)
        
        # 更新并保存因果图可视化
        self.visualize_graph()
        
        return merged_tracks

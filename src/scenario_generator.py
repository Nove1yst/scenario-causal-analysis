import os
import numpy as np
from typing import Dict
from src.scenario_editor import ScenarioEditor
from utils.visualization_utils import create_gif_from_scenario

class ScenarioGenerator(ScenarioEditor):
    def __init__(self, data_dir: str, output_dir: str = None, conflict_type_config: str = None):
        super().__init__(data_dir, output_dir, conflict_type_config)
        self.tracks = {}

    def generate_track(self, agent, edges: list, frame_shift: int, reference_id: int = 1, offset = None):
        """
        根据agent信息和边属性生成轨迹数据
        
        Args:
            agent: 代理对象，如果为None则根据edges生成
            edges: 边属性列表，如果为None则随机生成
            
        Returns:
            dict: 生成的轨迹数据
        """
        if not self.frame_data_processed:
            raise ValueError("请先加载数据")
        
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
            'Type': agent.agent_type,
            'Class': agent.agent_class,
            'CrossType': agent.cross_type,
            'Signal_Violation_Behavior': agent.signal_violation,
            'retrograde_type': agent.retrograde_type,
            'cardinal direction': agent.cardinal_direction,
            'num': 6
        }
        
        # get reference track
        reference_track_id = 0
        cd = agent.cardinal_direction
        ct = agent.cross_type[0]

        for k in self.typical_tracks.keys():
            if k in ct.lower():
                reference_track_id = self.typical_tracks[k].get(cd, [])

        if reference_track_id:
            reference_track_id = reference_track_id[reference_id]

        reference_track = []
        
        aligned_frame_id = frame_shift
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
            
        for ref in reference_track:
            # # 计算新位置
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
                
            # 车辆尺寸（根据agent类型设置）
            if agent.agent_class == 'car':
                width, length = 1.8, 4.5
            elif agent.agent_class == 'bus' or agent.agent_class == 'truck':
                width, length = 2.5, 10.0
            else:
                width, length = 1.5, 3.0

            if current_frame_id < 0:
                continue
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
            if key != 'frame_id':
                track_data['track_info'][key] = np.array(track_data['track_info'][key])
        
        return track_data

    def generate_scenario(self, cg_file: str, ego_id: int, frame_shift, reference_id, offset = None, output_dir: str = None):
        """
        根据给定的因果图和典型轨迹库生成场景。
        """
        self.ego_id = ego_id
        self.fragment_id = None
        self.load_graph(cg_file, self.ego_id)
        
        for tp_id, tp in self.tps.items():
            shift = frame_shift[tp_id]
            ref_id = reference_id[tp_id]
            track_data = self.generate_track(tp, None, shift, ref_id, offset.get(tp_id, None))
            self.tracks[tp_id] = track_data

        self.visualize_scenario(tps=self.tracks, save_dir=output_dir)

        self.visualize_graph(save_dir=output_dir)
        
        return self.tracks
    
    def visualize_scenario(self, tps: Dict[int, dict], save_dir: str):
        """
        可视化因果图中的节点对应的代理轨迹

        Args:
            frame_id: 指定要可视化的帧ID，如果为None则使用第一帧
            new_agents: 新添加的代理轨迹字典，键为代理ID，值为轨迹数据
        """
        if not self.cg:
            raise ValueError("请先加载因果图数据")
            
        if not self.frame_data_processed:
            self.load_data()

        first_frame = 0
        last_frame = max(tps[self.ego_id]['track_info']['frame_id'])

        frame_data_to_plot = []
        for frame in range(first_frame, last_frame+1):
            filtered_frame_data = []
                    
            for tp_id, track_data in tps.items():
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
                            'tp_id': tp_id,
                            'vehicle_info': agent_frame_data
                        })
                            
            frame_data_to_plot.append(filtered_frame_data)
        
        tracks_dict = {}
        if tps:
            for tp_id, track_data in tps.items():
                if tp_id not in tracks_dict:
                    tracks_dict[tp_id] = track_data
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
            create_gif_from_scenario(
                tracks_dict,
                frame_data_to_plot,
                self.ego_id,
                self.fragment_id,
                save_dir,
                0
            )
        
            print(f"已保存修改后的场景可视化结果到: {save_dir}")

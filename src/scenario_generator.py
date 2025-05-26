import os
import numpy as np
from typing import Dict
from src.scenario_editor import ScenarioEditor
from utils.visualization_utils import create_gif_from_scenario

class ScenarioGenerator(ScenarioEditor):
    def __init__(self, data_dir: str, output_dir: str = None, conflict_type_config: str = None):
        super().__init__(data_dir, output_dir, conflict_type_config)
        self.tracks = {}

    def generate_scenario(self, cg_file: str, ego_id: int, frame_shift, reference_id, offset = None, start_frame = None, output_dir: str = None):
        """
        根据给定的因果图和典型轨迹库生成场景。
        """
        self.ego_id = ego_id
        self.fragment_id = None
        self.load_graph(cg_file, self.ego_id)
        
        for tp_id, tp in self.tps.items():
            shift = frame_shift[tp_id]
            ref_id = reference_id[tp_id]
            track_data = self.generate_track(tp, None, shift, ref_id, offset.get(tp_id, None), start_frame)
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

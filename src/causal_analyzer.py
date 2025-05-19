"""
    The definition of thresholds for safety metrics are based on the following website:
    https://criticality-metrics.readthedocs.io/en/latest/
"""
import os
import sys
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())
from ssm.src.two_dimensional_ssms import TAdv, TTC2D, ACT
from ssm.src.geometry_utils import CurrentD
from src.my_utils import check_conflict, is_uturn, in_intersection, is_following
from src.agent import Agent

LATJ_THRESHOLD = 0.5 * 9.81  # 横向加加速度风险阈值
LONJ_THRESHOLD = 1.2 * 9.81  # 纵向加加速度风险阈值
LAT_ACC_THRESHOLD = 2.0      # 横向加速度阈值
LON_ACC_THRESHOLD = 3.0      # 纵向加速度阈值

TTC_CRITICAL_THRESHOLD = 3.0   # TTC critical threshold
TADV_CRITICAL_THRESHOLD = 3.0   # TADV critical threshold
ACT_CRITICAL_THRESHOLD = 3.0    # ACT critical threshold
DISTANCE_CRITICAL_THRESHOLD = 3.0  # Distance critical threshold
DISTANCE_CRITICAL_THRESHOLD_ACT = 5.0

TTC_NORMAL_THRESHOLD = 10.0
TADV_NORMAL_THRESHOLD = 10.0
ACT_NORMAL_THRESHOLD = 10.0
DISTANCE_NORMAL_THRESHOLD = 10.0

WIN_LEN = 50
RISK_LEN = 8

critical_thresholds = {
    "tadv": TADV_CRITICAL_THRESHOLD,
    "ttc2d": TTC_CRITICAL_THRESHOLD,
    "act": ACT_CRITICAL_THRESHOLD,
    "distance": DISTANCE_CRITICAL_THRESHOLD
}

safety_metrics_list = ['tadv', 'act', 'distance']
# ACT和距离一起判断

head2tail_types = ['following', 'diverging', 'converging', 'crossing conflict: same cross type']
head2head_types = ['left turn and straight cross conflict: same side', 
                   'left turn and straight cross conflict: opposite side', 
                   'right turn and straight cross conflict: start side', 
                   'right turn and straight cross conflict: end side', 
                   'left turn and right turn conflict: start side', 
                   'left turn and right turn conflict: end side']
minor_conflict_types = ['parallel']

class CausalAnalyzer:
    def __init__(self, data_dir, output_dir=None, fragment_id=None):
        """
        Args:
            data_dir: 数据目录路径
            output_dir: 输出目录路径
        """
        self.data_dir = data_dir
        self.output_dir = output_dir if output_dir else os.path.join(data_dir, 'output')
        self.risk_events = None
        os.makedirs(self.output_dir, exist_ok=True)

        self.fragment_id = fragment_id if fragment_id is not None else None
        self.ego_id = None
        self.frame_data = None
        self.frame_data_processed = None
        self.tp_info = None
        self.cg = None
        self.agents = {}  # 添加agents字典
        
    def load_data(self):
        with open(os.path.join(self.data_dir, "track_change_tj.pkl"), "rb") as f:
            self.track_change = pickle.load(f)
        with open(os.path.join(self.data_dir, "tp_info_tj.pkl"), "rb") as f:
            self.tp_info = pickle.load(f)
        with open(os.path.join(self.data_dir, "frame_data_tj.pkl"), "rb") as f:
            self.frame_data = pickle.load(f)
        with open(os.path.join(self.data_dir, "frame_data_tj_processed.pkl"), "rb") as f:
            self.frame_data_processed = pickle.load(f)

    def set_fragment_id(self, fragment_id):
        self.fragment_id = fragment_id
        self.fragment_data = self.frame_data_processed[fragment_id]

    def prepare_ssm_dataframe(self, ego_id, anomaly_frames_child=None):
        """
        Prepare the dataframe for SSM calculation
        
        Args:
            fragment_id: Scenario ID
            ego_id: Ego vehicle ID

        Returns:
            The dataframe for SSM calculation
        """
        fragment_id = self.fragment_id
        if anomaly_frames_child is not None:
            start_frame = self.tp_info[fragment_id][ego_id]['State']['frame_id'].iloc[0]
            end_frame = self.tp_info[fragment_id][ego_id]['State']['frame_id'].iloc[-1]
            anomaly_frames_id = set()
            if anomaly_frames_child is not None:
                for anomaly_frame in anomaly_frames_child:
                    anomaly_frames_id.update(range(max((start_frame, anomaly_frame-10)), min(end_frame, anomaly_frame+1)))
                anomaly_frames_id = sorted(list(anomaly_frames_id))
                start_frame = np.max([start_frame, np.min(anomaly_frames_id) - WIN_LEN])
                end_frame = np.min([end_frame, np.max(anomaly_frames_id)+10])
            else: # Should not be reached
                anomaly_frames_id = range(start_frame, end_frame+1)
        else:
            track = self.track_change[fragment_id].get(ego_id, None)
            start_frame = self.tp_info[fragment_id][ego_id]['State']['frame_id'].iloc[0]
            end_frame = self.tp_info[fragment_id][ego_id]['State']['frame_id'].iloc[-1]
            anomaly_frames_id = track['track_info']['frame_id'][np.where(track['anomalies'] == True)[0]]
            start_frame = np.max([start_frame, np.min(anomaly_frames_id) - WIN_LEN])
            end_frame = np.min([end_frame, np.max(anomaly_frames_id)+10])

        return self.frame_data_processed[fragment_id][start_frame: end_frame+1], anomaly_frames_id, start_frame, end_frame
    
    def compute_ssm(self, ego_id, anomaly_frames_child=None):
        """
        Compute Surrogate Safety Metrics (SSM)
        
        Args:
            fragment_id: Scenario ID
            ego_id: Ego vehicle ID
            
        Returns:
            A dictionary containing the safety metrics
        """
        df, anomaly_frames, ego_start_frame, ego_end_frame = self.prepare_ssm_dataframe(ego_id, anomaly_frames_child)
        if df is None:
            return None
        num_timesteps = len(df)

        distance_values = {}
        tadv_values = {}
        ttc2d_values = {}
        act_values = {}

        start_frame = {}
        start_frame[ego_id] = ego_start_frame
        
        # Calculate the SSM metrics for ego vehicle
        for t in range(num_timesteps):
            for i, tp_id in enumerate(df[t].keys()):
                j = ego_id
                if tp_id == ego_id:
                    continue

                ssm_data = pd.DataFrame({
                    'x_i': [df[t][tp_id]['x']],
                    'y_i': [df[t][tp_id]['y']],
                    'vx_i': [df[t][tp_id]['vx']],
                    'vy_i': [df[t][tp_id]['vy']],
                    'hx_i': [np.cos(df[t][tp_id]['heading_rad'])],
                    'hy_i': [np.sin(df[t][tp_id]['heading_rad'])],
                    'length_i': [df[t][tp_id]['length']],
                    'width_i': [df[t][tp_id]['width']],
                    'type_i': [df[t][tp_id]['agent_type']],
                    'x_j': [df[t][j]['x']],
                    'y_j': [df[t][j]['y']],
                    'vx_j': [df[t][j]['vx']],
                    'vy_j': [df[t][j]['vy']],
                    'hx_j': [np.cos(df[t][j]['heading_rad'])],
                    'hy_j': [np.sin(df[t][j]['heading_rad'])],
                    'length_j': [df[t][j]['length']],
                    'width_j': [df[t][j]['width']],
                    'type_j': [df[t][j]['agent_type']]
                })

                # Calculate the distance between the two agents
                # distance = np.linalg.norm(ssm_data['x_i'] - ssm_data['x_j'])
                distance = CurrentD(ssm_data, toreturn='values')
                tadv_result = TAdv(ssm_data, toreturn='values')
                ttc2d_result = TTC2D(ssm_data, toreturn='values')
                act_result = ACT(ssm_data, toreturn='values')
                
                # Add results to the result dictionary
                if tp_id in tadv_values.keys():
                    tadv_values[tp_id].append(tadv_result[0])
                else:
                    tadv_values[tp_id] = [tadv_result[0]]
                    start_frame[tp_id] = ego_start_frame + t
                if tp_id in ttc2d_values.keys():
                    ttc2d_values[tp_id].append(ttc2d_result[0])
                else:
                    ttc2d_values[tp_id] = [ttc2d_result[0]]
                if tp_id in act_values.keys():
                    act_values[tp_id].append(act_result[0])
                else:
                    act_values[tp_id] = [act_result[0]]
                if tp_id in distance_values.keys():
                    distance_values[tp_id].append(distance[0])
                else:
                    distance_values[tp_id] = [distance[0]]
        
        safety_metrics = {
            "ego_id": ego_id,
            "anomaly_frames": anomaly_frames,
            "start_frame": start_frame,
            "end_frame": ego_end_frame,
            "tadv_values": tadv_values,
            "ttc2d_values": ttc2d_values,
            "act_values": act_values,
            "distance_values": distance_values
        }

        return safety_metrics
    
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

    def get_agent_info_dict(self, tp_id):
        track = self.tp_info[self.fragment_id].get(tp_id, None)
        if track is None:
            return {}
        agent_type = track['Type']
        agent_class = track['Class']
        cross_type = track['CrossType']
        signal_violation = track['Signal_Violation_Behavior']
        retrograde_type = track.get('retrograde_type', None)
        cardinal_direction = track.get('cardinal direction', None)
        
        return {
            "id": tp_id,
            "agent_type": agent_type,
            "agent_class": agent_class,
            "cross_type": cross_type,
            "signal_violation": signal_violation,
            "retrograde_type": retrograde_type,
            "cardinal_direction": cardinal_direction
        }

    def visualize_ssm(self, safety_metrics):
        """
        Visualize the safety metrics
        
        Args:
            safety_metrics: A dictionary containing the safety metrics
        """
        fig, axs = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle(f'Causal Analysis for Agent {safety_metrics["ego_id"]}', fontsize=20, fontweight='bold')
        start_frame = safety_metrics['start_frame'][safety_metrics['ego_id']]
        end_frame = safety_metrics['end_frame']
        anomaly_frames = safety_metrics['anomaly_frames']

        for i, tp_id in enumerate(safety_metrics['tadv_values'].keys()):
            frame_range = np.arange(start_frame, start_frame + len(safety_metrics['tadv_values'][tp_id]))
            if np.any(np.array(safety_metrics['tadv_values'][tp_id]) < TADV_NORMAL_THRESHOLD):
                axs[0, 0].plot(frame_range, safety_metrics['tadv_values'][tp_id], label=f'tp_id={tp_id}')
            if np.any(np.array(safety_metrics['ttc2d_values'][tp_id]) < TTC_NORMAL_THRESHOLD):
                axs[0, 1].plot(frame_range, safety_metrics['ttc2d_values'][tp_id], label=f'tp_id={tp_id}')
            if np.any(np.array(safety_metrics['act_values'][tp_id]) < ACT_NORMAL_THRESHOLD):
                axs[1, 0].plot(frame_range, safety_metrics['act_values'][tp_id], label=f'tp_id={tp_id}')
            if np.any(np.array(safety_metrics['distance_values'][tp_id]) < DISTANCE_NORMAL_THRESHOLD):
                axs[1, 1].plot(frame_range, safety_metrics['distance_values'][tp_id], label=f'tp_id={tp_id}')

        axs[0, 0].set_title(f'Time Advantage (TAdv)')
        axs[0, 0].set_xlabel('Frame')
        axs[0, 0].set_ylabel('TAdv (s)')
        axs[0, 0].set_ylim(0, 10)
        axs[0, 0].axhline(y=TADV_CRITICAL_THRESHOLD, color='r', linestyle='--', alpha=0.5)
        # axs[1, 1].scatter(anomaly_frames, safety_metrics['tadv_values'][tp_id][anomaly_frames], 
        #                 color='red', marker='x', s=100, label=f'Anomaly (tp_id={tp_id})')
        
        axs[0, 1].set_title(f'2D Time-to-Collision (TTC2D)')
        axs[0, 1].set_xlabel('Frame')
        axs[0, 1].set_ylabel('TTC2D (s)')
        axs[0, 1].set_ylim(0, 10)
        axs[0, 1].axhline(y=TTC_CRITICAL_THRESHOLD, color='r', linestyle='--', alpha=0.5)
        # axs[0, 1].scatter(anomaly_frames, safety_metrics['ttc2d_values'][tp_id][anomaly_frames], 
        #                 color='red', marker='x', s=100, label=f'Anomaly (tp_id={tp_id})')
        
        axs[1, 0].set_title(f'Aggregated Collision Time (ACT)')
        axs[1, 0].set_xlabel('Frame')
        axs[1, 0].set_ylabel('ACT (s)')
        axs[1, 0].set_ylim(0, 10)
        axs[1, 0].axhline(y=ACT_CRITICAL_THRESHOLD, color='r', linestyle='--', alpha=0.5)
        # axs[2, 1].scatter(anomaly_frames, safety_metrics['act_values'][tp_id][anomaly_frames], 
        #                 color='red', marker='x', s=100, label=f'Anomaly (tp_id={tp_id})')
        
        axs[1, 1].set_title(f'Distance')
        axs[1, 1].set_xlabel('Frame')
        axs[1, 1].set_ylabel('Distance (m)')
        axs[1, 1].set_ylim(0, 20)
        
        for ax in axs.flat:
            ax.set_xlim(start_frame, end_frame)
            ax.grid(True)
            ax.legend()
            for frame in anomaly_frames:
                ax.axvline(x=frame, color='r', linestyle='--', alpha=0.5)
    
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f"{self.fragment_id}_{safety_metrics['ego_id']}")
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f"ssm_{self.fragment_id}_{safety_metrics['ego_id']}.png"))
        plt.close()
        plt.show()
    
    def visualize_acceleration_analysis(self, ego_id):
        """
        创建加速度和加加速度的详细分析图
        
        Args:
            safety_metrics: 安全指标字典
            agent_idx: 代理索引
            save_path: 保存路径
        """
        fragment_id = self.fragment_id
        df, anomaly_frames, start_frame, end_frame = self.prepare_ssm_dataframe(ego_id)
        if df is None:
            return None
        num_timesteps = len(df)
        longitudinal_jerk = np.zeros(num_timesteps-1)
        lateral_jerk = np.zeros(num_timesteps-1)

        a_lon = np.zeros(num_timesteps)
        a_lat = np.zeros(num_timesteps)
        yaw = np.zeros(num_timesteps)
        heading = np.zeros(num_timesteps)
        yaw_rate = np.zeros(num_timesteps-1)
        heading_rate = np.zeros(num_timesteps-1)
        yaw_acc = np.zeros(num_timesteps-2)
        heading_acc = np.zeros(num_timesteps-2)

        for t in range(num_timesteps):
            a_lon[t] = df[t][ego_id]['a_lon']
            a_lat[t] = df[t][ego_id]['a_lat']
            yaw[t] = df[t][ego_id]['yaw_rad']
            heading[t] = df[t][ego_id]['heading_rad']

        time_step = 0.1
        for t in range(num_timesteps-1):
            longitudinal_jerk[t] = (df[t+1][ego_id]['a_lon'] - df[t][ego_id]['a_lon']) / time_step
            lateral_jerk[t] = (df[t+1][ego_id]['a_lat'] - df[t][ego_id]['a_lat']) / time_step
            yaw_rate[t] = (df[t+1][ego_id]['yaw_rad'] - df[t][ego_id]['yaw_rad']) / time_step
            heading_rate[t] = (df[t+1][ego_id]['heading_rad'] - df[t][ego_id]['heading_rad']) / time_step

        for t in range(num_timesteps-2):
            yaw_acc[t] = (yaw_rate[t+1] - yaw_rate[t]) / time_step
            heading_acc[t] = (heading_rate[t+1] - heading_rate[t]) / time_step
        
        fig, axs = plt.subplots(5, 2, figsize=(12, 12))
        
        # anomaly frames to plot
        if end_frame in anomaly_frames:
            anomaly_frames = anomaly_frames[:-1]
        if end_frame-1 in anomaly_frames:
            anomaly_frames = anomaly_frames[:-1]
        
        # 1. 纵向加速度图
        axs[0, 0].plot(range(start_frame, end_frame+1), a_lon, 'b-', label='Longitudinal')
        axs[0, 0].set_title(f'Longitudinal Acceleration')
        axs[0, 0].set_xlabel('Frame')
        axs[0, 0].set_ylabel('Acceleration (m/s²)')
        if len(anomaly_frames) > 0:
            axs[0, 0].scatter(anomaly_frames, 
                            a_lon[anomaly_frames - start_frame], 
                            color='red', marker='x', s=100, label='Anomaly Frames')
        
        # 2. 纵向加加速度图
        axs[0, 1].plot(range(start_frame, end_frame), longitudinal_jerk, 'b-', label='Longitudinal')
        axs[0, 1].set_title(f'Longitudinal Jerk')
        axs[0, 1].set_xlabel('Frame')
        axs[0, 1].set_ylabel('Jerk (m/s³)')

        # 标记异常帧
        if len(anomaly_frames) > 0:
            axs[0, 1].scatter(anomaly_frames, 
                            longitudinal_jerk[anomaly_frames - start_frame], 
                            color='red', marker='x', s=100, label='Anomaly Frames')
        
        # 3. 横向加速度图
        axs[1, 0].plot(range(start_frame, end_frame+1), a_lat, 'g-', label='Lateral')
        axs[1, 0].set_title(f'Lateral Acceleration')
        axs[1, 0].set_xlabel('Frame')
        axs[1, 0].set_ylabel('Acceleration (m/s²)')

        # 标记异常帧
        if len(anomaly_frames) > 0:
            axs[1, 0].scatter(anomaly_frames, 
                            a_lat[anomaly_frames - start_frame], 
                            color='red', marker='x', s=100, label='Anomaly Frames')

        # 4. 横向加加速度图
        axs[1, 1].plot(range(start_frame, end_frame), lateral_jerk, 'g-', label='Lateral')
        axs[1, 1].set_title(f'Lateral Jerk')
        axs[1, 1].set_xlabel('Frame')
        axs[1, 1].set_ylabel('Jerk (m/s³)')

        # 标记异常帧
        if len(anomaly_frames) > 0:
            axs[1, 1].scatter(anomaly_frames, 
                            lateral_jerk[anomaly_frames - start_frame], 
                            color='red', marker='x', s=100, label='Anomaly Frames')

        # 5. 偏转角图
        axs[2, 0].plot(range(start_frame, end_frame+1), yaw, 'b-', label='Yaw')
        axs[2, 0].set_title(f'Yaw')
        axs[2, 0].set_xlabel('Frame')
        axs[2, 0].set_ylabel('Yaw (rad)')

        if len(anomaly_frames) > 0:
            axs[2, 0].scatter(anomaly_frames, 
                            yaw[anomaly_frames - start_frame], 
                            color='red', marker='x', s=100, label='Anomaly Frames')
        
        # 6. 偏转角变化率图
        axs[2, 1].plot(range(start_frame, end_frame), yaw_rate, 'g-', label='Yaw Rate')
        axs[2, 1].set_title(f'Yaw Rate')
        axs[2, 1].set_xlabel('Frame')
        axs[2, 1].set_ylabel('Yaw Rate (rad/s)')

        if len(anomaly_frames) > 0:
            axs[2, 1].scatter(anomaly_frames, 
                            yaw_rate[anomaly_frames - start_frame], 
                            color='red', marker='x', s=100, label='Anomaly Frames')

        # 7. 偏转角变化率图
        axs[3, 0].plot(range(start_frame, end_frame+1), heading, 'b-', label='Heading')
        axs[3, 0].set_title(f'Heading')
        axs[3, 0].set_xlabel('Frame')
        axs[3, 0].set_ylabel('Heading (rad)')

        if len(anomaly_frames) > 0:
            axs[3, 0].scatter(anomaly_frames, 
                            heading[anomaly_frames - start_frame], 
                            color='red', marker='x', s=100, label='Anomaly Frames')
            
        # 8. 偏转角变化率图
        axs[3, 1].plot(range(start_frame, end_frame), heading_rate, 'g-', label='Heading Rate')
        axs[3, 1].set_title(f'Heading Rate')
        axs[3, 1].set_xlabel('Frame')
        axs[3, 1].set_ylabel('Heading Rate (rad/s)')

        if len(anomaly_frames) > 0:
            axs[3, 1].scatter(anomaly_frames, 
                            heading_rate[anomaly_frames - start_frame], 
                            color='red', marker='x', s=100, label='Anomaly Frames')

        axs[4, 0].plot(range(start_frame, end_frame-1), yaw_acc, 'g-', label='Yaw Accleration')
        axs[4, 0].set_title(f'Yaw Acceleration')
        axs[4, 0].set_xlabel('Frame')
        axs[4, 0].set_ylabel('Yaw Acceleration (rad/s²)')

        if len(anomaly_frames) > 0:
            axs[4, 0].scatter(anomaly_frames, 
                            heading_rate[anomaly_frames - start_frame], 
                            color='red', marker='x', s=100, label='Anomaly Frames')

        axs[4, 1].plot(range(start_frame, end_frame-1), heading_acc, 'g-', label='Heading Acceleration')
        axs[4, 1].set_title(f'Heading Acceleration')
        axs[4, 1].set_xlabel('Frame')
        axs[4, 1].set_ylabel('Heading Acceleration (rad/s²)')

        if len(anomaly_frames) > 0:
            axs[4, 1].scatter(anomaly_frames, 
                            heading_rate[anomaly_frames - start_frame], 
                            color='red', marker='x', s=100, label='Anomaly Frames')

        for ax in axs.flat:
            ax.set_xlim(start_frame, end_frame)
            ax.grid(True)
            ax.legend()

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f"{fragment_id}_{ego_id}")
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f"acc_{fragment_id}_{ego_id}.png"))
        plt.close()
        plt.show()

    def visualize_risk_events(self, ego_id, save_pic=True, save_pdf=False):
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
        dot = graphviz.Digraph(comment=f'Risk Events for Fragment {fragment_id}, Ego {ego_id}')
        dot.attr(rankdir='LR', size='12,8', dpi='300', fontname='Arial', 
                 bgcolor='white', concentrate='true')
        
        dot.attr('node', shape='circle', style='filled', color='black', 
                 fillcolor='skyblue', fontname='Arial', fontsize='10', fontcolor='black',
                 width='1.0', height='1.0', penwidth='1.0', fixedsize='true')
        
        dot.attr('edge', color='black', fontname='Arial', fontsize='12', fontcolor='darkred',
                 penwidth='1.0', arrowsize='0.5', arrowhead='none')  # No arrow since the behavioral graph is undir
        
        # 收集所有节点
        all_nodes = set()
        for agent, influenced_agents in self.risk_events.items():
            all_nodes.add(agent)
            for influenced_agent, _, _ in influenced_agents:
                all_nodes.add(influenced_agent)
        
        # 添加节点
        for node in all_nodes:
            # 获取节点信息
            node_info = self.get_agent_info(node)
            if node_info:
                agent_type, agent_class, cross_type, signal_violation, retrograde_type, cardinal_direction = node_info
                node_label = f"{node}\n{agent_class}\n"
                
                # 添加违规信息
                if cross_type:
                    for ct in cross_type:
                        if ct != "Normal":
                            node_label += f"{ct}\n"
                if signal_violation:
                    for sv in signal_violation:
                        if sv != "No violation of traffic lights":
                            node_label += f"{sv}\n"
                if retrograde_type and retrograde_type != "normal" and retrograde_type != "unknown":
                    node_label += f"\n{retrograde_type}"
            else:
                node_label = f"ID: {node}"
            
            if node == ego_id:
                dot.node(str(node), node_label, fillcolor='lightcoral', fontcolor='white')
            else:
                dot.node(str(node), node_label)
        
        # 创建边字典，用于合并重复边的标签
        edge_dict = {}
        
        # 收集所有边和标签
        for agent, influenced_agents in self.risk_events.items():
            for influenced_agent, ssm_type, critical_frames in influenced_agents:
                edge_key = (str(agent), str(influenced_agent))
                edge_label = f"{ssm_type}: {critical_frames[0]}-{critical_frames[-1]}"
                
                if edge_key in edge_dict:
                    edge_dict[edge_key].append(edge_label)
                else:
                    edge_dict[edge_key] = [edge_label]
        
        # 添加边，合并标签
        for (src, dst), labels in edge_dict.items():
            if len(labels) > 1:
                combined_label = "\n".join(labels)
            else:
                combined_label = labels[0]
            
            dot.edge(src, dst, label=combined_label, minlen='2')
        
        dot.attr(label=f'Graph (Fragment: {fragment_id}, Ego: {ego_id})')
        dot.attr(fontsize='20')
        dot.attr(labelloc='t')

        save_path = os.path.join(self.output_dir, f"{fragment_id}_{ego_id}")
        os.makedirs(save_path, exist_ok=True)     
        dot_path = os.path.join(save_path, f"risk_events_{fragment_id}_{ego_id}")
        if save_pic:
            dot.render(dot_path, format='png', cleanup=True)
            print(f"Risk events saved to {dot_path}.png")
        if save_pdf:
            dot.render(dot_path, format='pdf', cleanup=True)
            print(f"Risk events saved to {dot_path}.pdf")

    def print_risk_events(self, ego_id):
        fragment_id = self.fragment_id
        save_path = os.path.join(self.output_dir, f"{fragment_id}_{ego_id}")
        os.makedirs(save_path, exist_ok=True)
        complete_risk_graph_data_file = os.path.join(save_path, f"risk_events_data_{fragment_id}_{ego_id}.pkl")
        with open(complete_risk_graph_data_file, "wb") as f:
            pickle.dump(self.risk_events, f)
        
        risk_graph_file = os.path.join(save_path, f"cg_{fragment_id}_{ego_id}.txt")
        with open(risk_graph_file, "w") as f:
            f.write(f"Fragment ID: {fragment_id}\n")
            f.write(f"Ego ID: {ego_id}\n")
            
            # Add agent information
            ego_info = self.get_agent_info(ego_id)
            if ego_info:
                agent_type, agent_class, cross_type, signal_violation, retrograde_type, cardinal_direction = ego_info
                f.write(f"Ego Type: {agent_type}, Class: {agent_class}\n")
                f.write(f"Ego CrossType: {cross_type}, SignalViolation: {signal_violation}, RetrogradeType: {retrograde_type}\n")
            
            f.write("\nRisk Events:\n")
            for tp_id, edges in self.risk_events.items():
                f.write(f"\nTarget vehicle {tp_id}:\n")
                
                # 添加目标车辆信息
                tp_info = self.get_agent_info(tp_id)
                if tp_info:
                    agent_type, agent_class, cross_type, signal_violation, retrograde_type, cardinal_direction = tp_info
                    f.write(f"  Type: {agent_type}, Class: {agent_class}\n")
                    f.write(f"  CrossType: {cross_type}, SignalViolation: {signal_violation}, RetrogradeType: {retrograde_type}\n")
                
                for target_id, ssm, critical_frames in edges:
                    f.write(f"  - Influences: {target_id}\n")
                    
                    # 添加被影响车辆信息
                    if target_id != ego_id:
                        target_info = self.get_agent_info(target_id)
                        if target_info:
                            agent_type, agent_class, cross_type, signal_violation, retrograde_type, cardinal_direction = target_info
                            f.write(f"    Target Type: {agent_type}, Class: {agent_class}\n")
                            f.write(f"    Target CrossType: {cross_type}, SignalViolation: {signal_violation}, RetrogradeType: {retrograde_type}\n")
                    
                    f.write(f"  - Safety metric: {ssm}\n")
                    f.write(f"  - Critical frames: {critical_frames[:5]}")
                    if len(critical_frames) > 5:
                        f.write(f" ... (total {len(critical_frames)} frames)")
                    f.write("\n")
        
        print(f"Complete risk graph saved to {risk_graph_file}")
        print(f"Risk graph data saved to {complete_risk_graph_data_file}")

    def detect_risk(self, ego_id, anomaly_frames_child=None, visualize_acc=False, visualize_ssm=False, visualize_cg=True, visited_nodes=None, depth=0, max_depth=3):
        """
        创建描述代理之间因果关系的因果图，通过分析安全指标，并迭代地寻找父节点
        
        Args:
            fragment_id: 片段ID
            ego_id: 自车ID
            visualize_acc: 是否可视化加速度分析
            visualize_ssm: 是否可视化安全指标
            visualize_cg: 是否可视化因果图
            visited_nodes: 已访问节点集合，用于避免环
            depth: 当前递归深度
            max_depth: 最大递归深度
        """
        # 初始化已访问节点集合，避免环
        if visited_nodes is None:
            visited_nodes = set()
        
        # 如果当前节点已访问或达到最大深度，则返回空图
        if ego_id in visited_nodes or depth >= max_depth:
            return {}
        
        # 将当前节点标记为已访问
        visited_nodes.add(ego_id)
        ego_info = self.get_agent_info(ego_id)
        # ego = Agent(ego_id, self.fragment_id, ego_info)
        
        if visualize_acc and depth == 0:
            self.visualize_acceleration_analysis(ego_id)
        
        ssm_dataframe = self.compute_ssm(ego_id, anomaly_frames_child)
        anomaly_frames = ssm_dataframe["anomaly_frames"]
        if visualize_ssm and depth == 0:
            self.visualize_ssm(ssm_dataframe)

        # Initialize a dictionary to store critical conditions for each SSM
        critical_conditions = {
            "tadv": [],
            "ttc2d": [],
            "act": [],
            "distance": []
        }

        # 存储每个交互对象的第一个关键帧
        first_critical_frames = {}
        critical_frames_child = {}

        # Check each SSM for critical conditions
        for ssm in safety_metrics_list:
            for tp_id, ssm_values in ssm_dataframe[ssm+"_values"].items():
                # Determine if the SSM values are critical
                is_critical = [value < critical_thresholds[ssm] for value in ssm_values]
                
                # Get the indices of critical frames
                critical_frames_indices = [i for i, critical in enumerate(is_critical) if critical]
                
                if tp_id in ssm_dataframe["start_frame"]:
                    start_frame_tp = ssm_dataframe["start_frame"][tp_id]
                    # The actual critical frames
                    critical_frames = [start_frame_tp + i for i in critical_frames_indices]
                    
                    # Only keep critical frames that occur within 30 frames before and after anomaly frames
                    # AND where distance is less than DISTANCE_NORMAL_THRESHOLD meters
                    relevant_critical_frames = []
                    for cf_idx, cf in enumerate(critical_frames):
                        for af in anomaly_frames:
                            if abs(cf - af) <= WIN_LEN:  # 3s
                                # Check distance at this critical frame
                                distance_idx = critical_frames_indices[cf_idx]

                                # Only consider situations where both agents are in the intersection
                                x_i, y_i = self.fragment_data[cf][ego_id]['x'], self.fragment_data[cf][ego_id]['y']
                                x_j, y_j = self.fragment_data[cf][tp_id]['x'], self.fragment_data[cf][tp_id]['y']
                                if not (in_intersection(x_i, y_i) and in_intersection(x_j, y_j)):
                                    break

                                if distance_idx < len(ssm_dataframe["distance_values"][tp_id]) and ssm_dataframe["distance_values"][tp_id][distance_idx] < DISTANCE_NORMAL_THRESHOLD:
                                    # relevant_critical_frames.append(cf)
                                    x_i = self.fragment_data[cf][ego_id]['x']
                                    y_i = self.fragment_data[cf][ego_id]['y']
                                    vx_i = self.fragment_data[cf][ego_id]['vx']
                                    vy_i = self.fragment_data[cf][ego_id]['vy']
                                    ax_i = self.fragment_data[cf][ego_id]['ax']
                                    ay_i = self.fragment_data[cf][ego_id]['ay']
                                    ego_speed = np.hypot(vx_i, vy_i)
                                    ego_accel = np.hypot(ax_i, ay_i)

                                    x_j = self.fragment_data[cf][tp_id]['x']
                                    y_j = self.fragment_data[cf][tp_id]['y']
                                    vx_j = self.fragment_data[cf][tp_id]['vx']
                                    vy_j = self.fragment_data[cf][tp_id]['vy']
                                    ax_j = self.fragment_data[cf][tp_id]['ax']
                                    ay_j = self.fragment_data[cf][tp_id]['ay']
                                    tp_speed = np.hypot(vx_j, vy_j)
                                    tp_accel = np.hypot(ax_j, ay_j)

                                    agent_type, agent_class, _, signal_violation, retrograde_type, _ = self.get_agent_info(tp_id)

                                    # Only detect moving agents, unless the agent is a pedestrian.
                                    # Filter with regard to speed and acceleration.
                                    if (ego_speed > 1.0 or ego_accel > 1.0) and ((tp_speed > 1.0 or tp_accel > 1.0) or agent_type == 'ped' or agent_type == 'nmv'):
                                        # Filter ACT values with regard to distance.
                                        if ssm == 'act' and ssm_dataframe["distance_values"][tp_id][distance_idx] < DISTANCE_CRITICAL_THRESHOLD_ACT:
                                            relevant_critical_frames.append(cf)
                                        else:
                                            relevant_critical_frames.append(cf)
                                        break
                    
                    # 检查是否存在连续10帧
                    continuous_frames = []
                    temp_frames = []
                    for i in range(len(relevant_critical_frames)):
                        if i == 0 or relevant_critical_frames[i] == relevant_critical_frames[i-1] + 1:
                            temp_frames.append(relevant_critical_frames[i])
                        else:
                            if len(temp_frames) >= RISK_LEN:
                                continuous_frames.extend(temp_frames)
                            temp_frames = [relevant_critical_frames[i]]
                    
                    # 检查最后一组
                    if len(temp_frames) >= RISK_LEN:
                        continuous_frames.extend(temp_frames)
                        
                    if continuous_frames:
                        critical_conditions[ssm].append((tp_id, is_critical, continuous_frames))
                        critical_frames_child[tp_id] = continuous_frames
                        
                        # 记录每个交互对象的第一个关键帧
                        if tp_id not in first_critical_frames or min(continuous_frames) < first_critical_frames[tp_id]:
                            first_critical_frames[tp_id] = min(continuous_frames)

        # 创建带边属性的因果图
        r_evt = {}
        for ssm, conditions in critical_conditions.items():
            for tp_id, is_critical, critical_frames in conditions:
                if critical_frames:
                    if tp_id not in r_evt:
                        r_evt[tp_id] = []
                    r_evt[tp_id].append((ego_id, ssm, critical_frames))

        # 递归地为每个父节点构建因果树
        risk_events = r_evt.copy()
        for parent_id in r_evt.keys():
            # 使用父节点的第一个关键帧作为其异常帧
            if parent_id in first_critical_frames:
                # 递归调用analyze函数，将父节点作为新的ego_id
                parent_graph = self.detect_risk(
                    parent_id, 
                    anomaly_frames_child=[first_critical_frames[parent_id]],
                    visualize_acc=False, 
                    visualize_ssm=False, 
                    visualize_cg=False,
                    visited_nodes=visited_nodes.copy(),
                    depth=depth+1,
                    max_depth=max_depth
                )
                
                # 合并父节点的因果图到完整因果图
                if parent_graph:
                    for p_id, edges in parent_graph.items():
                        if p_id not in risk_events:
                            risk_events[p_id] = []
                        
                        for target_id, ssm, critical_frames in edges:
                            # 检查是否已存在边（无论方向）
                            edge_exists = False
                            existing_edge_idx = -1
                            parent_id = None
                            child_id = None
                            
                            # 检查正向边
                            for idx, (existing_target, existing_ssm, existing_frames) in enumerate(risk_events[p_id]):
                                if existing_target == target_id and existing_ssm == ssm:
                                    edge_exists = True
                                    existing_edge_idx = idx
                                    parent_id = p_id
                                    child_id = target_id
                                    break
                            
                            # 检查反向边
                            if not edge_exists and target_id in risk_events:
                                for idx, (existing_target, existing_ssm, existing_frames) in enumerate(risk_events[target_id]):
                                    if existing_target == p_id and existing_ssm == ssm:
                                        edge_exists = True
                                        existing_edge_idx = idx
                                        parent_id = target_id
                                        child_id = p_id
                                        break
                            
                            if edge_exists:
                                # 合并关键帧
                                if existing_edge_idx >= 0:
                                    existing_frames = risk_events[parent_id][existing_edge_idx][2]
                                    merged_frames = sorted(list(set(existing_frames + critical_frames)))
                                    risk_events[parent_id][existing_edge_idx] = (child_id, ssm, merged_frames)
                            else:
                                # 添加新边
                                risk_events[p_id].append((target_id, ssm, critical_frames))
        
        # 只在顶层调用时可视化和保存完整因果图
        if depth == 0:
            self.risk_events = risk_events
        
        if depth == 0 and visualize_cg:
            self.visualize_risk_events(ego_id)
            self.print_risk_events(ego_id)
        
        return risk_events
    
    def extract_cg(self):
        self.cg = {}
        self.agents = {}  # 重置agents字典
        edges = set()

        for p_id, influence in self.risk_events.items():
            for t_id, ssm, critical_frames in influence:
                p_type, p_class, p_ct, p_sv, p_rt, p_cd = self.get_agent_info(p_id)
                t_type, t_class, t_ct, t_sv, t_rt, t_cd = self.get_agent_info(t_id)

                # Assume no causal relationship between pedestrians
                if p_type == 'ped' and t_type == 'ped':
                    continue

                # 创建Agent对象
                if p_id not in self.agents:
                    p_agent = Agent.from_tuple(p_id, self.get_agent_info(p_id))
                    self.agents[p_id] = p_agent

                if t_id not in self.agents:
                    t_agent = Agent.from_tuple(t_id, self.get_agent_info(t_id))
                    self.agents[t_id] = t_agent
                
                edge_attr = []
                parent_id, child_id = p_id, t_id
                edges.add((parent_id, child_id))
                rt_flag, sv_flag = False, False
                conflict = 'unknown'

                if p_ct and t_ct and p_ct != 'Others' and t_ct != 'Others':
                    conflict = check_conflict(p_cd, t_cd, p_ct, t_ct)
                    if conflict not in minor_conflict_types:
                        edge_attr.append(conflict)
                if p_ct:
                    if p_ct == 'Others' and is_uturn(p_cd):
                        p_ct = 'U-Turn'   
                        parent_id, child_id = p_id, t_id                 
                        edge_attr.append('U-turn conflict')
                    elif p_ct == 'Others':
                        parent_id, child_id = p_id, t_id                        
                        edge_attr.append('unusual cross type')
                if t_ct:
                    if t_ct == 'Others' and is_uturn(t_cd):
                        t_ct = 'U-Turn'
                        parent_id, child_id = t_id, p_id      
                        edge_attr.append('U-turn conflict')
                    elif t_ct == 'Others':
                        parent_id, child_id = t_id, p_id
                        edge_attr.append('unusual cross type')

                if t_sv:
                    t_sv = t_sv[0]
                if p_sv:
                    p_sv = p_sv[0]
                if t_sv and t_sv != "No violation of traffic lights":
                    edge_attr.append(f"{t_id}: {t_sv}")
                    sv_flag = True
                    if t_sv == 'red-light running':
                        parent_id, child_id = t_id, p_id
                    elif t_sv == 'yellow-light running' and p_sv != 'red-light running':
                        parent_id, child_id = t_id, p_id
                if p_sv and p_sv != "No violation of traffic lights":
                    edge_attr.append(f"{p_id}: {p_sv}")
                    sv_flag = True
                    if p_sv == 'red-light running':
                        parent_id, child_id = p_id, t_id
                    elif p_sv == 'yellow-light running' and t_sv != 'red-light running':
                        parent_id, child_id = p_id, t_id

                if p_rt is not None and t_rt is not None:
                    if t_rt != "normal" and t_rt != 'unknown':
                        parent_id, child_id = t_id, p_id
                        edge_attr.append(f"{t_id}: {t_rt}")
                        rt_flag = True
                    if p_rt != "normal" and p_rt != 'unknown':
                        parent_id, child_id = p_id, t_id
                        edge_attr.append(f"{p_id}: {p_rt}")
                        rt_flag = True

                # The retrograding agent must be the cause for the conflict.
                if not rt_flag:
                    # If there is no traffic violations, the pedstrian should be the cause.
                    if p_type == 'ped':
                        if not sv_flag:
                            parent_id, child_id = p_id, t_id
                            edge_attr.append(f"{p_id}: intrusion")
                    elif t_type == 'ped':
                        if not sv_flag:
                            parent_id, child_id = t_id, p_id
                            edge_attr.append(f"{t_id}: intrusion")
                    else:
                        # Determine following order and the cause and the effect accordingly
                        # if there is no serious traffic violation
                        cf = critical_frames[0]
                        x_i, y_i = self.fragment_data[cf][p_id]['x'], self.fragment_data[cf][p_id]['y']
                        vx_i, vy_i = self.fragment_data[cf][p_id]['vx'], self.fragment_data[cf][p_id]['vy']
                        h_i = self.fragment_data[cf][p_id]['heading_rad']

                        x_j, y_j = self.fragment_data[cf][t_id]['x'], self.fragment_data[cf][t_id]['y']
                        vx_j, vy_j = self.fragment_data[cf][t_id]['vx'], self.fragment_data[cf][t_id]['vy']
                        h_j = self.fragment_data[cf][t_id]['heading_rad']
                        # If p_id follows t_id and one is following the other ...
                        if conflict in head2tail_types:
                            if is_following(x_i, y_i, x_j, y_j, vx_i, vy_i, vx_j, vy_j, h_i, h_j):
                                # then the parent node (the cause) should be the agent in the front (the leader)
                                parent_id, child_id = t_id, p_id
                            else:
                                parent_id, child_id = p_id, t_id
                            # else if t_id and t_id are head2head ...
                            # elif conflict in head2head_types and is_head_on(x_i, y_i, x_j, y_j, vx_i, vy_i, vx_j, vy_j, h_i, h_j):

                # pruning: remove unclassified fake correlations
                if len(edge_attr) > 0:
                    has_edge = False
                    for p, e in self.cg.items():
                        for c, _ in e:
                            if (parent_id == p and child_id == c) or (parent_id ==c and child_id == p):
                                has_edge = True
                    
                    if not has_edge:
                        if parent_id not in self.cg.keys():
                            self.cg[parent_id] = []
                            self.cg[parent_id].append((child_id, edge_attr)) 
                        else:
                            self.cg[parent_id].append((child_id, edge_attr))

    def simplify_cg(self, ego_id):
        """
        简化因果图：
        1. 只保留包含ego_id的连通子图
        2. 从自车节点开始，深度<=2的节点（不论边方向）
        
        Args:
            ego_id: 自车ID
        """
        def find_connected_component(start_node, graph):
            """
            使用DFS找到包含start_node的连通子图
            
            Args:
                start_node: 起始节点
                graph: 原始图
                
            Returns:
                包含start_node的连通子图
            """
            visited = set()
            component = {}
            
            def dfs(node):
                if node in visited:
                    return
                visited.add(node)
                
                # 处理从当前节点出发的边
                if node in graph:
                    component[node] = graph[node]
                    for neighbor, _ in graph[node]:
                        dfs(neighbor)
                
                # 处理指向当前节点的边
                for parent, edges in graph.items():
                    for neighbor, _ in edges:
                        if neighbor == node and parent not in visited:
                            dfs(parent)
            
            dfs(start_node)
            return component

        def limit_depth(start_node, graph, max_depth=2):
            """
            从start_node开始，只保留深度<=max_depth的节点
            
            Args:
                start_node: 起始节点
                graph: 原始图
                max_depth: 最大深度
                
            Returns:
                深度限制后的子图
            """
            result = {}
            visited = set()
            depth_map = {start_node: 0}
            
            def dfs_up(node, current_depth):
                """向上追溯父节点"""
                if node in visited or current_depth > max_depth:
                    return
                visited.add(node)
                
                # 处理指向当前节点的边
                for parent, edges in graph.items():
                    for neighbor, _ in edges:
                        if neighbor == node and parent not in visited:
                            depth_map[parent] = current_depth + 1
                            dfs_up(parent, current_depth + 1)
            
            def dfs_down(node, current_depth):
                """向下追溯子节点"""
                if node in visited or current_depth > max_depth:
                    return
                visited.add(node)
                
                # 处理从当前节点出发的边
                if node in graph:
                    for neighbor, _ in graph[node]:
                        if neighbor not in visited:
                            depth_map[neighbor] = current_depth + 1
                            dfs_down(neighbor, current_depth + 1)
            
            # 从start_node开始向上追溯
            dfs_up(start_node, 0)
            visited.clear()  # 清除visited，准备向下追溯
            
            # 从start_node开始向下追溯
            dfs_down(start_node, 0)
            
            # 只保留深度在max_depth以内的节点及其边
            for node, depth in depth_map.items():
                if depth <= max_depth and node in graph:
                    result[node] = []
                    for neighbor, edge_attr in graph[node]:
                        if neighbor in depth_map and depth_map[neighbor] <= max_depth:
                            result[node].append((neighbor, edge_attr))
            
            return result
            
        # 找到包含ego_id的连通子图
        connected_component = find_connected_component(ego_id, self.cg)
        
        # 如果连通子图为空，说明ego_id不在任何连通子图中
        if not connected_component:
            print(f"Warning: ego_id {ego_id} is not in any connected component of the causal graph.")
            self.cg = {ego_id: []}
            return
            
        # 限制深度，只保留2层以内的节点
        self.cg = limit_depth(ego_id, connected_component)

    def extract_and_simplify(self, ego_id, visualize=False, save=True):
        self.extract_cg()
        self.simplify_cg(ego_id)
        if visualize:
            self.visualize_cg(ego_id)
        if save:
            self.save_cg(ego_id)

    def visualize_cg(self, ego_id, save_pic=True, save_pdf=False):
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
        dot = graphviz.Digraph(comment=f'Causal Graph for Fragment {fragment_id}, Ego {ego_id}')
        dot.attr(rankdir='LR', size='12,8', dpi='300', fontname='Arial', 
                 bgcolor='white', concentrate='true')
        
        dot.attr('node', shape='circle', style='filled', color='black', 
                 fillcolor='skyblue', fontname='Arial', fontsize='10', fontcolor='black',
                 width='1.0', height='1.0', penwidth='1.0', fixedsize='true')
        
        dot.attr('edge', color='black', fontname='Arial', fontsize='12', fontcolor='darkred',
                 penwidth='1.0', arrowsize='0.5', arrowhead='normal')
        
        # 收集所有节点
        all_nodes = set()
        for agent, influenced_agents in self.cg.items():
            all_nodes.add(agent)
            for influenced_agent, _ in influenced_agents:
                all_nodes.add(influenced_agent)
        
        # 添加节点
        for node in all_nodes:
            # 获取节点信息
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
                node_label = f"ID: {node}"
            
            if node == ego_id:
                dot.node(str(node), node_label, fillcolor='lightcoral', fontcolor='white')
            else:
                dot.node(str(node), node_label)
        
        # 创建边字典，用于合并重复边的标签
        edge_dict = {}
        
        # 收集所有边和标签
        for agent, influenced_agents in self.cg.items():
            for influenced_agent, edge_attr in influenced_agents:
                edge_key = (str(agent), str(influenced_agent))
                
                if edge_key not in edge_dict:
                    edge_dict[edge_key] = edge_attr
        
        # 添加边，合并标签
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
        dot_path = os.path.join(save_path, f"cg_{fragment_id}_{ego_id}")
        if save_pic:
            dot.render(dot_path, format='png', cleanup=True)
            print(f"Causal graph saved to {dot_path}.png")
        if save_pdf:
            dot.render(dot_path, format='pdf', cleanup=True)
            print(f"Causal graph saved to {dot_path}.pdf")

    def load_risk_events(self, fragment_id, ego_id):
        """
        读取保存的风险数据。

        Args:
            fragment_id: 片段ID
            ego_id: 自车ID

        Returns:
            加载的因果图数据字典
        """
        save_path = os.path.join(self.output_dir, f"{fragment_id}_{ego_id}")
        risk_events_file = os.path.join(save_path, f"risk_events_data_{fragment_id}_{ego_id}.pkl")
        
        if not os.path.exists(risk_events_file):
            print(f"Risk events file not found: {risk_events_file}")
            return None
        
        try:
            with open(risk_events_file, "rb") as f:
                risk_events = pickle.load(f)
            print(f"Successfully loaded risk events: {risk_events_file}")
            self.risk_events = risk_events
            self.fragment_id = fragment_id
        except Exception as e:
            print(f"Error loading risk events: {e}")
        
    def save_cg(self, ego_id: str) -> str:
        """
        将因果图保存为JSON文件
        
        Args:
            ego_id: 自车ID
            
        Returns:
            str: 保存的文件路径
        """
        if not self.cg:
            print("The causal graph is empty.")
            return None
            
        save_path = os.path.join(self.output_dir, f"{self.fragment_id}_{ego_id}")
        os.makedirs(save_path, exist_ok=True)
        
        # 将因果图转换为可序列化的格式
        serializable_cg = {}
        for parent_id, edges in self.cg.items():
            serializable_cg[str(parent_id)] = []
            for child_id, edge_attr in edges:
                serializable_cg[str(parent_id)].append({
                    'child_id': str(child_id),
                    'edge_attributes': edge_attr
                })
        
        # 将agents字典转换为可序列化的格式
        serializable_agents = {}
        for agent_id, agent in self.agents.items():
            serializable_agents[str(agent_id)] = agent.to_dict()
        
        # 保存为JSON文件
        json_file = os.path.join(save_path, f"cg_{self.fragment_id}_{ego_id}.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({
                'causal_graph': serializable_cg,
                'agents': serializable_agents
            }, f, ensure_ascii=False, indent=4)
            
        print(f"因果图已保存到: {json_file}")
        return json_file
        
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
            self.agents = {}
            for agent_id, agent_dict in serializable_agents.items():
                agent_id = int(agent_id) if agent_id.isdigit() else agent_id
                self.agents[agent_id] = Agent.from_dict(agent_dict)
                    
            self.fragment_id = fragment_id
            self.ego_id = ego_id
            print(f"成功加载因果图: {json_file}")
            return True
            
        except Exception as e:
            print(f"加载因果图时出错: {e}")
            return False
        
    
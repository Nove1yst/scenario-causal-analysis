"""
    The definition of thresholds for safety metrics are based on the following website:
    https://criticality-metrics.readthedocs.io/en/latest/
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())
from ssm.src.two_dimensional_ssms import TAdv, TTC2D, ACT
from ssm.src.geometry_utils import CurrentD
from src.utils import check_conflict
from src.agent import Agent

LATJ_THRESHOLD = 0.5 * 9.81  # 横向加加速度风险阈值
LONJ_THRESHOLD = 1.2 * 9.81  # 纵向加加速度风险阈值
LAT_ACC_THRESHOLD = 2.0      # 横向加速度阈值
LON_ACC_THRESHOLD = 3.0      # 纵向加速度阈值

TTC_CRITICAL_THRESHOLD = 5.0   # TTC critical threshold
TADV_CRITICAL_THRESHOLD = 5.0   # TADV critical threshold
ACT_CRITICAL_THRESHOLD = 3.0    # ACT critical threshold
# DISTANCE_CRITICAL_THRESHOLD = 20  # Distance critical threshold

TTC_NORMAL_THRESHOLD = 10.0
TADV_NORMAL_THRESHOLD = 10.0
ACT_NORMAL_THRESHOLD = 10.0
DISTANCE_NORMAL_THRESHOLD = 20.0

critical_thresholds = {
    "tadv": TADV_CRITICAL_THRESHOLD,
    "ttc2d": TTC_CRITICAL_THRESHOLD,
    "act": ACT_CRITICAL_THRESHOLD,
    # "distance": DISTANCE_CRITICAL_THRESHOLD
}

safety_metrics_list = ['tadv', 'ttc2d', 'act']

class CausalAnalyzer:
    def __init__(self, data_dir, output_dir=None, fragment_id=None):
        """
        Args:
            data_dir: 数据目录路径
            output_dir: 输出目录路径
        """
        self.data_dir = data_dir
        self.output_dir = output_dir if output_dir else os.path.join(data_dir, 'analysis_results')
        self.risk_events = None
        os.makedirs(self.output_dir, exist_ok=True)

        self.fragment_id = fragment_id if fragment_id is not None else None
        
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
                start_frame = np.max([start_frame, np.min(anomaly_frames_id) - 30])
                end_frame = np.min([end_frame, np.max(anomaly_frames_id)+1])
            else: # Should not come here
                anomaly_frames_id = range(start_frame, end_frame+1)
        else:
            track = self.track_change[fragment_id].get(ego_id, None)
            start_frame = self.tp_info[fragment_id][ego_id]['State']['frame_id'].iloc[0]
            end_frame = self.tp_info[fragment_id][ego_id]['State']['frame_id'].iloc[-1]
            anomaly_frames_id = track['track_info']['frame_id'][np.where(track['anomalies'] == True)[0]]
            start_frame = np.max([start_frame, np.min(anomaly_frames_id) - 30])
            end_frame = np.min([end_frame, np.max(anomaly_frames_id)+1])

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
        
        # TODO: detect the type of anomaly
        # for anomaly_frame in anomalies_frames:
        #     df_anomaly = df[anomaly_frame]

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
            "fragment_id": self.fragment_id,
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
        save_path = os.path.join(self.output_dir, f"{safety_metrics['fragment_id']}_{safety_metrics['ego_id']}")
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f"ssm_{safety_metrics['fragment_id']}_{safety_metrics['ego_id']}.png"))
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
        # 标记异常帧
        if len(anomaly_frames) > 0:
            axs[0, 0].scatter(anomaly_frames, 
                            a_lon[anomaly_frames - start_frame], 
                            color='red', marker='x', s=100, label='Anomaly Frames')
        
        # # 标记急刹车和急加速事件
        # braking_events = range(start_frame, end_frame+1)[hard_braking_events]
        # if len(braking_events) > 0:
        #     axs[0, 0].scatter(braking_events, 
        #                     longitudinal_acc[hard_braking_events], 
        #                     color='red', label='Hard braking')
        
        # acceleration_events = time_steps[hard_acceleration_events]
        # if len(acceleration_events) > 0:
        #     axs[0, 0].scatter(acceleration_events, 
        #                     longitudinal_acc[hard_acceleration_events], 
        #                     color='orange', label='Hard acceleration')
        
        # # 添加舒适阈值线
        # axs[0, 0].axhline(y=LON_ACC_THRESHOLD, color='r', linestyle='--', alpha=0.5)
        # axs[0, 0].axhline(y=-LON_ACC_THRESHOLD, color='r', linestyle='--', alpha=0.5)
        # axs[0, 0].text(0, LON_ACC_THRESHOLD+0.5, f'Comfort threshold (±{LON_ACC_THRESHOLD} m/s²)', color='r', alpha=0.7)
        # axs[0, 0].legend()
        
        # # 标记急转弯事件
        # turning_events = time_steps[hard_turning_events]
        # if len(turning_events) > 0:
        #     axs[0, 1].scatter(turning_events, 
        #                     lateral_acc[hard_turning_events], 
        #                     color='red', label='Hard turning')
        
        # # 添加舒适阈值线
        # axs[0, 1].axhline(y=LAT_ACC_THRESHOLD, color='r', linestyle='--', alpha=0.5)
        # axs[0, 1].axhline(y=-LAT_ACC_THRESHOLD, color='r', linestyle='--', alpha=0.5)
        # axs[0, 1].text(0, LAT_ACC_THRESHOLD+0.5, f'Comfort threshold (±{LAT_ACC_THRESHOLD} m/s²)', color='r', alpha=0.7)
        
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
        # # 添加舒适阈值线
        # axs[1, 0].axhline(y=LONJ_THRESHOLD, color='r', linestyle='--', alpha=0.5)
        # axs[1, 0].axhline(y=-LONJ_THRESHOLD, color='r', linestyle='--', alpha=0.5)
        # axs[1, 0].text(0, LONJ_THRESHOLD+0.5, f'Comfort threshold (±{LONJ_THRESHOLD} m/s³)', color='r', alpha=0.7)
        # axs[1, 0].legend()
        
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
                
                # 添加违规信息（如果有）
                if cross_type:
                    for ct in cross_type:
                        if ct != "Normal":
                            node_label += f"{ct}\n"
                if signal_violation:
                    # node_label += "\nSignal: "
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
        complete_causal_graph_data_file = os.path.join(save_path, f"risk_events_data_{fragment_id}_{ego_id}.pkl")
        with open(complete_causal_graph_data_file, "wb") as f:
            pickle.dump(self.risk_events, f)
        
        causal_graph_file = os.path.join(save_path, f"cg_{fragment_id}_{ego_id}.txt")
        with open(causal_graph_file, "w") as f:
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
        
        print(f"Complete graph saved to {causal_graph_file}")
        print(f"Graph data saved to {complete_causal_graph_data_file}")

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
            # "distance": []
        }

        # 存储每个交互对象的第一个关键帧
        first_critical_frames = {}
        critical_frames_child = {}

        # Check each SSM for critical conditions
        for ssm in safety_metrics_list:
            for tp_id, ssm_values in ssm_dataframe[ssm+"_values"].items():
                # Determine if the SSM values are critical
                if ssm in ["ttc", "psd", "tadv", "ttc2d", "act"]:  # Lower is critical
                    is_critical = [value < critical_thresholds[ssm] for value in ssm_values]
                else:  # Higher is critical for "drac"
                    is_critical = [value > critical_thresholds[ssm] for value in ssm_values]
                
                # Get the indices of critical frames
                critical_frames_indices = [i for i, critical in enumerate(is_critical) if critical]
                
                if tp_id in ssm_dataframe["start_frame"]:
                    start_frame_tp = ssm_dataframe["start_frame"][tp_id]
                    # The actual critical frames
                    critical_frames = [start_frame_tp + i for i in critical_frames_indices]
                    
                    # Only keep critical frames that occur within 5 frames before and after anomaly frames
                    # AND where distance is less than 15 meters
                    relevant_critical_frames = []
                    for cf_idx, cf in enumerate(critical_frames):
                        for af in anomaly_frames:
                            if abs(cf - af) <= 20:  # 2s
                                # Check distance at this critical frame
                                distance_idx = critical_frames_indices[cf_idx]
                                if distance_idx < len(ssm_dataframe["distance_values"][tp_id]) and ssm_dataframe["distance_values"][tp_id][distance_idx] < 30.0:
                                    # relevant_critical_frames.append(cf)
                                    vx_i = self.fragment_data[cf][ego_id]['vx']
                                    vy_i = self.fragment_data[cf][ego_id]['vy']
                                    ego_speed = np.hypot(vx_i, vy_i)

                                    vx_j = self.fragment_data[cf][tp_id]['vx']
                                    vy_j = self.fragment_data[cf][tp_id]['vy']
                                    tp_speed = np.hypot(vx_j, vy_j)

                                    agent_type, agent_class, cross_type, signal_violation, retrograde_type, cardinal_direction = self.get_agent_info(tp_id)

                                    # Only detect moving agents, unless the agent is a pedestrian.
                                    # Filter with regard to speed.
                                    if ego_speed > 1.0 and (tp_speed > 1.0 or agent_type == 'ped'):
                                        relevant_critical_frames.append(cf)
                                        break
                    
                    if relevant_critical_frames and len(relevant_critical_frames) > 10:
                        critical_conditions[ssm].append((tp_id, is_critical, relevant_critical_frames))
                        critical_frames_child[tp_id] = relevant_critical_frames
                        
                        # 记录每个交互对象的第一个关键帧
                        if tp_id not in first_critical_frames or min(relevant_critical_frames) < first_critical_frames[tp_id]:
                            first_critical_frames[tp_id] = min(relevant_critical_frames)

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
        edges = set()

        for p_id, influence in self.risk_events.items():
            for t_id, ssm, critical_frames in influence:
                p_type, p_class, p_ct, p_sv, p_rt, p_cd = self.get_agent_info(p_id)
                t_type, t_class, t_ct, t_sv, t_rt, t_cd = self.get_agent_info(t_id)

                edge_attr = []
                parent_id, child_id = p_id, t_id
                edges.add((parent_id, child_id))

                if p_class == 'ped':
                    parent_id, child_id = p_id, t_id
                    edge_attr.append(f"{parent_id}: invasive behavior")

                if t_sv:
                    for sv in t_sv:
                        if sv != "No violation of traffic lights":
                            parent_id, child_id = t_id, p_id
                            edge_attr.append(f"{parent_id}: {sv}")
                if p_sv:
                    for sv in p_sv: 
                        if sv != "No violation of traffic lights":
                            parent_id, child_id = p_id, t_id
                            edge_attr.append(f"{parent_id}: {sv}")

                if p_rt is not None and t_rt is not None:
                    if t_rt != "normal":
                        parent_id, child_id = t_id, p_id
                        edge_attr.append(f"{parent_id}: {t_rt}")
                    if p_rt != "normal":
                        parent_id, child_id = p_id, t_id
                        edge_attr.append(f"{parent_id}: {p_rt}")

                # TODO: determine unusual behavior
                if p_ct is None or t_ct is None:
                    edge_attr.append('unusual behavior')

                # If the conflict cannot be described for now, then check for potential crossing lanes.
                if len(edge_attr) == 0:
                    edge_attr.append(check_conflict(p_cd, t_cd, p_ct, t_ct))

                if parent_id not in self.cg.keys():
                    self.cg[parent_id] = []
                    self.cg[parent_id].append((child_id, edge_attr)) 
                else:
                    has_edge = False
                    for c, _ in self.cg[parent_id]:
                        if child_id == c:
                            has_edge = True
                            break
                    if not has_edge:
                        self.cg[parent_id].append((child_id, edge_attr))

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
        # style='curved'
        
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
        读取保存的因果图数据。

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
                causal_graph = pickle.load(f)
            print(f"Successfully loaded risk events: {risk_events_file}")
            self.risk_events = causal_graph
            self.fragment_id = fragment_id
        except Exception as e:
            print(f"Error loading risk events: {e}")
        
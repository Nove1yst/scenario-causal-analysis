"""
    The definition of thresholds for safety metrics are based on the following website:
    https://criticality-metrics.readthedocs.io/en/latest/
"""
import os
import sys
import json
import glob
import math
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch

sys.path.append(os.getcwd())
from ssm.src.longitudinal_ssms import TTC, DRAC, PSD
from ssm.src.two_dimensional_ssms import TAdv, TTC2D, ACT
from ssm.src.geometry_utils import CurrentD
# from utils.collision_utils import calculate_vehicle_bbox
from utils.visualization_utils import draw_vehicle, calculate_four_points, plot_vehicle_positions

FRAME_RATE = 20
TIME_STEP = 1.0 / FRAME_RATE

LATJ_THRESHOLD = 0.5 * 9.81  # 横向加加速度风险阈值
LONJ_THRESHOLD = 1.2 * 9.81  # 纵向加加速度风险阈值
LAT_ACC_THRESHOLD = 2.0      # 横向加速度阈值
LON_ACC_THRESHOLD = 3.0      # 纵向加速度阈值
TTC_CRITICAL_THRESHOLD = 1.5   # TTC critical threshold
DRAC_CRITICAL_THRESHOLD = 4.0   # DRAC critical threshold, for considerable reaction time
PSD_CRITICAL_THRESHOLD = 1.5    # PSD critical threshold, for scenario classification
TADV_CRITICAL_THRESHOLD = 3.0   # TADV critical threshold, for scenario classification
ACT_CRITICAL_THRESHOLD = 0.5    # ACT critical threshold, for scenario classification
DISTANCE_CRITICAL_THRESHOLD = 20  # Distance critical threshold

TTC_NORMAL_THRESHOLD = 10.0
DRAC_NORMAL_THRESHOLD = 10.0
PSD_NORMAL_THRESHOLD = 5.0
TADV_NORMAL_THRESHOLD = 10.0
ACT_NORMAL_THRESHOLD = 10.0
DISTANCE_NORMAL_THRESHOLD = 20.0

fragment_id_list = ['7_28_1 R21', '8_10_1 R18', '8_10_2 R19', '8_11_1 R20']
ego_id_dict = {
    '7_28_1 R21': [1, 9, 11, 13, 26, 31, 79, 141, 144, 148, 162, 167, 170, 181],
    '8_10_1 R18': [13, 70, 76, 157],
    '8_10_2 R19': [75, 112, 126, 178],
    '8_11_1 R20': [4, 9, 37, 57, 60, 80, 84, 87, 93, 109, 159, 160, 161, 175, 216, 219, 289, 295, 316, 333, 372, 385, 390, 400, 479]
}

class CausalAnalyzer:
    def __init__(self, data_dir, output_dir=None, debug=False):
        """
        Args:
            data_dir: 数据目录路径
            output_dir: 输出目录路径
        """
        self.data_dir = data_dir
        self.output_dir = output_dir if output_dir else os.path.join(data_dir, 'analysis_results')
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.load_data()
        self.debug = debug
        
    def load_data(self):
        with open(os.path.join(self.data_dir, "track_change_tj.pkl"), "rb") as f:
            self.track_change = pickle.load(f)
        with open(os.path.join(self.data_dir, "tp_info_tj.pkl"), "rb") as f:
            self.tp_info = pickle.load(f)
        with open(os.path.join(self.data_dir, "frame_data_tj.pkl"), "rb") as f:
            self.frame_data = pickle.load(f)
        with open(os.path.join(self.data_dir, "frame_data_tj_processed.pkl"), "rb") as f:
            self.frame_data_processed = pickle.load(f)

        # Processing fragment data. frame_data[fragment_id] is processed into a list with frame_id as the index
        # for fragment_id, fragment_data in self.frame_data.items():
        #     print('Processing fragment data: ', fragment_id)
        #     for frame_id, frame_data in fragment_data.items():
        #         # Processing frame data. frame_data[fragment_id][frame_id] is processed into a dict with track_id as the key
        #         self.frame_data[fragment_id][frame_id] = {track_data['tp_id']: track_data['vehicle_info'] for track_data in frame_data}

        #     self.frame_data[fragment_id] = list(self.frame_data[fragment_id].values())

        # print("Fragment data processed.")

    def prepare_ssm_dataframe(self, fragment_id, ego_id):
        """
        Prepare the dataframe for SSM calculation
        
        Args:
            fragment_id: Scenario ID
            ego_id: Ego vehicle ID

        Returns:
            The dataframe for SSM calculation
        """
        track = self.track_change[fragment_id].get(ego_id, None)
        if track is None:
            return None, None
        start_frame = self.tp_info[fragment_id][ego_id]['State']['frame_id'].iloc[0]
        end_frame = self.tp_info[fragment_id][ego_id]['State']['frame_id'].iloc[-1]

        # Debug
        if self.debug:
            end_frame = np.min([start_frame + 20, self.tp_info[fragment_id][ego_id]['State']['frame_id'].iloc[-1]])
        anomalies_frame_id = track['track_info']['frame_id'][np.where(track['anomalies'] == True)[0]]
        start_frame = np.max([start_frame, np.min(anomalies_frame_id) - 10])
        end_frame = np.min([end_frame, np.max(anomalies_frame_id) + 10])
        # if len(anomalies_frame_id) == 1 and (anomalies_frame_id == end_frame or anomalies_frame_id == start_frame):
        #     anomalies_frame_id = []
        # if anomalies_frame_id[-1] == end_frame:
        #     anomalies_frame_id = np.delete(anomalies_frame_id, -1)
        # if anomalies_frame_id[0] == start_frame:
        #     anomalies_frame_id = np.delete(anomalies_frame_id, 0)
            
        # ssm_frames = self.frame_data_processed[fragment_id][start_frame:end_frame+1]
        return self.frame_data_processed[fragment_id][start_frame: end_frame+1], anomalies_frame_id, start_frame, end_frame
    
    def compute_safety_metrics(self, fragment_id, ego_id):
        """
        Compute Surrogate Safety Metrics (SSM)
        
        Args:
            fragment_id: Scenario ID
            ego_id: Ego vehicle ID
            
        Returns:
            A dictionary containing the safety metrics
        """
        df, anomaly_frames, ego_start_frame, ego_end_frame = self.prepare_ssm_dataframe(fragment_id, ego_id)
        if df is None:
            return None
        num_timesteps = len(df)
        num_agents = len(df[0].keys())
        longitudinal_jerk = np.zeros((num_timesteps-1, num_agents))
        lateral_jerk = np.zeros((num_timesteps-1, num_agents))

        # Calculate the jerk of each agent for anomaly classification
        for t in range(num_timesteps-1):
            for tp_id in df[t].keys():
                if tp_id in df[t+1].keys():
                    time_step = df[t+1][tp_id]['timestamp_ms'] - df[t][tp_id]['timestamp_ms'] / 1000.0
                    longitudinal_jerk[t] = (df[t+1][tp_id]['a_lon'] - df[t][tp_id]['a_lon']) / time_step
                    lateral_jerk[t] = (df[t+1][tp_id]['a_lat'] - df[t][tp_id]['a_lat']) / time_step
                    df[t][tp_id]['longitudinal_jerk'] = longitudinal_jerk[t]
                    df[t][tp_id]['lateral_jerk'] = lateral_jerk[t]
        
        # TODO: detect the type of anomaly
        # for anomaly_frame in anomalies_frames:
        #     df_anomaly = df[anomaly_frame]

        #     # 计算SSM指标
        distance_values = {}
        ttc_values = {}
        # ttc_min = np.full((num_timesteps, num_agents), np.inf)
        drac_values = {}
        psd_values = {}
        tadv_values = {}
        ttc2d_values = {}
        act_values = {}

        start_frame = {}
        start_frame[ego_id] = ego_start_frame
        # drac_max = np.full((num_timesteps, num_agents), 0.0)
        
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
                    'type_i': [df[t][tp_id]['type']],
                    'x_j': [df[t][j]['x']],
                    'y_j': [df[t][j]['y']],
                    'vx_j': [df[t][j]['vx']],
                    'vy_j': [df[t][j]['vy']],
                    'hx_j': [np.cos(df[t][j]['heading_rad'])],
                    'hy_j': [np.sin(df[t][j]['heading_rad'])],
                    'length_j': [df[t][j]['length']],
                    'width_j': [df[t][j]['width']],
                    'type_j': [df[t][j]['type']]
                })

                # Calculate the distance between the two agents
                # distance = np.linalg.norm(ssm_data['x_i'] - ssm_data['x_j'])
                distance = CurrentD(ssm_data, toreturn='values')
                # 计算TTC和DRAC
                ttc_result = TTC(ssm_data, toreturn='values')
                drac_result = DRAC(ssm_data, toreturn='values')
                psd_result = PSD(ssm_data, toreturn='values')
                tadv_result = TAdv(ssm_data, toreturn='values')
                ttc2d_result = TTC2D(ssm_data, toreturn='values')
                act_result = ACT(ssm_data, toreturn='values')
                
                # Add results to the result dictionary
                if tp_id in ttc_values.keys():
                    ttc_values[tp_id].append(ttc_result[0])
                else:
                    ttc_values[tp_id] = [ttc_result[0]]
                    start_frame[tp_id] = ego_start_frame + t
                if tp_id in drac_values.keys():
                    drac_values[tp_id].append(drac_result[0])
                else:
                    drac_values[tp_id] = [drac_result[0]]
                if tp_id in psd_values.keys():
                    psd_values[tp_id].append(psd_result[0])
                else:
                    psd_values[tp_id] = [psd_result[0]]
                if tp_id in tadv_values.keys():
                    tadv_values[tp_id].append(tadv_result[0])
                else:
                    tadv_values[tp_id] = [tadv_result[0]]
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
            # # 计算最小TTC和最大DRAC
            # other_vehicles = [j for j in range(num_agents) if j != i]
            # if other_vehicles:
            #     ttc_min[t, i] = np.min(ttc_values[t, i, other_vehicles])
            #     drac_max[t, i] = np.max(drac_values[t, i, other_vehicles])
        
        # 汇总安全指标
        safety_metrics = {
            "ego_id": ego_id,
            "fragment_id": fragment_id,
            "anomaly_frames": anomaly_frames,
            "start_frame": start_frame,
            "end_frame": ego_end_frame,
            "ttc_values": ttc_values,
            "drac_values": drac_values,
            "psd_values": psd_values,
            "tadv_values": tadv_values,
            "ttc2d_values": ttc2d_values,
            "act_values": act_values,
            "distance_values": distance_values
        }

        return safety_metrics
        
        # # 计算角速度（朝向变化率）
        # angular_velocities = np.zeros((num_timesteps-1, num_agents, 1))
        # for t in range(num_timesteps-1):
        #     # 处理角度跨越±π边界的情况
        #     yaw_diff = yaws[t+1] - yaws[t]
        #     yaw_diff = np.where(yaw_diff > np.pi, yaw_diff - 2*np.pi, yaw_diff)
        #     yaw_diff = np.where(yaw_diff < -np.pi, yaw_diff + 2*np.pi, yaw_diff)
        #     angular_velocities[t] = yaw_diff / TIME_STEP
        
        # # 计算急刹车事件（纵向加速度低于阈值）
        # hard_braking_threshold = -LON_ACC_THRESHOLD  # m/s²
        # hard_braking_events = longitudinal_acc < hard_braking_threshold
        
        # # 计算急加速事件（纵向加速度高于阈值）
        # hard_acceleration_threshold = LON_ACC_THRESHOLD  # m/s²
        # hard_acceleration_events = longitudinal_acc > hard_acceleration_threshold
        
        # # 计算急转弯事件（横向加速度超过阈值或角速度超过阈值）
        # hard_turning_threshold_angular = 0.3  # rad/s
        # hard_turning_threshold_lateral = LAT_ACC_THRESHOLD  # m/s²
        # hard_turning_events_angular = np.zeros((num_timesteps, num_agents), dtype=bool)
        # for t in range(num_timesteps-1):
        #     hard_turning_events_angular[t] = np.abs(angular_velocities[t, :, 0]) > hard_turning_threshold_angular
        # hard_turning_events_lateral = np.abs(lateral_acc) > hard_turning_threshold_lateral
        
        # # 合并两种急转弯判断
        # hard_turning_events = np.zeros((num_timesteps, num_agents), dtype=bool)
        # for t in range(num_timesteps):
        #     for i in range(num_agents):
        #         if t < num_timesteps-1:
        #             hard_turning_events[t, i] = (hard_turning_events_angular[t, i] or 
        #                                         hard_turning_events_lateral[t, i])
        #         else:
        #             hard_turning_events[t, i] = hard_turning_events_lateral[t, i]
        
        # # 计算危险接近事件（距离低于阈值）
        # dangerous_proximity_threshold = 5.0  # 米
        # dangerous_proximity_events = min_distances < dangerous_proximity_threshold
        
        # # 计算急加加速事件（加加速度超过阈值）
        # risky_jerk_threshold_longitudinal = LONJ_THRESHOLD  # m/s³
        # risky_jerk_threshold_lateral = LATJ_THRESHOLD  # m/s³
        # risky_longitudinal_jerk_events = np.abs(longitudinal_jerk) > risky_jerk_threshold_longitudinal
        # risky_lateral_jerk_events = np.abs(lateral_jerk) > risky_jerk_threshold_lateral
    
    def visualize_safety_metrics(self, safety_metrics):
        """
        Visualize the safety metrics
        
        Args:
            safety_metrics: A dictionary containing the safety metrics
        """
        fig, axs = plt.subplots(4, 2, figsize=(20, 20))
        fig.suptitle(f'Causal Analysis for Agent {safety_metrics["ego_id"]}', fontsize=20, fontweight='bold')
        start_frame = safety_metrics['start_frame'][safety_metrics['ego_id']]
        end_frame = safety_metrics['end_frame']
        anomaly_frames = safety_metrics['anomaly_frames']
        # frame_range = np.arange(start_frame, end_frame + 1)

        for i, tp_id in enumerate(safety_metrics['ttc_values'].keys()):
            frame_range = np.arange(start_frame, start_frame + len(safety_metrics['ttc_values'][tp_id]))
            if np.any(np.array(safety_metrics['ttc_values'][tp_id]) < TTC_NORMAL_THRESHOLD):
                axs[0, 0].plot(frame_range, safety_metrics['ttc_values'][tp_id], label=f'tp_id={tp_id}')
            if np.any(np.array(safety_metrics['drac_values'][tp_id]) > DRAC_NORMAL_THRESHOLD):
                axs[0, 1].plot(frame_range, safety_metrics['drac_values'][tp_id], label=f'tp_id={tp_id}')
            if np.any(np.array(safety_metrics['psd_values'][tp_id]) < PSD_NORMAL_THRESHOLD):
                axs[1, 0].plot(frame_range, safety_metrics['psd_values'][tp_id], label=f'tp_id={tp_id}')
            if np.any(np.array(safety_metrics['tadv_values'][tp_id]) < TADV_NORMAL_THRESHOLD):
                axs[1, 1].plot(frame_range, safety_metrics['tadv_values'][tp_id], label=f'tp_id={tp_id}')
            if np.any(np.array(safety_metrics['ttc2d_values'][tp_id]) < TTC_NORMAL_THRESHOLD):
                axs[2, 0].plot(frame_range, safety_metrics['ttc2d_values'][tp_id], label=f'tp_id={tp_id}')
            if np.any(np.array(safety_metrics['act_values'][tp_id]) < ACT_NORMAL_THRESHOLD):
                axs[2, 1].plot(frame_range, safety_metrics['act_values'][tp_id], label=f'tp_id={tp_id}')
            if np.any(np.array(safety_metrics['distance_values'][tp_id]) < DISTANCE_NORMAL_THRESHOLD):
                axs[3, 0].plot(frame_range, safety_metrics['distance_values'][tp_id], label=f'tp_id={tp_id}')

        axs[0, 0].set_title(f"Time-to-Collision (TTC)")
        axs[0, 0].set_xlabel('Frame')
        axs[0, 0].set_ylabel('TTC (s)')
        axs[0, 0].set_ylim(0, 10)
        axs[0, 0].axhline(y=TTC_CRITICAL_THRESHOLD, color='r', linestyle='--', alpha=0.5)
        # axs[0, 0].text(start_frame, TTC_CRITICAL_THRESHOLD+0.5, f'Critical threshold ({TTC_CRITICAL_THRESHOLD} s)', color='r', alpha=0.7)
        # Add vertical lines for anomaly frames
        # axs[0, 0].axvline(x=min(safety_metrics['anomaly_frames']), color='gray', linestyle='--', alpha=0.3)
        # axs[0, 0].axvline(x=max(safety_metrics['anomaly_frames']), color='gray', linestyle='--', alpha=0.3)
        # axs[0, 0].scatter(anomaly_frames, safety_metrics['ttc_values'][tp_id][anomaly_frames], 
        #                 color='red', marker='x', s=100, label=f'Anomaly (tp_id={tp_id})')
            
        # 绘制DRAC
        axs[0, 1].set_title(f"Deceleration Rate to Avoid Crash (DRAC)")
        axs[0, 1].set_xlabel('Frame')
        axs[0, 1].set_ylabel('DRAC (m/s²)')
        axs[0, 1].axhline(y=DRAC_CRITICAL_THRESHOLD, color='r', linestyle='--', alpha=0.5)
        # axs[0, 1].text(start_frame, DRAC_CRITICAL_THRESHOLD+0.5, f'Critical threshold ({DRAC_CRITICAL_THRESHOLD} m/s²)', color='r', alpha=0.7)
        # axs[0, 1].scatter(anomaly_frames, safety_metrics['drac_values'][tp_id][anomaly_frames], 
        #                 color='red', marker='x', s=100, label=f'Anomaly (tp_id={tp_id})')
        # 绘制PSD
        axs[1, 0].set_title(f"Proportion of Stopping Distance (PSD)")
        axs[1, 0].set_xlabel('Frame')
        axs[1, 0].set_ylabel('PSD')
        axs[1, 0].set_ylim(0, 5)
        axs[1, 0].axhline(y=PSD_CRITICAL_THRESHOLD, color='r', linestyle='--', alpha=0.5)
        # axs[1, 0].text(start_frame, PSD_CRITICAL_THRESHOLD+0.5, f'Critical threshold ({PSD_CRITICAL_THRESHOLD})', color='r', alpha=0.7)
        # axs[1, 0].scatter(anomaly_frames, safety_metrics['psd_values'][tp_id][anomaly_frames], 
        #                 color='red', marker='x', s=100, label=f'Anomaly (tp_id={tp_id})')
        # 绘制TAdv
        axs[1, 1].set_title(f'Time Advantage (TAdv)')
        axs[1, 1].set_xlabel('Frame')
        axs[1, 1].set_ylabel('TAdv (s)')
        axs[1, 1].set_ylim(0, 10)
        axs[1, 1].axhline(y=TADV_CRITICAL_THRESHOLD, color='r', linestyle='--', alpha=0.5)
        # axs[1, 1].scatter(anomaly_frames, safety_metrics['tadv_values'][tp_id][anomaly_frames], 
        #                 color='red', marker='x', s=100, label=f'Anomaly (tp_id={tp_id})')
        
        # 绘制TTC2D
        axs[2, 0].set_title(f'2D Time-to-Collision (TTC2D)')
        axs[2, 0].set_xlabel('Frame')
        axs[2, 0].set_ylabel('TTC2D (s)')
        axs[2, 0].set_ylim(0, 10)
        axs[2, 0].axhline(y=TTC_CRITICAL_THRESHOLD, color='r', linestyle='--', alpha=0.5)
        # axs[2, 0].scatter(anomaly_frames, safety_metrics['ttc2d_values'][tp_id][anomaly_frames], 
        #                 color='red', marker='x', s=100, label=f'Anomaly (tp_id={tp_id})')
        
        # 绘制ACT
        axs[2, 1].set_title(f'Aggregated Collision Time (ACT)')
        axs[2, 1].set_xlabel('Frame')
        axs[2, 1].set_ylabel('ACT (s)')
        axs[2, 1].set_ylim(0, 10)
        axs[2, 1].axhline(y=ACT_CRITICAL_THRESHOLD, color='r', linestyle='--', alpha=0.5)
        # axs[2, 1].scatter(anomaly_frames, safety_metrics['act_values'][tp_id][anomaly_frames], 
        #                 color='red', marker='x', s=100, label=f'Anomaly (tp_id={tp_id})')
        
        # 绘制距离
        axs[3, 0].set_title(f'Distance')
        axs[3, 0].set_xlabel('Frame')
        axs[3, 0].set_ylabel('Distance (m)')
        axs[3, 0].set_ylim(0, 20)
        
        for frame in anomaly_frames:
            for ax in axs.flat:
                ax.set_xlim(start_frame, end_frame)
                ax.grid(True)
                ax.legend()
                ax.axvline(x=frame, color='r', linestyle='--', alpha=0.5)
    
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"ssm_{safety_metrics['fragment_id']}_{safety_metrics['ego_id']}.png"))
        plt.close()
        plt.show()
    
    def visualize_acceleration_analysis(self, fragment_id, ego_id):
        """
        创建加速度和加加速度的详细分析图
        
        Args:
            safety_metrics: 安全指标字典
            agent_idx: 代理索引
            save_path: 保存路径
        """
        df, anomaly_frames, start_frame, end_frame = self.prepare_ssm_dataframe(fragment_id, ego_id)
        if df is None:
            return None
        num_timesteps = len(df)
        num_agents = len(df[0].keys())
        longitudinal_jerk = np.zeros(num_timesteps-1)
        lateral_jerk = np.zeros(num_timesteps-1)

        a_lon = np.zeros(num_timesteps)
        a_lat = np.zeros(num_timesteps)
        yaw = np.zeros(num_timesteps)
        heading = np.zeros(num_timesteps)
        yaw_rate = np.zeros(num_timesteps-1)
        heading_rate = np.zeros(num_timesteps-1)

        for t in range(num_timesteps):
            a_lon[t] = df[t][ego_id]['a_lon']
            a_lat[t] = df[t][ego_id]['a_lat']
            yaw[t] = df[t][ego_id]['yaw_rad']
            heading[t] = df[t][ego_id]['heading_rad']

        # # 计算急刹车事件（纵向加速度低于阈值）
        # hard_braking_threshold = -LON_BRAKING_THRESHOLD  # m/s²
        # hard_braking_events = a_lon < hard_braking_threshold
        
        # # 计算急加速事件（纵向加速度高于阈值）
        # hard_acceleration_threshold = LON_ACCELERATION_THRESHOLD  # m/s²
        # hard_acceleration_events = a_lon > hard_acceleration_threshold
        
        # # 计算急转弯事件（横向加速度超过阈值或角速度超过阈值）
        # hard_turning_threshold_angular = 0.3  # rad/s
        # hard_turning_threshold_lateral = LAT_ACC_THRESHOLD  # m/s²
        # hard_turning_events_angular = np.zeros((num_timesteps, num_agents), dtype=bool)
        # for t in range(num_timesteps-1):
        #     hard_turning_events_angular[t] = np.abs(angular_velocities[t, :, 0]) > hard_turning_threshold_angular
        # hard_turning_events_lateral = np.abs(lateral_acc) > hard_turning_threshold_lateral  

        for t in range(num_timesteps-1):
            time_step = df[t+1][ego_id]['timestamp_ms'] - df[t][ego_id]['timestamp_ms'] / 1000.0
            longitudinal_jerk[t] = (df[t+1][ego_id]['a_lon'] - df[t][ego_id]['a_lon']) / time_step
            lateral_jerk[t] = (df[t+1][ego_id]['a_lat'] - df[t][ego_id]['a_lat']) / time_step
            yaw_rate[t] = (df[t+1][ego_id]['yaw_rad'] - df[t][ego_id]['yaw_rad']) / time_step
            heading_rate[t] = (df[t+1][ego_id]['heading_rad'] - df[t][ego_id]['heading_rad']) / time_step
        
        fig, axs = plt.subplots(4, 2, figsize=(30, 12))
        
        # 1. 纵向加速度图
        axs[0, 0].plot(range(start_frame, end_frame+1), a_lon, 'b-', label='Longitudinal')
        axs[0, 0].set_title(f'Longitudinal Acceleration')
        axs[0, 0].set_xlabel('Frame')
        axs[0, 0].set_ylabel('Acceleration (m/s²)')
        axs[0, 0].grid(True)
        # 标记异常帧
        if len(anomaly_frames) > 0:
            axs[0, 0].scatter(anomaly_frames, 
                            a_lon[anomaly_frames - start_frame], 
                            color='red', marker='x', s=100, label='Anomaly Frames')
        axs[0, 0].legend()
        
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
        
        # 2. 横向加速度图
        axs[0, 1].plot(range(start_frame, end_frame+1), a_lat, 'g-', label='Lateral')
        axs[0, 1].set_title(f'Lateral Acceleration')
        axs[0, 1].set_xlabel('Frame')
        axs[0, 1].set_ylabel('Acceleration (m/s²)')
        axs[0, 1].grid(True)

        # 标记异常帧
        if len(anomaly_frames) > 0:
            axs[0, 1].scatter(anomaly_frames, 
                            a_lat[anomaly_frames - start_frame], 
                            color='red', marker='x', s=100, label='Anomaly Frames')
        axs[0, 1].legend()
        
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
        # axs[0, 1].legend()
        
        # 3. 纵向加加速度图
        axs[1, 0].plot(range(start_frame, end_frame), longitudinal_jerk, 'b-', label='Longitudinal')
        axs[1, 0].set_title(f'Longitudinal Jerk')
        axs[1, 0].set_xlabel('Frame')
        axs[1, 0].set_ylabel('Jerk (m/s³)')
        axs[1, 0].grid(True)

        # 标记异常帧
        if len(anomaly_frames) > 0:
            axs[1, 0].scatter(anomaly_frames, 
                            longitudinal_jerk[anomaly_frames - start_frame], 
                            color='red', marker='x', s=100, label='Anomaly Frames')
        axs[1, 0].legend()
        # # 添加舒适阈值线
        # axs[1, 0].axhline(y=LONJ_THRESHOLD, color='r', linestyle='--', alpha=0.5)
        # axs[1, 0].axhline(y=-LONJ_THRESHOLD, color='r', linestyle='--', alpha=0.5)
        # axs[1, 0].text(0, LONJ_THRESHOLD+0.5, f'Comfort threshold (±{LONJ_THRESHOLD} m/s³)', color='r', alpha=0.7)
        # axs[1, 0].legend()

        # 4. 横向加加速度图
        axs[1, 1].plot(range(start_frame, end_frame), lateral_jerk, 'g-', label='Lateral')
        axs[1, 1].set_title(f'Lateral Jerk')
        axs[1, 1].set_xlabel('Frame')
        axs[1, 1].set_ylabel('Jerk (m/s³)')
        axs[1, 1].grid(True)

        # 标记异常帧
        if len(anomaly_frames) > 0:
            axs[1, 1].scatter(anomaly_frames, 
                            lateral_jerk[anomaly_frames - start_frame], 
                            color='red', marker='x', s=100, label='Anomaly Frames')
        axs[1, 1].legend()

        # 5. 偏转角图
        axs[2, 0].plot(range(start_frame, end_frame+1), yaw, 'b-', label='Yaw')
        axs[2, 0].set_title(f'Yaw')
        axs[2, 0].set_xlabel('Frame')
        axs[2, 0].set_ylabel('Yaw (rad)')
        axs[2, 0].grid(True)

        if len(anomaly_frames) > 0:
            axs[2, 0].scatter(anomaly_frames, 
                            yaw[anomaly_frames - start_frame], 
                            color='red', marker='x', s=100, label='Anomaly Frames')
        axs[2, 0].legend()
        
        # 6. 偏转角变化率图
        axs[2, 1].plot(range(start_frame, end_frame), yaw_rate, 'b-', label='Yaw Rate')
        axs[2, 1].set_title(f'Yaw Rate')
        axs[2, 1].set_xlabel('Frame')
        axs[2, 1].set_ylabel('Yaw Rate (rad/s)')
        axs[2, 1].grid(True)

        if len(anomaly_frames) > 0:
            axs[2, 1].scatter(anomaly_frames, 
                            yaw_rate[anomaly_frames - start_frame], 
                            color='red', marker='x', s=100, label='Anomaly Frames')
        axs[2, 1].legend()

        # 7. 偏转角变化率图
        axs[3, 0].plot(range(start_frame, end_frame+1), heading, 'b-', label='Heading')
        axs[3, 0].set_title(f'Heading')
        axs[3, 0].set_xlabel('Frame')
        axs[3, 0].set_ylabel('Heading (rad)')
        axs[3, 0].grid(True)

        if len(anomaly_frames) > 0:
            axs[3, 0].scatter(anomaly_frames, 
                            heading[anomaly_frames - start_frame], 
                            color='red', marker='x', s=100, label='Anomaly Frames')
        axs[3, 0].legend()

        # 8. 偏转角变化率图
        axs[3, 1].plot(range(start_frame, end_frame), heading_rate, 'b-', label='Heading Rate')
        axs[3, 1].set_title(f'Heading Rate')
        axs[3, 1].set_xlabel('Frame')
        axs[3, 1].set_ylabel('Heading Rate (rad/s)')
        axs[3, 1].grid(True)

        if len(anomaly_frames) > 0:
            axs[3, 1].scatter(anomaly_frames, 
                            heading_rate[anomaly_frames - start_frame], 
                            color='red', marker='x', s=100, label='Anomaly Frames')
        axs[3, 1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"acc_{fragment_id}_{ego_id}.png"))
        plt.close()
        plt.show()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/tj")
    parser.add_argument("--output_dir", type=str, default="./output/tj/causal_analysis")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    ego_id = 1
    fragment_id = "7_28_1 R21"

    causal_analyzer = CausalAnalyzer(args.data_dir, args.output_dir, args.debug)
    # causal_analyzer.analyze()
    # for fragment_id in fragment_id_list:
    #     for ego_id in ego_id_dict[fragment_id]:
    #         causal_analyzer.visualize_acceleration_analysis(fragment_id, ego_id)
    #         ssm = causal_analyzer.compute_safety_metrics(fragment_id, ego_id)
    #         causal_analyzer.visualize_safety_metrics(ssm)
    causal_analyzer.visualize_acceleration_analysis(fragment_id, ego_id)
    ssm = causal_analyzer.compute_safety_metrics(fragment_id, ego_id)
    causal_analyzer.visualize_safety_metrics(ssm)
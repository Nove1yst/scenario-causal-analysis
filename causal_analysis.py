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
import networkx as nx

sys.path.append(os.getcwd())
from ssm.src.longitudinal_ssms import TTC, DRAC, PSD
from ssm.src.two_dimensional_ssms import TAdv, TTC2D, ACT
from ssm.src.geometry_utils import CurrentD
# from utils.collision_utils import calculate_vehicle_bbox
from utils.visualization_utils import visualize_causal_graph

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
# DISTANCE_CRITICAL_THRESHOLD = 20  # Distance critical threshold

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
safety_metrics_list = ['tadv', 'ttc2d', 'act']

class CausalAnalyzer:
    def __init__(self, data_dir, output_dir=None):
        """
        Args:
            data_dir: 数据目录路径
            output_dir: 输出目录路径
        """
        self.data_dir = data_dir
        self.output_dir = output_dir if output_dir else os.path.join(data_dir, 'analysis_results')
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.load_data()
        
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

    def prepare_ssm_dataframe(self, fragment_id, ego_id, anomaly_frames_child=None):
        """
        Prepare the dataframe for SSM calculation
        
        Args:
            fragment_id: Scenario ID
            ego_id: Ego vehicle ID

        Returns:
            The dataframe for SSM calculation
        """
        if anomaly_frames_child is not None:
            start_frame = self.tp_info[fragment_id][ego_id]['State']['frame_id'].iloc[0]
            end_frame = self.tp_info[fragment_id][ego_id]['State']['frame_id'].iloc[-1]
            anomaly_frames_id = set()
            if anomaly_frames_child is not None:
                for anomaly_frame in anomaly_frames_child:
                    anomaly_frames_id.update(range(max((start_frame, anomaly_frame-10)), min(end_frame, anomaly_frame+1)))
                anomaly_frames_id = sorted(list(anomaly_frames_id))
                # if anomaly_frames_child.min() - 10 > start_frame:
                    # anomaly_frames_id = list(set(anomaly_frames_id).union(range(anomaly_frames_child.min()-10, anomaly_frames_child.min())))
                # if anomaly_frames_child.max() + 10 < end_frame:
                    # anomaly_frames_id = list(set(anomaly_frames_id).union(range(anomaly_frames_child.max()+1, anomaly_frames_child.max()+10)))
                start_frame = np.max([start_frame, np.min(anomaly_frames_id) - 10])
                end_frame = np.min([end_frame, np.max(anomaly_frames_id)+1])
            else: # Should not come here
                anomaly_frames_id = range(start_frame, end_frame+1)
        else:
            track = self.track_change[fragment_id].get(ego_id, None)
            start_frame = self.tp_info[fragment_id][ego_id]['State']['frame_id'].iloc[0]
            end_frame = self.tp_info[fragment_id][ego_id]['State']['frame_id'].iloc[-1]
            anomaly_frames_id = track['track_info']['frame_id'][np.where(track['anomalies'] == True)[0]]
            start_frame = np.max([start_frame, np.min(anomaly_frames_id) - 10])
            end_frame = np.min([end_frame, np.max(anomaly_frames_id)+1])

        if end_frame in anomaly_frames_id:
            anomaly_frames_id = anomaly_frames_id[:-1]

        return self.frame_data_processed[fragment_id][start_frame: end_frame+1], anomaly_frames_id, start_frame, end_frame
    
    def compute_safety_metrics(self, fragment_id, ego_id, anomaly_frames_child=None):
        """
        Compute Surrogate Safety Metrics (SSM)
        
        Args:
            fragment_id: Scenario ID
            ego_id: Ego vehicle ID
            
        Returns:
            A dictionary containing the safety metrics
        """
        df, anomaly_frames, ego_start_frame, ego_end_frame = self.prepare_ssm_dataframe(fragment_id, ego_id, anomaly_frames_child)
        if df is None:
            return None
        num_timesteps = len(df)
        
        # TODO: detect the type of anomaly
        # for anomaly_frame in anomalies_frames:
        #     df_anomaly = df[anomaly_frame]

        distance_values = {}
        ttc_values = {}
        drac_values = {}
        psd_values = {}
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
                # 计算TTC和DRAC
                # ttc_result = TTC(ssm_data, toreturn='values')
                # drac_result = DRAC(ssm_data, toreturn='values')
                # psd_result = PSD(ssm_data, toreturn='values')
                tadv_result = TAdv(ssm_data, toreturn='values')
                ttc2d_result = TTC2D(ssm_data, toreturn='values')
                act_result = ACT(ssm_data, toreturn='values')
                
                # Add results to the result dictionary
                # if tp_id in ttc_values.keys():
                #     ttc_values[tp_id].append(ttc_result[0])
                # else:
                #     ttc_values[tp_id] = [ttc_result[0]]
                # if tp_id in drac_values.keys():
                #     drac_values[tp_id].append(drac_result[0])
                # else:
                #     drac_values[tp_id] = [drac_result[0]]
                # if tp_id in psd_values.keys():
                #     psd_values[tp_id].append(psd_result[0])
                # else:
                #     psd_values[tp_id] = [psd_result[0]]
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
            "fragment_id": fragment_id,
            "anomaly_frames": anomaly_frames,
            "start_frame": start_frame,
            "end_frame": ego_end_frame,
            # "ttc_values": ttc_values,
            # "drac_values": drac_values,
            # "psd_values": psd_values,
            "tadv_values": tadv_values,
            "ttc2d_values": ttc2d_values,
            "act_values": act_values,
            "distance_values": distance_values
        }

        return safety_metrics
    
    def visualize_safety_metrics(self, safety_metrics):
        """
        Visualize the safety metrics
        
        Args:
            safety_metrics: A dictionary containing the safety metrics
        """
        fig, axs = plt.subplots(2, 2, figsize=(20, 20))
        fig.suptitle(f'Causal Analysis for Agent {safety_metrics["ego_id"]}', fontsize=20, fontweight='bold')
        start_frame = safety_metrics['start_frame'][safety_metrics['ego_id']]
        end_frame = safety_metrics['end_frame']
        anomaly_frames = safety_metrics['anomaly_frames']
        # frame_range = np.arange(start_frame, end_frame + 1)

        for i, tp_id in enumerate(safety_metrics['tadv_values'].keys()):
            frame_range = np.arange(start_frame, start_frame + len(safety_metrics['tadv_values'][tp_id]))
            # if np.any(np.array(safety_metrics['ttc_values'][tp_id]) < TTC_NORMAL_THRESHOLD):
            #     axs[0, 0].plot(frame_range, safety_metrics['ttc_values'][tp_id], label=f'tp_id={tp_id}')
            # if np.any(np.array(safety_metrics['drac_values'][tp_id]) > DRAC_NORMAL_THRESHOLD):
            #     axs[0, 1].plot(frame_range, safety_metrics['drac_values'][tp_id], label=f'tp_id={tp_id}')
            # if np.any(np.array(safety_metrics['psd_values'][tp_id]) < PSD_NORMAL_THRESHOLD):
            #     axs[1, 0].plot(frame_range, safety_metrics['psd_values'][tp_id], label=f'tp_id={tp_id}')
            if np.any(np.array(safety_metrics['tadv_values'][tp_id]) < TADV_NORMAL_THRESHOLD):
                axs[0, 0].plot(frame_range, safety_metrics['tadv_values'][tp_id], label=f'tp_id={tp_id}')
            if np.any(np.array(safety_metrics['ttc2d_values'][tp_id]) < TTC_NORMAL_THRESHOLD):
                axs[0, 1].plot(frame_range, safety_metrics['ttc2d_values'][tp_id], label=f'tp_id={tp_id}')
            if np.any(np.array(safety_metrics['act_values'][tp_id]) < ACT_NORMAL_THRESHOLD):
                axs[1, 0].plot(frame_range, safety_metrics['act_values'][tp_id], label=f'tp_id={tp_id}')
            if np.any(np.array(safety_metrics['distance_values'][tp_id]) < DISTANCE_NORMAL_THRESHOLD):
                axs[1, 1].plot(frame_range, safety_metrics['distance_values'][tp_id], label=f'tp_id={tp_id}')

        # axs[0, 0].set_title(f"Time-to-Collision (TTC)")
        # axs[0, 0].set_xlabel('Frame')
        # axs[0, 0].set_ylabel('TTC (s)')
        # axs[0, 0].set_ylim(0, 10)
        # axs[0, 0].axhline(y=TTC_CRITICAL_THRESHOLD, color='r', linestyle='--', alpha=0.5)
        # # axs[0, 0].text(start_frame, TTC_CRITICAL_THRESHOLD+0.5, f'Critical threshold ({TTC_CRITICAL_THRESHOLD} s)', color='r', alpha=0.7)
            
        # # 绘制DRAC
        # axs[0, 1].set_title(f"Deceleration Rate to Avoid Crash (DRAC)")
        # axs[0, 1].set_xlabel('Frame')
        # axs[0, 1].set_ylabel('DRAC (m/s²)')
        # axs[0, 1].axhline(y=DRAC_CRITICAL_THRESHOLD, color='r', linestyle='--', alpha=0.5)
        # # axs[0, 1].text(start_frame, DRAC_CRITICAL_THRESHOLD+0.5, f'Critical threshold ({DRAC_CRITICAL_THRESHOLD} m/s²)', color='r', alpha=0.7)
        # # axs[0, 1].scatter(anomaly_frames, safety_metrics['drac_values'][tp_id][anomaly_frames], 
        # #                 color='red', marker='x', s=100, label=f'Anomaly (tp_id={tp_id})')
        # # 绘制PSD
        # axs[1, 0].set_title(f"Proportion of Stopping Distance (PSD)")
        # axs[1, 0].set_xlabel('Frame')
        # axs[1, 0].set_ylabel('PSD')
        # axs[1, 0].set_ylim(0, 5)
        # axs[1, 0].axhline(y=PSD_CRITICAL_THRESHOLD, color='r', linestyle='--', alpha=0.5)
        # # axs[1, 0].text(start_frame, PSD_CRITICAL_THRESHOLD+0.5, f'Critical threshold ({PSD_CRITICAL_THRESHOLD})', color='r', alpha=0.7)
        # # axs[1, 0].scatter(anomaly_frames, safety_metrics['psd_values'][tp_id][anomaly_frames], 
        # #                 color='red', marker='x', s=100, label=f'Anomaly (tp_id={tp_id})')
        # 绘制TAdv
        axs[0, 0].set_title(f'Time Advantage (TAdv)')
        axs[0, 0].set_xlabel('Frame')
        axs[0, 0].set_ylabel('TAdv (s)')
        axs[0, 0].set_ylim(0, 10)
        axs[0, 0].axhline(y=TADV_CRITICAL_THRESHOLD, color='r', linestyle='--', alpha=0.5)
        # axs[1, 1].scatter(anomaly_frames, safety_metrics['tadv_values'][tp_id][anomaly_frames], 
        #                 color='red', marker='x', s=100, label=f'Anomaly (tp_id={tp_id})')
        
        # 绘制TTC2D
        axs[0, 1].set_title(f'2D Time-to-Collision (TTC2D)')
        axs[0, 1].set_xlabel('Frame')
        axs[0, 1].set_ylabel('TTC2D (s)')
        axs[0, 1].set_ylim(0, 10)
        axs[0, 1].axhline(y=TTC_CRITICAL_THRESHOLD, color='r', linestyle='--', alpha=0.5)
        # axs[0, 1].scatter(anomaly_frames, safety_metrics['ttc2d_values'][tp_id][anomaly_frames], 
        #                 color='red', marker='x', s=100, label=f'Anomaly (tp_id={tp_id})')
        
        # 绘制ACT
        axs[1, 0].set_title(f'Aggregated Collision Time (ACT)')
        axs[1, 0].set_xlabel('Frame')
        axs[1, 0].set_ylabel('ACT (s)')
        axs[1, 0].set_ylim(0, 10)
        axs[1, 0].axhline(y=ACT_CRITICAL_THRESHOLD, color='r', linestyle='--', alpha=0.5)
        # axs[2, 1].scatter(anomaly_frames, safety_metrics['act_values'][tp_id][anomaly_frames], 
        #                 color='red', marker='x', s=100, label=f'Anomaly (tp_id={tp_id})')
        
        # 绘制距离
        axs[1, 1].set_title(f'Distance')
        axs[1, 1].set_xlabel('Frame')
        axs[1, 1].set_ylabel('Distance (m)')
        axs[1, 1].set_ylim(0, 20)
        
        for frame in anomaly_frames:
            for ax in axs.flat:
                ax.set_xlim(start_frame, end_frame)
                ax.grid(True)
                ax.legend()
                ax.axvline(x=frame, color='r', linestyle='--', alpha=0.5)
    
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f"{safety_metrics['fragment_id']}_{safety_metrics['ego_id']}")
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f"ssm_{safety_metrics['fragment_id']}_{safety_metrics['ego_id']}.png"))
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

    def visualize_causal_graph(self, causal_graph, fragment_id, ego_id):
        """
        Visualize the causal graph using networkx and matplotlib.

        Args:
            causal_graph: A dictionary representing the causal graph where keys are agent IDs and values are lists of agent IDs they influence.
        """
        G = nx.DiGraph()

        # Add edges to the graph
        for agent, influenced_agents in causal_graph.items():
            for influenced_agent, ssm_type, critical_frames in influenced_agents:
                # 添加带有关键帧信息的边
                G.add_edge(agent, influenced_agent, 
                          ssm=ssm_type, 
                          critical_frames=critical_frames,
                          label=f"{ssm_type}\n Frames: {critical_frames[:3]}...")

        # 绘制图形
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(G, seed=42)  # 所有节点的位置
        
        # 绘制节点
        nx.draw_networkx_nodes(G, pos, 
                              node_color='lightblue', 
                              node_size=2000)
        
        # 绘制边
        nx.draw_networkx_edges(G, pos, 
                              arrowstyle='->', 
                              arrowsize=20, 
                              width=2)
        
        # 绘制节点标签
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
        
        # 获取边标签并绘制
        edge_labels = {(u, v): d['label'] for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

        plt.title(f"Causal Graph (Fragment: {fragment_id}, Ego: {ego_id})")
        plt.axis('off')
        
        # 保存图形
        save_path = os.path.join(self.output_dir, f"{fragment_id}_{ego_id}")
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f"cg_{fragment_id}_{ego_id}.png"))
        plt.close()

    def analyze(self, fragment_id, ego_id, anomaly_frames_child=None, visualize_acc=True, visualize_ssm=True, visualize_cg=True, visited_nodes=None, depth=0, max_depth=3):
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
        
        # df, anomaly_frames, start_frame, end_frame = self.prepare_ssm_dataframe(fragment_id, ego_id, anomaly_frames_child)
        # if df is None:
        #     return None
        
        if visualize_acc and depth == 0:
            self.visualize_acceleration_analysis(fragment_id, ego_id)
        
        ssm_dataframe = self.compute_safety_metrics(fragment_id, ego_id, anomaly_frames_child)
        anomaly_frames = ssm_dataframe["anomaly_frames"]
        if visualize_ssm and depth == 0:
            self.visualize_safety_metrics(ssm_dataframe)

        # Initialize a dictionary to store critical conditions for each SSM
        critical_conditions = {
            "ttc": [],
            "drac": [],
            "psd": [],
            "tadv": [],
            "ttc2d": [],
            "act": [],
            # "distance": []
        }

        # Define critical thresholds for each SSM
        critical_thresholds = {
            "ttc": TTC_CRITICAL_THRESHOLD,
            "drac": DRAC_CRITICAL_THRESHOLD,
            "psd": PSD_CRITICAL_THRESHOLD,
            "tadv": TADV_CRITICAL_THRESHOLD,
            "ttc2d": TTC_CRITICAL_THRESHOLD,
            "act": ACT_CRITICAL_THRESHOLD,
            # "distance": DISTANCE_CRITICAL_THRESHOLD
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
                            if abs(cf - af) <= 10:  # 1s
                                # Check distance at this critical frame
                                distance_idx = critical_frames_indices[cf_idx]
                                if distance_idx < len(ssm_dataframe["distance_values"][tp_id]) and ssm_dataframe["distance_values"][tp_id][distance_idx] < 15.0:
                                    relevant_critical_frames.append(cf)
                                    break
                    
                    if relevant_critical_frames and len(relevant_critical_frames) > 3:  # 0.3s
                        critical_conditions[ssm].append((tp_id, is_critical, relevant_critical_frames))
                        critical_frames_child[tp_id] = relevant_critical_frames
                        
                        # 记录每个交互对象的第一个关键帧
                        if tp_id not in first_critical_frames or min(relevant_critical_frames) < first_critical_frames[tp_id]:
                            first_critical_frames[tp_id] = min(relevant_critical_frames)

        # 创建带边属性的因果图
        causal_graph = {}
        for ssm, conditions in critical_conditions.items():
            for tp_id, is_critical, critical_frames in conditions:
                if critical_frames:
                    if tp_id not in causal_graph:
                        causal_graph[tp_id] = []
                    causal_graph[tp_id].append((ego_id, ssm, critical_frames))

        # 递归地为每个父节点构建因果树
        complete_causal_graph = causal_graph.copy()
        for parent_id in causal_graph.keys():
            # 使用父节点的第一个关键帧作为其异常帧
            if parent_id in first_critical_frames:
                # 递归调用analyze函数，将父节点作为新的ego_id
                parent_graph = self.analyze(
                    fragment_id, 
                    parent_id, 
                    anomaly_frames_child=[first_critical_frames[parent_id]],
                    visualize_acc=False, 
                    visualize_ssm=False, 
                    visualize_cg=False,
                    visited_nodes=visited_nodes.copy(),  # 传递已访问节点的副本
                    depth=depth+1,
                    max_depth=max_depth
                )
                
                # 合并父节点的因果图到完整因果图
                if parent_graph:
                    for p_id, edges in parent_graph.items():
                        if p_id not in complete_causal_graph:
                            complete_causal_graph[p_id] = []
                        complete_causal_graph[p_id].extend(edges)

        # 只在顶层调用时可视化和保存完整因果图
        if depth == 0 and visualize_cg:
            self.visualize_causal_graph(complete_causal_graph, fragment_id, ego_id)
            
            # 保存完整因果图到文本文件
            save_path = os.path.join(self.output_dir, f"{fragment_id}_{ego_id}")
            os.makedirs(save_path, exist_ok=True)
            causal_graph_file = os.path.join(save_path, f"cg_{fragment_id}_{ego_id}.txt")
            with open(causal_graph_file, "w") as f:
                f.write(f"Fragment ID: {fragment_id}\n")
                f.write(f"Ego ID: {ego_id}\n\n")
                f.write("Causal Graph:\n")
                for tp_id, edges in complete_causal_graph.items():
                    f.write(f"\nTarget vehicle {tp_id}:\n")
                    for target_id, ssm, critical_frames in edges:
                        f.write(f"  - Influences: {target_id}\n")
                        f.write(f"  - Safety metric: {ssm}\n")
                        f.write(f"  - Critical frames: {critical_frames[:5]}")
                        if len(critical_frames) > 5:
                            f.write(f" ... (total {len(critical_frames)} frames)")
                        f.write("\n")
                        
                        # 添加距离信息
                        if tp_id in ssm_dataframe["distance_values"] and target_id == ego_id:
                            f.write(f"  - Distances at critical frames: [")
                            distances = []
                            for cf in critical_frames[:5]:  # 只显示前5帧以避免输出过多
                                if tp_id in ssm_dataframe["start_frame"]:
                                    start_frame_tp = ssm_dataframe["start_frame"][tp_id]
                                    cf_idx = cf - start_frame_tp
                                    if 0 <= cf_idx < len(ssm_dataframe["distance_values"][tp_id]):
                                        distances.append(f"{ssm_dataframe['distance_values'][tp_id][cf_idx]:.2f}m")
                            f.write(", ".join(distances))
                            if len(critical_frames) > 5:
                                f.write(", ...")
                            f.write("]\n")
            
            print(f"Complete causal graph saved to {causal_graph_file}")
        
        return complete_causal_graph

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/tj")
    parser.add_argument("--output_dir", type=str, default="./output/tj/causal_analysis")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    causal_analyzer = CausalAnalyzer(args.data_dir, args.output_dir)
    # causal_analyzer.analyze("7_28_1 R21", 9, visualize_acc=False, visualize_ssm=False, visualize_cg=True, depth=0, max_depth=2)
    for fragment_id in fragment_id_list:
        for ego_id in ego_id_dict[fragment_id]:
            causal_analyzer.analyze(fragment_id, ego_id, visualize_acc=True, visualize_ssm=True, visualize_cg=True)
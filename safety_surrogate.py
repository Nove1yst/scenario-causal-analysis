import os
import sys

sys.path.append(os.getcwd())

import json
import glob
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.animation import FuncAnimation
from my_utils.collision_utils import check_collision, calculate_vehicle_bbox

VEHICLE_LENGTH = 4.4
VEHICLE_WIDTH = 1.8
FRAME_RATE = 20
TIME_STEP = 1.0 / FRAME_RATE

LATJ_THRESHOLD = 0.5 * 9.81
LONJ_THRESHOLD = 1.2 * 9.81

class SafetySurrogateAnalyzer:
    def __init__(self, args):
        """
        初始化安全代理分析器
        
        Args:
            args: 命令行参数
        """
        self.args = args
        self.log_file_paths, self.results_file_paths = self.parse_scenario_log_dir()
        self.max_iter = self.fetch_max_iter() if hasattr(args, 'optim_method') else None
        
    def analyze(self):
        """
        分析场景日志并提取安全相关指标
        """
        # 遍历所有日志
        for log_path, results_path in tqdm(
            zip(self.log_file_paths, self.results_file_paths), total=len(self.log_file_paths)
            ):
            log = self.parse_json_file(log_path)
            results = self.parse_json_file(results_path)

            # 提取元数据
            name = log["meta_data"]["name"]
            town = log["meta_data"]["town"]
            density = log_path.split("/")[1].split("_")[-1] if len(log_path.split("/")) > 1 else "unknown"
            critical_iter = results["first_metrics"]["iteration"] if "first_metrics" in results else 0

            # 确定要分析的迭代
            if hasattr(self.args, 'opt_iter') and self.args.opt_iter == -1 and self.max_iter is not None:
                if critical_iter <= self.max_iter[density][town]:
                    opt_iter = critical_iter
                else:
                    opt_iter = self.max_iter[density][town]
            else:
                opt_iter = self.args.opt_iter if hasattr(self.args, 'opt_iter') else 0
            
            # 提取状态数据
            states = log["states"][opt_iter]
            
            # 计算安全指标
            safety_metrics = self.compute_safety_metrics(states)
            
            # 可视化安全指标
            save_path = os.path.join(os.path.dirname(log_path), name + f"_safety_iter_{opt_iter}")
            self.visualize_safety_metrics(safety_metrics, save_path)
            
            # 如果需要，保存安全指标数据
            if self.args.save_metrics:
                self.save_safety_metrics(safety_metrics, save_path)
    
    def compute_safety_metrics(self, states):
        """
        计算安全相关指标
        
        Args:
            states: 场景状态数据
            
        Returns:
            包含安全指标的字典
        """
        # 初始化指标存储
        num_agents = len(states[0]["pos"])
        num_timesteps = len(states)
        
        # 提取位置、速度和朝向数据
        positions = np.zeros((num_timesteps, num_agents, 2))
        velocities = np.zeros((num_timesteps, num_agents, 2))
        yaws = np.zeros((num_timesteps, num_agents, 1))
        
        for t, state in enumerate(states):
            positions[t] = np.array(state["pos"])
            velocities[t] = np.array(state["vel"])
            yaws[t] = np.array(state["yaw"])
        
        # 计算加速度（速度变化率）
        accelerations_amplitude = np.zeros((num_timesteps-1, num_agents))
        accelerations = np.zeros((num_timesteps-1, num_agents, 2))
        # 纵向加速度（沿车头方向）和横向加速度（垂直于车头方向）
        longitudinal_acc = np.zeros((num_timesteps-1, num_agents))
        lateral_acc = np.zeros((num_timesteps-1, num_agents))
        
        for t in range(num_timesteps-1):
            a = (velocities[t+1] - velocities[t]) / TIME_STEP
            accelerations_amplitude[t] = np.linalg.norm(a, axis=-1)
            accelerations[t] = a
            
            # 计算纵向和横向加速度
            for i in range(num_agents):
                # 获取车辆朝向（车头方向）
                heading = np.array([np.cos(yaws[t, i][0]), np.sin(yaws[t, i][0])])
                # 计算垂直于车头的方向（右侧）
                perpendicular = np.array([-heading[1], heading[0]])
                
                # 将加速度投影到车头方向（纵向）和垂直方向（横向）
                longitudinal_acc[t, i] = np.dot(a[i], heading)
                lateral_acc[t, i] = np.dot(a[i], perpendicular)
        
        # 计算加加速度（加速度变化率）
        # jerks = np.zeros((num_timesteps-2, num_agents, 2))
        # 纵向和横向加加速度
        longitudinal_jerk = np.zeros((num_timesteps-2, num_agents))
        lateral_jerk = np.zeros((num_timesteps-2, num_agents))
        
        for t in range(num_timesteps-2):
            # jerks[t] = (accelerations[t+1] - accelerations[t]) / TIME_STEP
            # 计算纵向和横向加加速度
            longitudinal_jerk[t] = (longitudinal_acc[t+1] - longitudinal_acc[t]) / TIME_STEP
            lateral_jerk[t] = (lateral_acc[t+1] - lateral_acc[t]) / TIME_STEP
        
        # 计算角速度（朝向变化率）
        angular_velocities = np.zeros((num_timesteps-1, num_agents, 1))
        for t in range(num_timesteps-1):
            # 处理角度跨越±π边界的情况
            yaw_diff = yaws[t+1] - yaws[t]
            yaw_diff = np.where(yaw_diff > np.pi, yaw_diff - 2*np.pi, yaw_diff)
            yaw_diff = np.where(yaw_diff < -np.pi, yaw_diff + 2*np.pi, yaw_diff)
            angular_velocities[t] = yaw_diff / TIME_STEP
        
        # 计算车辆间距离
        distances = np.zeros((num_timesteps, num_agents, num_agents))
        for t in range(num_timesteps):
            for i in range(num_agents):
                for j in range(num_agents):
                    if i != j:
                        distances[t, i, j] = np.linalg.norm(positions[t, i] - positions[t, j])
        
        # 计算最小车辆间距离（对每个车辆）
        min_distances = np.zeros((num_timesteps, num_agents))
        for t in range(num_timesteps):
            for i in range(num_agents):
                other_vehicles = [j for j in range(num_agents) if j != i]
                if other_vehicles:  # 确保有其他车辆
                    min_distances[t, i] = np.min(distances[t, i, other_vehicles])
                else:
                    min_distances[t, i] = float('inf')
        
        # 计算急刹车事件（纵向加速度低于阈值）
        hard_braking_threshold = -3.0  # m/s²，可调整
        hard_braking_events = longitudinal_acc < hard_braking_threshold
        
        # 计算急加速事件（纵向加速度高于阈值）
        hard_acceleration_threshold = 3.0  # m/s²，可调整
        hard_acceleration_events = longitudinal_acc > hard_acceleration_threshold
        
        # 计算急转弯事件（横向加速度超过阈值或角速度超过阈值）
        hard_turning_threshold_angular = 0.3  # rad/s，可调整
        hard_turning_threshold_lateral = 2.0  # m/s²，可调整
        hard_turning_events_angular = np.abs(angular_velocities) > hard_turning_threshold_angular
        hard_turning_events_lateral = np.abs(lateral_acc) > hard_turning_threshold_lateral
        # 合并两种急转弯判断
        hard_turning_events = np.zeros((num_timesteps-1, num_agents), dtype=bool)
        for t in range(num_timesteps-1):
            for i in range(num_agents):
                hard_turning_events[t, i] = (hard_turning_events_angular[t, i][0] or 
                                            hard_turning_events_lateral[t, i])
        
        # 计算危险接近事件（距离低于阈值）
        dangerous_proximity_threshold = 5.0  # 米，可调整
        dangerous_proximity_events = min_distances < dangerous_proximity_threshold
        
        # 计算急加加速事件（加加速度超过阈值）
        risky_jerk_threshold_longitudinal = 1.2 * 9.81  # m/s³
        risky_jerk_threshold_lateral = 0.5 * 9.81  # m/s³
        # risky_jerk_events = np.abs(jerks) > risky_jerk_threshold
        risky_longitudinal_jerk_events = np.abs(longitudinal_jerk) > risky_jerk_threshold_longitudinal
        risky_lateral_jerk_events = np.abs(lateral_jerk) > risky_jerk_threshold_lateral
        # risky_jerk_events = risky_longitudinal_jerk_events or risky_lateral_jerk_events

        # 计算 Time-to-collision (TTC)
        ttc_values = np.full((num_timesteps, num_agents, num_agents), np.inf)  # 初始化为无穷大
        ttc_min = np.full((num_timesteps, num_agents), np.inf)  # 每个车辆在每个时间步的最小 TTC
        ttc_critical_threshold = 3.0  # 临界 TTC 阈值（秒）
        ttc_critical_events = np.zeros((num_timesteps, num_agents), dtype=bool)  # 临界 TTC 事件
        
        # 计算 Time Advantage (TAdv)
        tadv_values = np.full((num_timesteps, num_agents, num_agents), np.inf)  # 初始化为无穷大
        tadv_min = np.full((num_timesteps, num_agents), np.inf)  # 每个车辆在每个时间步的最小 TAdv
        tadv_critical_threshold = 1.5  # 临界 TAdv 阈值（秒）
        tadv_critical_events = np.zeros((num_timesteps, num_agents), dtype=bool)  # 临界 TAdv 事件
        
        # 计算 Post Encroachment Time (PET)
        pet_values = np.full((num_timesteps, num_agents, num_agents), np.inf)  # 初始化为无穷大
        pet_min = np.full((num_timesteps, num_agents), np.inf)  # 每个车辆在每个时间步的最小 PET
        pet_critical_threshold = 1.0  # 临界 PET 阈值（秒）
        pet_critical_events = np.zeros((num_timesteps, num_agents), dtype=bool)  # 临界 PET 事件
        
        # 计算 Aggregated Collision Time (ACT)
        act_values = np.full((num_timesteps, num_agents, num_agents), np.inf)  # 初始化为无穷大
        act_min = np.full((num_timesteps, num_agents), np.inf)  # 每个车辆在每个时间步的最小 ACT
        act_critical_threshold = 2.0  # 临界 ACT 阈值（秒）
        act_critical_events = np.zeros((num_timesteps, num_agents), dtype=bool)  # 临界 ACT 事件
        
        # 计算 2D Time-to-collision (2DTTC)
        ttc2d_values = np.full((num_timesteps, num_agents, num_agents), np.inf)  # 初始化为无穷大
        ttc2d_min = np.full((num_timesteps, num_agents), np.inf)  # 每个车辆在每个时间步的最小 2DTTC
        ttc2d_critical_threshold = 3.0  # 临界 2DTTC 阈值（秒）
        ttc2d_critical_events = np.zeros((num_timesteps, num_agents), dtype=bool)  # 临界 2DTTC 事件
        
        vehicle_length = VEHICLE_LENGTH
        vehicle_width = VEHICLE_WIDTH
        
        for t in range(num_timesteps):
            for i in range(num_agents):
                pos_i = positions[t, i]
                vel_i = velocities[t, i]
                yaw_i = yaws[t, i]
                heading_i = np.array([np.cos(yaw_i), np.sin(yaw_i)])
                
                # 计算车辆i的边界框
                bbox_i = calculate_vehicle_bbox(pos_i, yaw_i, vehicle_length, vehicle_width)
                
                for j in range(num_agents):
                    if i == j:
                        continue
                    
                    pos_j = positions[t, j]
                    vel_j = velocities[t, j]
                    yaw_j = yaws[t, j]
                    heading_j = np.array([np.cos(yaw_j), np.sin(yaw_j)])
                    
                    # 计算车辆j的边界框
                    bbox_j = calculate_vehicle_bbox(pos_j, yaw_j, vehicle_length, vehicle_width)
                    
                    # 计算 TTC
                    ttc = self.calculate_ttc(pos_i, pos_j, vel_i, vel_j, bbox_i, bbox_j)
                    ttc_values[t, i, j] = ttc
                    
                    # 计算 Time Advantage (TAdv)
                    tadv = self.calculate_tadv(pos_i, pos_j, vel_i, vel_j, bbox_i, bbox_j, heading_i, heading_j)
                    tadv_values[t, i, j] = tadv
                    
                    # 计算 Post Encroachment Time (PET)
                    if t > 0:  # 需要前一个时间步的数据
                        pet = self.calculate_pet(
                            positions[t-1:t+1, i], positions[t-1:t+1, j],
                            velocities[t-1:t+1, i], velocities[t-1:t+1, j],
                            yaws[t-1:t+1, i], yaws[t-1:t+1, j],
                            vehicle_length, vehicle_width
                        )
                        pet_values[t, i, j] = pet
                    
                    # 计算 Aggregated Collision Time (ACT)
                    act = self.calculate_act(pos_i, pos_j, vel_i, vel_j, bbox_i, bbox_j)
                    act_values[t, i, j] = act
                    
                    # 计算 2D Time-to-collision (2DTTC)
                    ttc2d = self.calculate_ttc2d(pos_i, pos_j, vel_i, vel_j, bbox_i, bbox_j)
                    ttc2d_values[t, i, j] = ttc2d
                    
                    # 更新最小值
                    if ttc < ttc_min[t, i]:
                        ttc_min[t, i] = ttc
                    if tadv < tadv_min[t, i]:
                        tadv_min[t, i] = tadv
                    if t > 0 and pet_values[t, i, j] < pet_min[t, i]:
                        pet_min[t, i] = pet_values[t, i, j]
                    if act < act_min[t, i]:
                        act_min[t, i] = act
                    if ttc2d < ttc2d_min[t, i]:
                        ttc2d_min[t, i] = ttc2d
                
                # 检查是否为临界事件
                if ttc_min[t, i] < ttc_critical_threshold and ttc_min[t, i] > 0:
                    ttc_critical_events[t, i] = True
                if tadv_min[t, i] < tadv_critical_threshold and tadv_min[t, i] > 0:
                    tadv_critical_events[t, i] = True
                if t > 0 and pet_min[t, i] < pet_critical_threshold and pet_min[t, i] > 0:
                    pet_critical_events[t, i] = True
                if act_min[t, i] < act_critical_threshold and act_min[t, i] > 0:
                    act_critical_events[t, i] = True
                if ttc2d_min[t, i] < ttc2d_critical_threshold and ttc2d_min[t, i] > 0:
                    ttc2d_critical_events[t, i] = True
        
        # 汇总安全指标
        safety_metrics = {
            "positions": positions,
            "velocities": velocities,
            "yaws": yaws,
            "accelerations": accelerations,
            "accelerations_amplitude": accelerations_amplitude,
            "longitudinal_acc": longitudinal_acc,
            "lateral_acc": lateral_acc,
            "longitudinal_jerk": longitudinal_jerk,
            "lateral_jerk": lateral_jerk,
            "angular_velocities": angular_velocities,
            "min_distances": min_distances,
            "hard_braking_events": hard_braking_events,
            "hard_acceleration_events": hard_acceleration_events,
            "hard_turning_events": hard_turning_events,
            "dangerous_proximity_events": dangerous_proximity_events,
            "risky_longitudinal_jerk_events": risky_longitudinal_jerk_events,
            "risky_lateral_jerk_events": risky_lateral_jerk_events,
            "ttc_values": ttc_values,
            "ttc_min": ttc_min,
            "ttc_critical_events": ttc_critical_events,
            "tadv_values": tadv_values,
            "tadv_min": tadv_min,
            "tadv_critical_events": tadv_critical_events,
            "pet_values": pet_values,
            "pet_min": pet_min,
            "pet_critical_events": pet_critical_events,
            "act_values": act_values,
            "act_min": act_min,
            "act_critical_events": act_critical_events,
            "ttc2d_values": ttc2d_values,
            "ttc2d_min": ttc2d_min,
            "ttc2d_critical_events": ttc2d_critical_events,
            "num_agents": num_agents,
            "num_timesteps": num_timesteps
        }
        
        return safety_metrics
    
    def calculate_ttc(self, pos_i, pos_j, vel_vec_i, vel_vec_j, bbox_i, bbox_j):
        """
        计算两车之间的 Time-to-collision (TTC)
        
        Args:
            pos_i: 车辆i的位置
            pos_j: 车辆j的位置
            vel_vec_i: 车辆i的速度向量
            vel_vec_j: 车辆j的速度向量
            bbox_i: 车辆i的边界框
            bbox_j: 车辆j的边界框
            
        Returns:
            TTC值（秒），如果不会碰撞则返回无穷大
        """
        # 检查当前是否已经碰撞
        if check_collision(bbox_i, bbox_j):
            return 0.0
        
        # 相对速度向量
        rel_vel_vec = vel_vec_i - vel_vec_j
        rel_speed = np.linalg.norm(rel_vel_vec)
        
        if rel_speed < 0.1:
            return np.inf
        
        # 使用固定间隔搜索来找到碰撞时间
        max_time = 30.0
        time_step = 0.1
        
        for t in np.arange(0.0, max_time, time_step):
            future_pos_i = pos_i + vel_vec_i * t
            future_pos_j = pos_j + vel_vec_j * t
            
            future_bbox_i = [corner + vel_vec_i * t for corner in bbox_i]
            future_bbox_j = [corner + vel_vec_j * t for corner in bbox_j]
            
            if check_collision(future_bbox_i, future_bbox_j):
                return t
        
        return np.inf
    
    def calculate_tadv(self, pos_i, pos_j, vel_i, vel_j, bbox_i, bbox_j, heading_i, heading_j):
        """
        计算两车之间的 Time Advantage (TAdv)
        
        Args:
            pos_i: 车辆i的位置
            pos_j: 车辆j的位置
            vel_i: 车辆i的速度向量
            vel_j: 车辆j的速度向量
            bbox_i: 车辆i的边界框
            bbox_j: 车辆j的边界框
            heading_i: 车辆i的朝向向量
            heading_j: 车辆j的朝向向量
            
        Returns:
            TAdv值（秒），如果不会碰撞则返回无穷大
        """
        # 检查当前是否已经碰撞
        if check_collision(bbox_i, bbox_j):
            return 0.0
        
        speed_i = np.linalg.norm(vel_i)
        speed_j = np.linalg.norm(vel_j)
        
        if speed_i < 0.1 and speed_j < 0.1:
            return np.inf
        
        rel_pos = pos_j - pos_i
        rel_vel = vel_j - vel_i
        rel_speed = np.linalg.norm(rel_vel)
        
        if speed_i > 0.1 and speed_j > 0.1:
            vel_i_norm = vel_i / speed_i
            vel_j_norm = vel_j / speed_j
            angle_ij = np.arccos(np.clip(np.dot(vel_i_norm, vel_j_norm), -1.0, 1.0))
        else:
            # If one vehicle is stationary, use heading instead of velocity
            if speed_i < 0.1:
                vel_i_norm = heading_i.flatten()
            else:
                vel_i_norm = vel_i / speed_i
            
            if speed_j < 0.1:
                vel_j_norm = heading_j.flatten()
            else:
                vel_j_norm = vel_j / speed_j
            
            angle_ij = np.arccos(np.clip(np.dot(vel_i_norm, vel_j_norm), -1.0, 1.0))
        
        # If the two vehicles are parallel (or nearly parallel), use TTC as TAdv
        if abs(angle_ij) < np.pi/60 or abs(angle_ij) > np.pi*59/60:
            # In parallel cases, calculate TTC
            # If the relative speed is close to 0, then no collision will occur
            if rel_speed < 0.1:
                return np.inf
            
            current_distance = np.linalg.norm(rel_pos)
            ttc = current_distance / rel_speed
            
            # If TTC < 0, the two vehicles are moving away from each other, set to infinity
            if ttc < 0:
                return np.inf
            
            return ttc
        
        # Non-parallel cases, calculate intersection point and time to reach intersection point
        tadv_min = np.inf
        
        for corner_i in bbox_i:
            for corner_j in bbox_j:
                if speed_i > 0.1:
                    line_i = np.array([corner_i, corner_i + vel_i])
                else:
                    line_i = np.array([corner_i, corner_i + heading_i.flatten()])
                
                if speed_j > 0.1:
                    line_j = np.array([corner_j, corner_j + vel_j])
                else:
                    line_j = np.array([corner_j, corner_j + heading_j.flatten()])
                
                # Calculate the intersection point of the two lines
                intersection = self.line_intersection(line_i[0], line_i[1], line_j[0], line_j[1])
                
                if intersection is not None:
                    dist_i = np.linalg.norm(intersection - corner_i)
                    dist_j = np.linalg.norm(intersection - corner_j)
                    
                    # Calculate the time to reach the intersection point
                    time_i = dist_i / max(speed_i, 0.1)
                    time_j = dist_j / max(speed_j, 0.1)
                    
                    # Calculate the time difference
                    time_diff = abs(time_i - time_j)
                    
                    # Check if the intersection point is in front of the two vehicles
                    vec_i_to_int = intersection - corner_i
                    vec_j_to_int = intersection - corner_j
                    
                    ahead_i = np.dot(vec_i_to_int, vel_i_norm if speed_i > 0.1 else heading_i.flatten()) > 0
                    ahead_j = np.dot(vec_j_to_int, vel_j_norm if speed_j > 0.1 else heading_j.flatten()) > 0
                    
                    if ahead_i and ahead_j:
                        tadv_min = min(tadv_min, time_diff)
        
        return tadv_min
    
    def calculate_pet(self, pos_i, pos_j, vel_i, vel_j, yaw_i, yaw_j, vehicle_length, vehicle_width):
        """
        计算两车之间的 Post Encroachment Time (PET)
        
        Args:
            pos_i: 车辆i的位置序列 [t-1, t]
            pos_j: 车辆j的位置序列 [t-1, t]
            vel_i: 车辆i的速度向量序列 [t-1, t]
            vel_j: 车辆j的速度向量序列 [t-1, t]
            yaw_i: 车辆i的朝向序列 [t-1, t]
            yaw_j: 车辆j的朝向序列 [t-1, t]
            vehicle_length: 车辆长度
            vehicle_width: 车辆宽度
            
        Returns:
            PET值（秒），如果不会交叉则返回无穷大
        """
        # Check if there is enough data
        if len(pos_i) < 2 or len(pos_j) < 2:
            return np.inf
        
        # Calculate the bounding box of the current and previous time step
        bbox_i_prev = calculate_vehicle_bbox(pos_i[0], yaw_i[0], vehicle_length, vehicle_width)
        bbox_j_prev = calculate_vehicle_bbox(pos_j[0], yaw_j[0], vehicle_length, vehicle_width)
        bbox_i_curr = calculate_vehicle_bbox(pos_i[1], yaw_i[1], vehicle_length, vehicle_width)
        bbox_j_curr = calculate_vehicle_bbox(pos_j[1], yaw_j[1], vehicle_length, vehicle_width)
        
        # Check if there is a collision
        collision_prev = check_collision(bbox_i_prev, bbox_j_prev)
        collision_curr = check_collision(bbox_i_curr, bbox_j_curr)
        
        # If the current or previous time step has already collided, PET is 0
        if collision_prev or collision_curr:
            return 0.0
        
        # Calculate the intersection point of the two trajectories
        traj_i = np.array([pos_i[0], pos_i[1]])
        traj_j = np.array([pos_j[0], pos_j[1]])
        intersection = self.line_intersection(traj_i[0], traj_i[1], traj_j[0], traj_j[1])
        
        if intersection is None:
            return np.inf
        
        dist_i_prev = np.linalg.norm(intersection - pos_i[0])
        dist_i_curr = np.linalg.norm(intersection - pos_i[1])
        dist_j_prev = np.linalg.norm(intersection - pos_j[0])
        dist_j_curr = np.linalg.norm(intersection - pos_j[1])
        
        # Check if the vehicles have passed the intersection point
        i_passed = dist_i_prev < dist_i_curr
        j_passed = dist_j_prev < dist_j_curr
        
        if i_passed or j_passed:
            return np.inf
        
        speed_i = np.linalg.norm(vel_i[1])
        speed_j = np.linalg.norm(vel_j[1])
        
        # If the speed is too small, consider the vehicle stationary
        if speed_i < 0.1 and speed_j < 0.1:
            return np.inf
        
        # Estimate the time for the vehicles to reach the intersection point
        time_i = dist_i_curr / max(speed_i, 0.1) if not i_passed else 0
        time_j = dist_j_curr / max(speed_j, 0.1) if not j_passed else 0
        
        pet = abs(time_i - time_j)
        
        return pet
    
    def calculate_act(self, pos_i, pos_j, vel_i, vel_j, bbox_i, bbox_j):
        """
        计算两车之间的 Aggregated Collision Time (ACT)
        
        Args:
            pos_i: 车辆i的位置
            pos_j: 车辆j的位置
            vel_i: 车辆i的速度向量
            vel_j: 车辆j的速度向量
            bbox_i: 车辆i的边界框
            bbox_j: 车辆j的边界框
            
        Returns:
            ACT值（秒），如果不会碰撞则返回无穷大
        """
        # 检查当前是否已经碰撞
        if check_collision(bbox_i, bbox_j):
            return 0.0
        
        # 计算速度大小
        speed_i = np.linalg.norm(vel_i)
        speed_j = np.linalg.norm(vel_j)
        
        # 如果两车都静止，则不会发生碰撞
        if speed_i < 0.1 and speed_j < 0.1:
            return np.inf
        
        # 找到两个边界框之间的最短距离和对应的点
        min_dist = np.inf
        closest_point_i = None
        closest_point_j = None
        
        # 对车辆i的每个角点
        for corner_i in bbox_i:
            # 计算到车辆j每个边的最短距离
            for j in range(4):
                edge_start = bbox_j[j]
                edge_end = bbox_j[(j+1)%4]
                
                # 计算点到线段的最短距离
                edge_vec = edge_end - edge_start
                edge_len = np.linalg.norm(edge_vec)
                edge_unit = edge_vec / edge_len if edge_len > 0 else np.zeros(2)
                
                # 计算从边起点到角点的向量
                to_corner = corner_i - edge_start
                
                # 计算投影长度
                proj_len = np.dot(to_corner, edge_unit)
                
                # 找到线段上最近的点
                if proj_len < 0:
                    closest_on_edge = edge_start
                elif proj_len > edge_len:
                    closest_on_edge = edge_end
                else:
                    closest_on_edge = edge_start + edge_unit * proj_len
                
                # 计算距离
                dist = np.linalg.norm(corner_i - closest_on_edge)
                
                if dist < min_dist:
                    min_dist = dist
                    closest_point_i = corner_i
                    closest_point_j = closest_on_edge
        
        # 对车辆j的每个角点重复上述过程
        for corner_j in bbox_j:
            # 计算到车辆i每个边的最短距离
            for i in range(4):
                edge_start = bbox_i[i]
                edge_end = bbox_i[(i+1)%4]
                
                # 计算点到线段的最短距离
                edge_vec = edge_end - edge_start
                edge_len = np.linalg.norm(edge_vec)
                edge_unit = edge_vec / edge_len if edge_len > 0 else np.zeros(2)
                
                # 计算从边起点到角点的向量
                to_corner = corner_j - edge_start
                
                # 计算投影长度
                proj_len = np.dot(to_corner, edge_unit)
                
                # 找到线段上最近的点
                if proj_len < 0:
                    closest_on_edge = edge_start
                elif proj_len > edge_len:
                    closest_on_edge = edge_end
                else:
                    closest_on_edge = edge_start + edge_unit * proj_len
                
                # 计算距离
                dist = np.linalg.norm(corner_j - closest_on_edge)
                
                if dist < min_dist:
                    min_dist = dist
                    closest_point_i = closest_on_edge
                    closest_point_j = corner_j
        
        # 如果没有找到最近点（理论上不应该发生）
        if closest_point_i is None or closest_point_j is None:
            return np.inf
        
        # 计算从i到j和从j到i的方向向量
        dir_i_to_j = closest_point_j - closest_point_i
        dir_j_to_i = closest_point_i - closest_point_j
        
        # 单位化方向向量
        dist_ij = np.linalg.norm(dir_i_to_j)
        if dist_ij > 0:
            dir_i_to_j = dir_i_to_j / dist_ij
            dir_j_to_i = dir_j_to_i / dist_ij
        else:
            return 0.0  # 如果距离为0，则已经碰撞
        
        # 计算速度在最短距离方向上的投影
        v_ij = np.dot(vel_i, dir_i_to_j)
        v_ji = np.dot(vel_j, dir_j_to_i)
        
        # 计算相对接近速度
        p_delta_pt = v_ij + v_ji
        
        # 避免除以接近于0的值
        if abs(p_delta_pt) < 0.1:
            return np.inf
        
        # 计算ACT
        act = min_dist / p_delta_pt
        
        # 如果ACT为负值，表示车辆正在远离，设置为无穷大
        if act < 0:
            return np.inf
        
        return act
    
    def calculate_ttc2d(self, pos_i, pos_j, vel_i, vel_j, bbox_i, bbox_j):
        """
        计算两车之间的 2D Time-to-collision (TTC2D)
        
        Args:
            pos_i: 车辆i的位置
            pos_j: 车辆j的位置
            vel_i: 车辆i的速度向量
            vel_j: 车辆j的速度向量
            bbox_i: 车辆i的边界框
            bbox_j: 车辆j的边界框
            
        Returns:
            TTC2D值（秒），如果不会碰撞则返回无穷大
        """
        # 检查当前是否已经碰撞
        if check_collision(bbox_i, bbox_j):
            return 0.0
        
        # 计算速度大小
        speed_i = np.linalg.norm(vel_i)
        speed_j = np.linalg.norm(vel_j)
        
        # 如果两车都静止，则不会发生碰撞
        if speed_i < 0.1 and speed_j < 0.1:
            return np.inf
        
        # 找到两个边界框之间的最短距离和对应的点
        min_dist = np.inf
        closest_point_i = None
        closest_point_j = None
        
        # 对车辆i的每个角点，计算到车辆j每个边的最短距离
        for corner_i in bbox_i:
            for j in range(4):
                edge_start = bbox_j[j]
                edge_end = bbox_j[(j+1)%4]
                
                # 计算点到线段的最短距离
                edge_vec = edge_end - edge_start
                edge_len = np.linalg.norm(edge_vec)
                edge_unit = edge_vec / edge_len if edge_len > 0 else np.zeros(2)
                
                # 计算从边起点到角点的向量
                to_corner = corner_i - edge_start
                
                # 计算投影长度
                proj_len = np.dot(to_corner, edge_unit)
                
                # 找到线段上最近的点
                if proj_len < 0:
                    closest_on_edge = edge_start
                elif proj_len > edge_len:
                    closest_on_edge = edge_end
                else:
                    closest_on_edge = edge_start + edge_unit * proj_len
                
                # 计算距离
                dist = np.linalg.norm(corner_i - closest_on_edge)
                
                if dist < min_dist:
                    min_dist = dist
                    closest_point_i = corner_i
                    closest_point_j = closest_on_edge
        
        # 对车辆j的每个角点，计算到车辆i每个边的最短距离
        for corner_j in bbox_j:
            for i in range(4):
                edge_start = bbox_i[i]
                edge_end = bbox_i[(i+1)%4]
                
                # 计算点到线段的最短距离
                edge_vec = edge_end - edge_start
                edge_len = np.linalg.norm(edge_vec)
                edge_unit = edge_vec / edge_len if edge_len > 0 else np.zeros(2)
                
                # 计算从边起点到角点的向量
                to_corner = corner_j - edge_start
                
                # 计算投影长度
                proj_len = np.dot(to_corner, edge_unit)
                
                # 找到线段上最近的点
                if proj_len < 0:
                    closest_on_edge = edge_start
                elif proj_len > edge_len:
                    closest_on_edge = edge_end
                else:
                    closest_on_edge = edge_start + edge_unit * proj_len
                
                # 计算距离
                dist = np.linalg.norm(corner_j - closest_on_edge)
                
                if dist < min_dist:
                    min_dist = dist
                    closest_point_i = closest_on_edge
                    closest_point_j = corner_j
        
        # 如果没有找到最近点（理论上不应该发生）
        if closest_point_i is None or closest_point_j is None:
            return np.inf
        
        # 计算从i到j和从j到i的方向向量
        dir_i_to_j = closest_point_j - closest_point_i
        dir_j_to_i = closest_point_i - closest_point_j
        
        # 单位化方向向量
        dist_ij = np.linalg.norm(dir_i_to_j)
        if dist_ij > 0:
            dir_i_to_j = dir_i_to_j / dist_ij
            dir_j_to_i = dir_j_to_i / dist_ij
        else:
            return 0.0  # 如果距离为0，则已经碰撞
        
        # 计算速度在最短距离方向上的投影
        v_ij = np.dot(vel_i, dir_i_to_j)
        v_ji = np.dot(vel_j, dir_j_to_i)
        
        # 计算相对接近速度
        rel_approach_speed = v_ij + v_ji
        
        # 如果相对接近速度小于等于0，车辆不会碰撞
        if rel_approach_speed <= 0.1:
            return np.inf
        
        # 计算2DTTC
        ttc2d = min_dist / rel_approach_speed
        
        return ttc2d
    
    def visualize_safety_metrics(self, safety_metrics, save_path):
        """
        可视化安全指标
        
        Args:
            safety_metrics: 安全指标字典
            save_path: 保存路径
        """
        # 确保保存目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 提取指标
        positions = safety_metrics["positions"]
        velocities = safety_metrics["velocities"]
        accelerations = safety_metrics["accelerations"]
        accelerations_amplitude = safety_metrics["accelerations_amplitude"]
        longitudinal_acc = safety_metrics["longitudinal_acc"]
        lateral_acc = safety_metrics["lateral_acc"]
        longitudinal_jerk = safety_metrics["longitudinal_jerk"]
        # jerks = safety_metrics.get("jerks", None)
        longitudinal_jerk = safety_metrics.get("longitudinal_jerk", None)
        lateral_jerk = safety_metrics.get("lateral_jerk", None)
        min_distances = safety_metrics["min_distances"]
        hard_braking_events = safety_metrics["hard_braking_events"]
        hard_acceleration_events = safety_metrics["hard_acceleration_events"]
        hard_turning_events = safety_metrics["hard_turning_events"]
        dangerous_proximity_events = safety_metrics["dangerous_proximity_events"]
        risky_jerk_events = safety_metrics["risky_jerk_events"]
        ttc_values = safety_metrics["ttc_values"]
        ttc_min = safety_metrics["ttc_min"]
        ttc_critical_events = safety_metrics["ttc_critical_events"]
        tadv_values = safety_metrics["tadv_values"]
        tadv_min = safety_metrics["tadv_min"]
        tadv_critical_events = safety_metrics["tadv_critical_events"]
        pet_values = safety_metrics["pet_values"]
        pet_min = safety_metrics["pet_min"]
        pet_critical_events = safety_metrics["pet_critical_events"]
        act_values = safety_metrics["act_values"]
        act_min = safety_metrics["act_min"]
        act_critical_events = safety_metrics["act_critical_events"]
        ttc2d_values = safety_metrics["ttc2d_values"]
        ttc2d_min = safety_metrics["ttc2d_min"]
        ttc2d_critical_events = safety_metrics["ttc2d_critical_events"]
        num_agents = safety_metrics["num_agents"]
        num_timesteps = safety_metrics["num_timesteps"]
        
        # 创建时间轴
        time_steps = np.arange(num_timesteps)
        time_steps_acc = np.arange(num_timesteps-1)
        time_steps_jerk = np.arange(num_timesteps-2) if longitudinal_jerk is not None else None
        
        # 为每个代理创建单独的图表
        for agent_idx in range(num_agents):
            agent_name = "ego" if agent_idx == 0 else f"adv_{agent_idx}"
            
            # 创建多子图 - 增加子图用于纵向和横向加速度
            fig, axs = plt.subplots(10, 1, figsize=(12, 60))
            
            # 1. 速度图
            axs[0].plot(time_steps, np.linalg.norm(velocities[:, agent_idx], axis=1))
            axs[0].set_title(f'{agent_name} velocity')
            axs[0].set_xlabel('time step')
            axs[0].set_ylabel('velocity (m/s)')
            axs[0].grid(True)
            
            # 2. 总加速度图
            axs[1].plot(time_steps_acc, accelerations_amplitude[:, agent_idx])
            axs[1].set_title(f'{agent_name} acceleration magnitude')
            axs[1].set_xlabel('time step')
            axs[1].set_ylabel('acceleration (m/s²)')
            axs[1].grid(True)
            
            # 3. 纵向加速度图
            axs[2].plot(time_steps_acc, longitudinal_acc[:, agent_idx])
            axs[2].set_title(f'{agent_name} longitudinal acceleration')
            axs[2].set_xlabel('time step')
            axs[2].set_ylabel('acceleration (m/s²)')
            axs[2].grid(True)
            
            # 标记急刹车和急加速事件
            if agent_idx < hard_braking_events.shape[1]:
                braking_events = time_steps_acc[hard_braking_events[:, agent_idx]]
                if len(braking_events) > 0:
                    axs[2].scatter(braking_events, 
                                  longitudinal_acc[hard_braking_events[:, agent_idx], agent_idx], 
                                  color='red', label='hard braking')
                
                acceleration_events = time_steps_acc[hard_acceleration_events[:, agent_idx]]
                if len(acceleration_events) > 0:
                    axs[2].scatter(acceleration_events, 
                                  longitudinal_acc[hard_acceleration_events[:, agent_idx], agent_idx], 
                                  color='orange', label='hard acceleration')
                
                if len(braking_events) > 0 or len(acceleration_events) > 0:
                    axs[2].legend()
            
            # 添加舒适阈值线
            axs[2].axhline(y=3.0, color='r', linestyle='--', alpha=0.5)
            axs[2].axhline(y=-3.0, color='r', linestyle='--', alpha=0.5)
            axs[2].text(0, 3.5, 'comfort threshold (±3 m/s²)', color='r', alpha=0.7)
            
            # 4. 横向加速度图
            axs[3].plot(time_steps_acc, lateral_acc[:, agent_idx])
            axs[3].set_title(f'{agent_name} lateral acceleration')
            axs[3].set_xlabel('time step')
            axs[3].set_ylabel('acceleration (m/s²)')
            axs[3].grid(True)
            
            # 标记急转弯事件
            if agent_idx < hard_turning_events.shape[1]:
                turning_events = time_steps_acc[hard_turning_events[:, agent_idx]]
                if len(turning_events) > 0:
                    axs[3].scatter(turning_events, 
                                  lateral_acc[hard_turning_events[:, agent_idx], agent_idx], 
                                  color='red', label='hard turning')
                    axs[3].legend()
            
            axs[3].axhline(y=2.0, color='r', linestyle='--', alpha=0.5)
            axs[3].axhline(y=-2.0, color='r', linestyle='--', alpha=0.5)
            axs[3].text(0, 2.5, 'comfort threshold (±2 m/s²)', color='r', alpha=0.7)
            
            # 5. 加加速度图
            if longitudinal_jerk is not None and time_steps_jerk is not None:                
                # 纵向加加速度
                axs[4].plot(time_steps_jerk, longitudinal_jerk[:, agent_idx])
                axs[4].set_title(f'{agent_name} longitudinal jerk')
                axs[4].set_xlabel('time step')
                axs[4].set_ylabel('jerk (m/s³)')
                axs[4].grid(True)
                axs[4].axhline(y=LONJ_THRESHOLD, color='r', linestyle='--', alpha=0.5)
                axs[4].axhline(y=-LONJ_THRESHOLD, color='r', linestyle='--', alpha=0.5)
                axs[4].text(0, LONJ_THRESHOLD + 0.5, f'comfort threshold ({LONJ_THRESHOLD} m/s³)', color='r', alpha=0.7)
                
                # 横向加加速度
                axs[5].plot(time_steps_jerk, lateral_jerk[:, agent_idx])
                axs[5].set_title(f'{agent_name} lateral jerk')
                axs[5].set_xlabel('time step')
                axs[5].set_ylabel('jerk (m/s³)')
                axs[5].grid(True)
                axs[5].axhline(y=LATJ_THRESHOLD, color='r', linestyle='--', alpha=0.5)
                axs[5].axhline(y=-LATJ_THRESHOLD, color='r', linestyle='--', alpha=0.5)
                axs[5].text(0, LATJ_THRESHOLD + 0.5, f'comfort threshold ({LATJ_THRESHOLD} m/s³)', color='r', alpha=0.7)
                
            # 6. 最小距离图
            axs[6].plot(time_steps, min_distances[:, agent_idx])
            axs[6].set_title(f'{agent_name} minimum distance to other vehicles')
            axs[6].set_xlabel('time step')
            axs[6].set_ylabel('distance (m)')
            axs[6].grid(True)
            
            # 标记危险接近事件
            if agent_idx < dangerous_proximity_events.shape[1]:
                proximity_events = time_steps[dangerous_proximity_events[:, agent_idx]]
                if len(proximity_events) > 0:
                    axs[6].scatter(proximity_events, 
                                  min_distances[dangerous_proximity_events[:, agent_idx], agent_idx], 
                                  color='red', label='dangerous proximity')
                    axs[6].legend()
            
            # 7-10. 其他安全指标图
            axs[7].plot(time_steps, ttc_values[:, agent_idx])
            axs[7].set_title(f'{agent_name} TTC')
            axs[7].set_xlabel('time step')
            axs[7].set_ylabel('TTC (s)')
            axs[7].grid(True)

            axs[8].plot(time_steps, tadv_values[:, agent_idx])
            axs[8].set_title(f'{agent_name} TADV')
            axs[8].set_xlabel('time step')
            axs[8].set_ylabel('TADV (s)')
            axs[8].grid(True)

            axs[9].plot(time_steps, pet_values[:, agent_idx])
            axs[9].set_title(f'{agent_name} PET')
            axs[9].set_xlabel('time step')
            axs[9].set_ylabel('PET (s)')
            axs[9].grid(True)

            axs[10].plot(time_steps, act_values[:, agent_idx])
            axs[10].set_title(f'{agent_name} ACT')
            axs[10].set_xlabel('time step')
            axs[10].set_ylabel('ACT (s)')
            axs[10].grid(True)
            
            axs[11].plot(time_steps, ttc2d_values[:, agent_idx])
            axs[11].set_title(f'{agent_name} 2DTTC')
            axs[11].set_xlabel('time step')
            axs[11].set_ylabel('2DTTC (s)')
            axs[11].grid(True)                 

            # 调整布局并保存
            plt.tight_layout()
            plt.savefig(f"{save_path}_{agent_name}_metrics.png")
            plt.close()
            
            # 创建加速度详细分析图
            self.visualize_acceleration_analysis(
                time_steps_acc, 
                time_steps_jerk, 
                accelerations_amplitude[:, agent_idx],
                longitudinal_acc[:, agent_idx],
                lateral_acc[:, agent_idx],
                longitudinal_jerk[:, agent_idx] if longitudinal_jerk is not None else None,
                lateral_jerk[:, agent_idx] if lateral_jerk is not None else None,
                hard_braking_events[:, agent_idx] if agent_idx < hard_braking_events.shape[1] else None,
                hard_acceleration_events[:, agent_idx] if agent_idx < hard_acceleration_events.shape[1] else None,
                hard_turning_events[:, agent_idx] if agent_idx < hard_turning_events.shape[1] else None,
                agent_name,
                f"{save_path}_{agent_name}_acceleration_analysis.png"
            )
    
    def create_trajectory_animation(self, safety_metrics, save_path):
        """
        创建轨迹动画
        
        Args:
            safety_metrics: 安全指标字典
            save_path: 保存路径
        """
        positions = safety_metrics["positions"]
        num_agents = safety_metrics["num_agents"]
        num_timesteps = safety_metrics["num_timesteps"]
        
        # 计算位置范围以设置固定的坐标轴范围
        x_min, x_max = np.min(positions[:, :, 0]), np.max(positions[:, :, 0])
        y_min, y_max = np.min(positions[:, :, 1]), np.max(positions[:, :, 1])
        
        # 添加一些边距
        margin = max(x_max - x_min, y_max - y_min) * 0.1
        x_min -= margin
        x_max += margin
        y_min -= margin
        y_max += margin
        
        # 创建图形和轴
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_title('vehicle trajectory animation')
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        
        # 为每个代理创建散点对象
        scatters = []
        for i in range(num_agents):
            color = 'red' if i == 0 else 'blue'  # ego为红色，其他为蓝色
            label = 'ego' if i == 0 else f'adv_{i}'
            scatter = ax.scatter([], [], s=100, color=color, label=label)
            scatters.append(scatter)
        
        # 添加轨迹线
        lines = []
        for i in range(num_agents):
            color = 'red' if i == 0 else 'blue'
            line, = ax.plot([], [], color=color, alpha=0.3)
            lines.append(line)
        
        # 添加时间步显示
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        
        # 初始化函数
        def init():
            for scatter in scatters:
                scatter.set_offsets(np.empty((0, 2)))
            for line in lines:
                line.set_data([], [])
            time_text.set_text('')
            return scatters + lines + [time_text]
        
        # 更新函数
        def update(frame):
            for i, (scatter, line) in enumerate(zip(scatters, lines)):
                scatter.set_offsets(positions[frame, i].reshape(1, 2))
                line.set_data(positions[:frame+1, i, 0], positions[:frame+1, i, 1])
            time_text.set_text(f'时间步: {frame}')
            return scatters + lines + [time_text]
        
        # 创建动画
        ani = FuncAnimation(fig, update, frames=range(num_timesteps),
                            init_func=init, blit=True, interval=100)
        
        # 添加图例
        ax.legend()
        
        # 保存动画
        ani.save(f"{save_path}_trajectory.gif", writer='pillow', fps=10)
        plt.close()
    
    def save_safety_metrics(self, safety_metrics, save_path):
        """
        保存安全指标数据
        
        Args:
            safety_metrics: 安全指标字典
            save_path: 保存路径
        """
        # 创建可序列化的指标字典
        serializable_metrics = {}
        
        # 转换NumPy数组为列表
        for key, value in safety_metrics.items():
            if isinstance(value, np.ndarray):
                serializable_metrics[key] = value.tolist()
            else:
                serializable_metrics[key] = value
        
        # 保存为JSON文件
        with open(f"{save_path}_safety_metrics.json", 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
    
    def fetch_max_iter(self):
        """
        从timings.json文件中获取最大迭代次数
        
        Returns:
            包含每个密度和城市最大迭代次数的字典
        """
        # 读取timings字典
        with open('tools/timings.json') as f:
            timings = json.load(f)

        max_iter = {}
        max_GPU_seconds = 180
        for density in ["1", "2", "4"]:
            timings_per_town_per_density = timings[self.args.optim_method][density]
            max_iter[density] = {} 
            for town, timing in timings_per_town_per_density.items():
                max_iter[density][str(town)] = math.floor(max_GPU_seconds / timing)
        return max_iter
    
    def parse_scenario_log_dir(self):
        """
        解析场景日志目录并收集JSON文件路径
        
        Returns:
            记录文件路径和结果文件路径的元组
        """
        route_scenario_dirs = sorted(
            glob.glob(
                self.args.scenario_log_dir + "/**/RouteScenario*/", recursive=True
            ),
            key=lambda path: (path.split("_")[-6]),
        )

        # 收集所有记录和结果JSON文件
        results_files = []
        records_files = []
        for dir in route_scenario_dirs:
            results_files.extend(
                sorted(
                    glob.glob(dir + "results.json")
                )
            )
            records_files.extend(
                sorted(
                    glob.glob(dir + "scenario_records.json")
                )
            )
        return records_files, results_files

    def parse_json_file(self, records_file):
        """
        解析JSON文件
        
        Args:
            records_file: JSON文件路径
            
        Returns:
            解析后的JSON数据
        """
        return json.loads(open(records_file).read())

    def line_intersection(self, p1, p2, p3, p4):
        """
        计算两条线段的交点
        
        Args:
            p1, p2: 第一条线段的端点
            p3, p4: 第二条线段的端点
            
        Returns:
            交点坐标，如果没有交点则返回None
        """
        # 线段表示：p1 + t1(p2-p1) = p3 + t2(p4-p3)
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        # 计算分母
        denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
        
        # 如果分母为0，则线段平行或共线
        if abs(denom) < 1e-8:
            return None
        
        # 计算参数 t1 和 t2
        ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
        ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom
        x = x1 + ua * (x2 - x1)
        y = y1 + ua * (y2 - y1)
        return np.array([x, y])
        
        # # 检查交点是否在两条线段上
        # if 0 <= ua <= 1 and 0 <= ub <= 1:
        #     # 计算交点坐标
        #     x = x1 + ua * (x2 - x1)
        #     y = y1 + ua * (y2 - y1)
        #     return np.array([x, y])
        
        # return None

    def visualize_acceleration_analysis(self, time_steps_acc, time_steps_jerk, 
                                   acc_magnitude, longitudinal_acc, lateral_acc,
                                   longitudinal_jerk, lateral_jerk,
                                   hard_braking_events, hard_acceleration_events, hard_turning_events,
                                   agent_name, save_path):
        """
        创建加速度和加加速度的详细分析图
        
        Args:
            time_steps_acc: 加速度的时间步
            time_steps_jerk: 加加速度的时间步
            acc_magnitude: 加速度幅值
            longitudinal_acc: 纵向加速度
            lateral_acc: 横向加速度
            longitudinal_jerk: 纵向加加速度
            lateral_jerk: 横向加加速度
            hard_braking_events: 急刹车事件
            hard_acceleration_events: 急加速事件
            hard_turning_events: 急转弯事件
            agent_name: 代理名称
            save_path: 保存路径
        """
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 纵向加速度图
        axs[0, 0].plot(time_steps_acc, longitudinal_acc, 'b-', label='Longitudinal')
        axs[0, 0].set_title(f'{agent_name} Longitudinal Acceleration')
        axs[0, 0].set_xlabel('Time step')
        axs[0, 0].set_ylabel('Acceleration (m/s²)')
        axs[0, 0].grid(True)
        
        # 标记急刹车和急加速事件
        if hard_braking_events is not None:
            braking_events = time_steps_acc[hard_braking_events]
            if len(braking_events) > 0:
                axs[0, 0].scatter(braking_events, 
                                longitudinal_acc[hard_braking_events], 
                                color='red', label='Hard braking')
        
        if hard_acceleration_events is not None:
            acceleration_events = time_steps_acc[hard_acceleration_events]
            if len(acceleration_events) > 0:
                axs[0, 0].scatter(acceleration_events, 
                                longitudinal_acc[hard_acceleration_events], 
                                color='orange', label='Hard acceleration')
        
        # 添加舒适阈值线
        axs[0, 0].axhline(y=3.0, color='r', linestyle='--', alpha=0.5)
        axs[0, 0].axhline(y=-3.0, color='r', linestyle='--', alpha=0.5)
        axs[0, 0].text(0, 3.5, 'Comfort threshold (±3 m/s²)', color='r', alpha=0.7)
        axs[0, 0].legend()
        
        # 2. 横向加速度图
        axs[0, 1].plot(time_steps_acc, lateral_acc, 'g-', label='Lateral')
        axs[0, 1].set_title(f'{agent_name} Lateral Acceleration')
        axs[0, 1].set_xlabel('Time step')
        axs[0, 1].set_ylabel('Acceleration (m/s²)')
        axs[0, 1].grid(True)
        
        # 标记急转弯事件
        if hard_turning_events is not None:
            turning_events = time_steps_acc[hard_turning_events]
            if len(turning_events) > 0:
                axs[0, 1].scatter(turning_events, 
                                lateral_acc[hard_turning_events], 
                                color='red', label='Hard turning')
        
        # 添加舒适阈值线
        axs[0, 1].axhline(y=2.0, color='r', linestyle='--', alpha=0.5)
        axs[0, 1].axhline(y=-2.0, color='r', linestyle='--', alpha=0.5)
        axs[0, 1].text(0, 2.5, 'Comfort threshold (±2 m/s²)', color='r', alpha=0.7)
        axs[0, 1].legend()
        
        # 3. 纵向加加速度图
        if longitudinal_jerk is not None and time_steps_jerk is not None:
            axs[1, 0].plot(time_steps_jerk, longitudinal_jerk, 'b-', label='Longitudinal')
            axs[1, 0].set_title(f'{agent_name} Longitudinal Jerk')
            axs[1, 0].set_xlabel('Time step')
            axs[1, 0].set_ylabel('Jerk (m/s³)')
            axs[1, 0].grid(True)
            
            # 添加舒适阈值线
            axs[1, 0].axhline(y=5.0, color='r', linestyle='--', alpha=0.5)
            axs[1, 0].axhline(y=-5.0, color='r', linestyle='--', alpha=0.5)
            axs[1, 0].text(0, 5.5, 'Comfort threshold (±5 m/s³)', color='r', alpha=0.7)
            axs[1, 0].legend()
        
        # 4. 横向加加速度图
        if lateral_jerk is not None and time_steps_jerk is not None:
            axs[1, 1].plot(time_steps_jerk, lateral_jerk, 'g-', label='Lateral')
            axs[1, 1].set_title(f'{agent_name} Lateral Jerk')
            axs[1, 1].set_xlabel('Time step')
            axs[1, 1].set_ylabel('Jerk (m/s³)')
            axs[1, 1].grid(True)
            
            # 添加舒适阈值线
            axs[1, 1].axhline(y=5.0, color='r', linestyle='--', alpha=0.5)
            axs[1, 1].axhline(y=-5.0, color='r', linestyle='--', alpha=0.5)
            axs[1, 1].text(0, 5.5, 'Comfort threshold (±5 m/s³)', color='r', alpha=0.7)
            axs[1, 1].legend()
        
        # 计算统计数据
        lon_acc_mean = np.mean(longitudinal_acc)
        lon_acc_std = np.std(longitudinal_acc)
        lon_acc_max = np.max(longitudinal_acc)
        lon_acc_min = np.min(longitudinal_acc)
        
        lat_acc_mean = np.mean(lateral_acc)
        lat_acc_std = np.std(lateral_acc)
        lat_acc_max = np.max(lateral_acc)
        lat_acc_min = np.min(lateral_acc)
        
        # 计算舒适性指标
        uncomfortable_lon_acc_ratio = np.sum(np.abs(longitudinal_acc) > 3.0) / len(longitudinal_acc) * 100
        uncomfortable_lat_acc_ratio = np.sum(np.abs(lateral_acc) > 2.0) / len(lateral_acc) * 100
        
        # 添加统计信息文本框
        stats_text = (
            f"Longitudinal Acceleration:\n"
            f"  Mean: {lon_acc_mean:.2f} m/s²\n"
            f"  Std Dev: {lon_acc_std:.2f} m/s²\n"
            f"  Max: {lon_acc_max:.2f} m/s²\n"
            f"  Min: {lon_acc_min:.2f} m/s²\n"
            f"  Uncomfortable: {uncomfortable_lon_acc_ratio:.1f}%\n\n"
            f"Lateral Acceleration:\n"
            f"  Mean: {lat_acc_mean:.2f} m/s²\n"
            f"  Std Dev: {lat_acc_std:.2f} m/s²\n"
            f"  Max: {lat_acc_max:.2f} m/s²\n"
            f"  Min: {lat_acc_min:.2f} m/s²\n"
            f"  Uncomfortable: {uncomfortable_lat_acc_ratio:.1f}%"
        )
        
        # 在图表右上角添加文本框
        plt.figtext(0.5, 0.01, stats_text, fontsize=10,
                   bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)  # 为底部文本框留出空间
        plt.savefig(save_path)
        plt.close()


def main(args):
    """
    主函数
    
    Args:
        args: 命令行参数
    """
    analyzer = SafetySurrogateAnalyzer(args)
    analyzer.analyze()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze safety surrogate metrics")
    
    parser.add_argument(
        "--scenario_log_dir",
        type=str,
        default="generation_results",
        help="The directory containing the per-route directories with the "
             "corresponding scenario log .json files.",
    )
    parser.add_argument(
        "--opt_iter",
        type=int,
        default=-1,
        help="Specifies at which iteration in the optimization process the "
             "scenarios should be visualized. Set to -1 to automatically "
             "select the critical perturbation for each scenario.",
    )
    parser.add_argument(
        "--optim_method",
        default="Adam",
        choices=["Adam", "Both_Paths"]
    )
    parser.add_argument(
        "--save_metrics",
        action="store_true"
    )
    parser.add_argument(
        "--create_animation",
        action="store_true"
    )
    
    args = parser.parse_args()
    
    main(args) 
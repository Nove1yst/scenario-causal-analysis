from matplotlib import pyplot as plt
import os
import pickle
import math
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.patches import FancyArrowPatch
from PIL import Image
from scipy.special import comb
import numpy as np
from tqdm import tqdm
import networkx as nx

# def visualize_causal_graph(causal_graph, save_path=None):
#     """
#     Visualize the causal graph.

#     Args:
#         causal_graph: A dictionary representing the causal graph, where the keys are agent IDs and the values are lists of agent IDs they influence.
#         save_path: The path to save the image
#     """
#     G = nx.DiGraph()

#     # 添加边到图中
#     for agent, influenced_agents in causal_graph.items():
#         for influenced_agent, ssm_type, critical_frames in influenced_agents:
#             # 添加带有关键帧信息的边
#             G.add_edge(agent, influenced_agent, 
#                       ssm=ssm_type, 
#                       critical_frames=critical_frames,
#                       label=f"{ssm_type}\n Frame: {critical_frames[:3]}...")

#     # 绘制图形
#     plt.figure(figsize=(12, 10))
#     pos = nx.spring_layout(G, seed=42)  # 所有节点的位置
    
#     # 绘制节点
#     nx.draw_networkx_nodes(G, pos, 
#                           node_color='lightblue', 
#                           node_size=2000)
    
#     # 绘制边
#     nx.draw_networkx_edges(G, pos, 
#                           arrowstyle='->', 
#                           arrowsize=20, 
#                           width=2)
    
#     # 绘制节点标签
#     nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    
#     # 获取边标签并绘制
#     edge_labels = {(u, v): d['label'] for u, v, d in G.edges(data=True)}
#     nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

#     plt.title("Causal Graph")
#     plt.axis('off')
    
#     # 保存图形
#     if save_path:
#         plt.savefig(save_path)
#     plt.close()

def create_mp4_from_scenario(track_info, frame_info, track_id, scene_id, out_path_scene_id, fps=10):
    images = []
    video_path = os.path.join(out_path_scene_id, 'video')
    os.makedirs(video_path, exist_ok=True)

    i = 0
    frame_nums = len(frame_info)
    image_base_path = os.path.join(out_path_scene_id, f'ID_[{track_id}]')
    os.makedirs(image_base_path, exist_ok=True)

    # Video writer object
    video_output_path = os.path.join(video_path, f"ID_[{track_id}]_frames_{frame_nums}.mp4")

    # Initialize video writer (mp4 format, codec is 'mp4v')
    video_writer = None

    start_frame = frame_info[0][0]['vehicle_info']['frame_id']
    end_frame = frame_info[-1][0]['vehicle_info']['frame_id']

    for num in tqdm(frame_info, desc=f"Processing track",leave=True):

        frame_id = num[0]['vehicle_info']['frame_id']

        # Plot and save individual frames as images
        plot_vehicle_positions(track_id, track_info, num, scene_id, frame_id, start_frame, end_frame)
        image_path = os.path.join(image_base_path,
                                  f"ID_{track_id}_{i}_in_{frame_nums}_frame_id_{frame_id}.png")
        plt.savefig(image_path)
        plt.close()

        # Convert saved image to video frame
        img = Image.open(image_path)
        img_array = np.array(img)

        # Initialize the video writer once with the correct frame size (based on first frame)
        if video_writer is None:
            height, width, _ = img_array.shape
            video_writer = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        # Write the frame to the video
        video_writer.write(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))

        i += 1

    # Release the video writer object
    if video_writer is not None:
        video_writer.release()

    # print(f"Saved video: {video_output_path}, with {frame_nums} frames")


def create_gif_from_scenario(track_info, frame_info, track_id, scene_id, out_path_scene_id,gif_number):
    images = []
    gif_path = os.path.join(out_path_scene_id, 'gif')
    os.makedirs(gif_path, exist_ok=True)
    i=0
    frame_nums = len(frame_info)

    image_base_path = os.path.join(out_path_scene_id, f'ID_{track_id}({gif_number})_in_{frame_nums}')
    os.makedirs(image_base_path, exist_ok=True)

    # The first index means which frame
    # The second index means whose track
    start_frame = frame_info[0][0]['vehicle_info']['frame_id']
    end_frame = frame_info[-1][0]['vehicle_info']['frame_id']
    for frame in frame_info:
        frame_id = frame[0]['vehicle_info']['frame_id']
        plot_vehicle_positions(track_id, track_info, frame, scene_id, frame_id, start_frame, end_frame)
        image_path = os.path.join(
            image_base_path, 
            f"ID_{track_id}({gif_number})_{i}_in_{frame_nums}_frame_id_{frame_id}.png"
        )
        plt.savefig(image_path)
        plt.close()
        
        images.append(Image.open(image_path))
        i += 1

    if images:
        len_img=len(images)
        gif_output_path = os.path.join(gif_path, f"ID_{track_id}({gif_number})_frames_{frame_nums}.gif")

        images[0].save(gif_output_path, save_all=True, append_images=images[1:], duration=100, loop=0)
        print(f"Saved GIF to {gif_output_path}")

def plot_vehicle_positions(track_id, track_info, frame_info, scene_id, frame_id, start_frame, end_frame):
    # 定义颜色映射，用于不同类型的车辆
    color_map = {
        'mv': 'blue', 
        'nmv': 'blue', 
        'ped': 'blue', 
        'track_1': 'red', 
        'track_2': 'green', 
        'track_3': 'yellow', 
        'track_4': 'purple', 
        'track_5': 'orange', 
        'track_6': 'red'
    }

    fig, ax = plt.subplots()
    # 绘制网格
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # 添加道路绘制逻辑
    plot_roads(ax)

    motor_vehicles = {'car', 'bus', 'truck'}
    non_motor_vehicles = {'bicycle', 'tricycle', 'motorcycle'}

    # 遍历所有车辆并绘制
    for tp in frame_info:
        # 检查 track_id 的类型，如果是 int，则将其转换为字符串
        if isinstance(track_id, int):
            track_id = [track_id]  # 将 int 转换为列表，这样可以进行迭代

        # 将所有 track_id 和 tp['tp_id'] 转换为字符串进行比较
        if str(tp['tp_id']) in map(str, track_id):
            main_vehicle_state = tp['vehicle_info']
            if tp['tp_id'] in track_info:
                n = track_info[tp['tp_id']]['num']
                if n>5:
                    n=n-5
            else:
                n=6

            draw_vehicle(ax, main_vehicle_state, color_map[f'track_{n}'], tp['tp_id'])
            # 绘制主车的可视区域
            draw_track(tp['tp_id'], track_info, ax, color_map,n)
        else:
            if tp['vehicle_info']['agent_type'] in motor_vehicles:
                draw_vehicle(ax, tp['vehicle_info'], color_map['mv'], tp['tp_id'])
            elif tp['vehicle_info']['agent_type'] in non_motor_vehicles:
                draw_vehicle(ax, tp['vehicle_info'], color_map['nmv'], tp['tp_id'])
            else:
                draw_vehicle(ax, tp['vehicle_info'], color_map['ped'], tp['tp_id'])

    # # 设置坐标轴的显示范围
    # min_x, max_x = -30, 60
    # min_y, max_y = -15, 50
    # ax.set_xlim(min_x, max_x)
    # ax.set_ylim(min_y, max_y)

    # 设置图表标题
    ax.set_title(f' track_id {track_id} frame_id {frame_id} \n frames from {start_frame}to {end_frame} ')

    # 自定义图例
    handles = [
        # plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map['main_vehicle'], markersize=10, label='main_vehicle'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map['mv'], markersize=10, label='mv'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map['nmv'], markersize=10, label='nmv'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map['ped'], markersize=10, label='ped'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=10, label='anomalies'),
    ]
    ax.legend(handles=handles, loc='upper right')
    plt.show()

def draw_track(tp_id, track_info, ax, color_map, n):
    if n != 6:
        # 获取轨迹信息
        track = track_info[tp_id]
    else:
        track = track_info
    x = np.array(track['track_info']['x'])
    y = np.array(track['track_info']['y'])
    ax.scatter(x[track['anomalies'][:len(x)]], y[track['anomalies'][:len(y)]], color='k',
               label='Anomalies', zorder=3, s=0.7)

    ax.scatter(x[0], y[0], color="green", marker='o', label='Start')
    ax.scatter(x[-1], y[-1], color="red", marker='x', label='End')
    ax.scatter(x, y,color=color_map[f'track_{n}'], label='Trajectory', zorder=2, s=0.3)

def draw_vehicle(ax, vehicle, color, label):
    # 获取车辆的中心坐标
    x, y = vehicle['x'], vehicle['y']
    # 获取车辆的代理类型，默认为 'vehicle'
    agent_type = vehicle.get('agent_type', 'vehicle')

    # 获取车辆的速度分量
    vx, vy = vehicle.get('vx', 0), vehicle.get('vy', 0)
    # 获取车辆的朝向角度（弧度）
    heading_rad = vehicle.get('heading_rad', 0)

    # 如果车辆在移动（速度不为0），绘制箭头表示行进方向
    if vx != 0 or vy != 0:
        arrow_length = 2.0
        arrow_start = (x, y)
        arrow_end = (x + arrow_length * math.cos(heading_rad), y + arrow_length * math.sin(heading_rad))
        # 创建箭头图形
        arrow = FancyArrowPatch(arrow_start, arrow_end, color='b', arrowstyle='->',
                                mutation_scale=10, alpha=1)
        # 将箭头添加到坐标轴
        ax.add_patch(arrow)

    # 根据车辆类型绘制不同的图形
    if agent_type == 'pedestrian':
        radius = vehicle.get('radius', 0.5)  # 行人的默认半径
        # 绘制圆形表示行人
        circle = plt.Circle((x, y), radius, color=color, fill=True, alpha=0.5)
        ax.add_patch(circle)
        # 在行人位置上方绘制标签
        ax.text(x, y, label, fontsize=8, ha='center', va='center', color='black',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    else:
        # 获取车辆的尺寸和朝向角度
        width, length = vehicle['width'], vehicle['length']
        angle_deg = math.degrees(vehicle['heading_rad'])
        front_left, front_right, rear_left, rear_right = calculate_four_points(x, y, heading_rad, width, length)

        # 绘制矩形表示车辆
        ax.fill([front_left[0], front_right[0], rear_right[0], rear_left[0]],
                [front_left[1], front_right[1], rear_right[1], rear_left[1]], color=color, alpha=0.5, zorder=10,
                edgecolor='k', label=label)
        # rect = plt.Rectangle((x - length / 2, y - width / 2), length, width, angle=angle_deg, color=color, alpha=0.5, label=label)
        # ax.add_patch(rect)
        # 在车辆中心下方绘制标签
        ax.text(x, y, label, fontsize=8, ha='center', va='bottom')
        
def calculate_four_points(center_x, center_y, heading, width, length):
    """
    在车辆中心点center_x, center_y,heading处,长width,宽length,
    返回四个角点的坐标
    """
    # 车辆前进方向的单位向量
    heading_vec = np.array([np.cos(heading), np.sin(heading)])

    # 车辆左前和右前方向向量
    left_vec = np.array([-heading_vec[1], heading_vec[0]])
    right_vec = np.array([heading_vec[1], -heading_vec[0]])

    # 计算四个角点在车辆坐标系下的位置
    front_left = length / 2 * heading_vec + width / 2 * left_vec
    front_right = length / 2 * heading_vec + width / 2 * right_vec
    rear_left = -length / 2 * heading_vec + width / 2 * left_vec
    rear_right = -length / 2 * heading_vec + width / 2 * right_vec

    # 转换到全局坐标系
    front_left += np.array([center_x, center_y])
    front_right += np.array([center_x, center_y])
    rear_left += np.array([center_x, center_y])
    rear_right += np.array([center_x, center_y])

    return front_left, front_right, rear_left, rear_right

def plot_roads(ax):
    points_list = [
        [(-26.56, 25.57), (-1.6, 25.57)],
        [(-26.56, 15.86), (-1.71, 16.18)],
        [(-26.56, 6.16), (-1.6, 6.48)],
        [(4.8, 32.29), (5.12, 43.7)],
        [(12.8, 32.18), (12.58, 43.59)],
        [(20.58, 32.61), (20.05, 43.70)],
        [(31.57, 26.1), (58.02, 26.63)],
        [(31.25, 16.18), (57.91, 16.5)],
        [(30.93, 7.12), (58.02, 6.8)],
        [(7.25, -9.84), (6.93, -0.56)],
        [(14.61, -10.05), (14.61, -0.45)],
        [(22.18, -10.05), (22.61, -0.13)]
    ]

    for i, line_points in enumerate(points_list):
        ax.plot([point[0] for point in line_points], [point[1] for point in line_points], linestyle='-', color='gray', alpha=0.5)

    all_points = [point for line_points in points_list for point in line_points]
    ax.scatter([point[0] for point in all_points], [point[1] for point in all_points], color='gray', alpha=0.5)

    lines_points_list = [
        [(-1.71, 16.18), (31.25, 16.18)],
        [(12.8, 32.18), (14.61, -0.45)]
    ]

    for i, line_points in enumerate(lines_points_list):
        ax.plot([point[0] for point in line_points], [point[1] for point in line_points], linestyle='--', color='gray', alpha=0.5)

    points_list = [(-1.6, 25.57), (4.8, 32.29), (20.58,32.61), (31.57, 26.1), \
                   (-1.6, 6.48), (6.93, -0.56), (22.61, -0.13), (30.93, 7.12), \
                   (-1.71, 16.18), (12.8, 32.18), (31.25, 16.18), (12.8, 32.18), \
                   (-1.71, 16.18), (14.61, -0.45), (31.25, 16.18), (14.61, -0.45)]

    control_points_list = [(4.5, 26), (25, 25), (4.5, 5), (25, 5), \
                           (14, 16), (14, 16), (14, 16), (14, 16)]

    for i in range(0, len(points_list) - 1, 2):
        if i + 1 < len(points_list):
            curve_x, curve_y = generate_quadratic_bezier_curve(points_list[i], control_points_list[int(i / 2)], points_list[i + 1])
            ax.plot(curve_x, curve_y, color='gray', linestyle='--', alpha=0.5)

    ax.scatter([point[0] for point in points_list], [point[1] for point in points_list], color='gray', alpha=0.5)

def plot_visual_view(ax, main_vehicle_state, main_vehicle_id, view_angle=120, view_distance=50):
    """
    绘制主车的可视扇形区域
    :param ax: Matplotlib的坐标轴对象
    :param main_vehicle_state: 主车的状态信息，包括位置和朝向
    :param main_vehicle_id: 主车的ID
    :param view_angle: 可视角度（度数）
    :param view_distance: 可视距离（米）
    """
    x, y = main_vehicle_state['x'], main_vehicle_state['y']
    if main_vehicle_id==8:
        print('wait!')
    heading_rad = main_vehicle_state['heading_rad']
    half_view_angle_rad = np.deg2rad(view_angle) / 2
    # 计算视野扇形区域的边界点

    start_angle = heading_rad - half_view_angle_rad
    end_angle = heading_rad + half_view_angle_rad

    # 获取扇形的坐标点
    X, Y = circle_array(x, y, view_distance, start_angle, end_angle)

    # 绘制扇形区域
    X = [x] + list(X) + [x]
    Y = [y] + list(Y) + [y]
    ax.fill(X, Y, color='green', alpha=0.2)

def circle_array(xc, yc, r, start, end):
    # 根据圆心、半径、起始角度、结束角度，生成圆弧的数据点
    phi1 = start
    phi2 = end
    dphi = (phi2 - phi1) / np.ceil(200 * np.pi * r * (phi2 - phi1))  # 根据圆弧周长设置等间距
    # array = np.arange(phi1, phi2+dphi, dphi) #生成双闭合区间(Python默认为左闭右开)
    # 当dphi为无限小数时，上述方法可能由于小数精度问题，导致array多一个元素，将代码优化为下面两行
    array = np.arange(phi1, phi2, dphi)
    array = np.append(array, array[-1] + dphi)  # array在结尾等距增加一个元素
    return xc + r * np.cos(array), yc + r * np.sin(array)
def bezier(t, points):
    n = len(points) - 1
    return sum(comb(n, i) * (1 - t)**(n - i) * t**i * points[i] for i in range(n + 1))

def generate_quadratic_bezier_curve(point1, control_point, point2, num_points=100):
    t_values = np.linspace(0, 1, num_points)
    x_values = [bezier(t, [point1[0], control_point[0], point2[0]]) for t in t_values]
    y_values = [bezier(t, [point1[1], control_point[1], point2[1]]) for t in t_values]
    return x_values, y_values

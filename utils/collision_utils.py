import numpy as np
import matplotlib.pyplot as plt

def get_faces(polygon):
    """
    获取多边形的边（面）。
    """
    faces = np.column_stack([polygon, np.roll(polygon, -1, axis=0)])
    return faces

def get_normals(faces):
    """
    获取边的法线。
    """
    edge_vectors = faces[:, 2:] - faces[:, :2]
    normals = np.column_stack([-edge_vectors[:, 1], edge_vectors[:, 0]])
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    return normals

def check_collision(polygon_a, polygon_b):
    """
    检查两个2D多边形之间是否发生碰撞。每次只处理一组多边形。
    """
    # 获取两个多边形的边
    faces_a = get_faces(polygon_a)
    faces_b = get_faces(polygon_b)
    
    # 合并边并获取法线
    faces = np.concatenate([faces_a, faces_b], axis=0)
    normals = get_normals(faces)
    
    # 将多边形投影到法线方向上
    polygon_a_proj = np.dot(polygon_a, normals.T)
    polygon_b_proj = np.dot(polygon_b, normals.T)
    
    # 计算投影的最小和最大值
    polygon_a_proj_min = np.min(polygon_a_proj, axis=0)
    polygon_a_proj_max = np.max(polygon_a_proj, axis=0)
    polygon_b_proj_min = np.min(polygon_b_proj, axis=0)
    polygon_b_proj_max = np.max(polygon_b_proj, axis=0)
    
    # 检查是否可分离
    not_separable = (polygon_a_proj_max > polygon_b_proj_min) & \
                    (polygon_b_proj_max > polygon_a_proj_min)
    
    # 返回碰撞结果
    return np.all(not_separable)

def plot_polygons(polygon_a, polygon_b, collision):
    """
    可视化两个多边形及其碰撞结果。
    """
    plt.figure()
    plt.fill(*zip(*np.vstack([polygon_a, polygon_a[0]])), 'b', alpha=0.5, label='Polygon A')
    plt.fill(*zip(*np.vstack([polygon_b, polygon_b[0]])), 'r', alpha=0.5, label='Polygon B')
    plt.title(f'Collision: {collision}')
    plt.legend()
    plt.show()

def calculate_vehicle_bbox(position, yaw, length, width):
        """
        计算车辆的边界框四个角点坐标
        
        Args:
            position: 车辆位置 [x, y]
            yaw: 车辆朝向（弧度）
            length: 车辆长度
            width: 车辆宽度
            
        Returns:
            边界框四个角点的坐标列表
        """
        # 计算车辆朝向向量和垂直向量
        heading_vec = np.array([np.cos(yaw), np.sin(yaw)])
        perpendicular_vec = np.array([-heading_vec[1], heading_vec[0]])
        heading_vec = heading_vec.reshape(-1)
        perpendicular_vec = perpendicular_vec.reshape(-1)
        
        # 计算边界框四个角点
        half_length = length / 2
        half_width = width / 2
        
        corners = [
            position + heading_vec * half_length + perpendicular_vec * half_width,  # 右前
            position + heading_vec * half_length - perpendicular_vec * half_width,  # 左前
            position - heading_vec * half_length - perpendicular_vec * half_width,  # 左后
            position - heading_vec * half_length + perpendicular_vec * half_width   # 右后
        ]
        
        return corners
import numpy as np
import warnings

def line(point0, point1):
    '''
    Get the line equation from two points: ax+by+c=0.
    '''
    x0, y0 = point0
    x1, y1 = point1
    a = y0 - y1
    b = x1 - x0
    c = x0*y1 - x1*y0
    return a, b, c


def angle(vec1x, vec1y, vec2x, vec2y):
    '''
    Calculate the angle between two vectors.

    Parameters:
    - vec1x: x-component of the first vector
    - vec1y: y-component of the first vector
    - vec2x: x-component of the second vector
    - vec2y: y-component of the second vector

    Returns:
    - angle: angle between the two vectors in [-pi, pi], with the first vector as the reference
    '''
    sin = vec1x * vec2y - vec2x * vec1y
    cos = vec1x * vec2x + vec1y * vec2y
    return np.arctan2(sin, cos)


def intersect(line0, line1):
    '''
    Get the intersection point of two lines.
    '''
    a0, b0, c0 = line0
    a1, b1, c1 = line1
    D = a0*b1 - a1*b0 # D==0 then two lines overlap
    D[D==0] = np.nan
    x = (b0*c1 - b1*c0)/D # x and y can be nan if D==0, which will be handled in the later steps
    y = (a1*c0 - a0*c1)/D
    return np.array([x, y])


def ison(line_start, line_end, point, tol=1e-5):
    '''
    Check if a point is on a line segment.
    tol is the tolerance for considering the point is on the line segment.
    '''
    crossproduct = (point[1]-line_start[1])*(line_end[0]-line_start[0]) - (point[0]-line_start[0])*(line_end[1]-line_start[1])
    dotproduct = (point[0]-line_start[0])*(line_end[0]-line_start[0]) + (point[1]-line_start[1])*(line_end[1]-line_start[1])
    squaredlength = (line_end[0]-line_start[0])**2 + (line_end[1]-line_start[1])**2
    return (np.absolute(crossproduct)<=tol)&(dotproduct>=0)&(dotproduct<=squaredlength)


def dist_p2l(point, line_start, line_end):
    '''
    Get the distance from a point to a line.
    '''
    return np.absolute((line_end[0]-line_start[0])*(line_start[1]-point[1])-(line_start[0]-point[0])*(line_end[1]-line_start[1]))/np.sqrt((line_end[0]-line_start[0])**2+(line_end[1]-line_start[1])**2)


def getpoints(samples, front_rear_only=False):
    '''
    Get the four points of the bounding box of vehicles i and j.
    '''
    # vehicle i
    type_i = samples['type_i'].values
    heading_i = samples[['hx_i','hy_i']].values
    perp_heading_i = np.array([-heading_i[:,1], heading_i[:,0]]).T
    heading_scale_i = np.tile(np.sqrt(heading_i[:,0]**2+heading_i[:,1]**2), (2,1)).T

    if type_i == 'pedestrian':
        point_i1 = samples[['x_i','y_i']].values.T
        point_i2 = samples[['x_i','y_i']].values.T
        point_i3 = samples[['x_i','y_i']].values.T
        point_i4 = samples[['x_i','y_i']].values.T
        point_up_i = samples[['x_i','y_i']].values
        point_down_i = samples[['x_i','y_i']].values
    else:
        length_i = np.tile(samples.length_i.values, (2,1)).T
        width_i = np.tile(samples.width_i.values, (2,1)).T
        point_up_i = samples[['x_i','y_i']].values + heading_i/heading_scale_i*length_i/2
        point_down_i = samples[['x_i','y_i']].values - heading_i/heading_scale_i*length_i/2
        if not front_rear_only:
            point_i1 = (point_up_i + perp_heading_i/heading_scale_i*width_i/2).T
            point_i2 = (point_up_i - perp_heading_i/heading_scale_i*width_i/2).T
            point_i3 = (point_down_i + perp_heading_i/heading_scale_i*width_i/2).T
            point_i4 = (point_down_i - perp_heading_i/heading_scale_i*width_i/2).T

    # vehicle j
    type_j = samples['type_j'].values
    heading_j = samples[['hx_j','hy_j']].values
    perp_heading_j = np.array([-heading_j[:,1], heading_j[:,0]]).T
    heading_scale_j= np.tile(np.sqrt(heading_j[:,0]**2+heading_j[:,1]**2), (2,1)).T

    if type_j == 'pedestrian':
        point_j1 = samples[['x_j','y_j']].values.T
        point_j2 = samples[['x_j','y_j']].values.T
        point_j3 = samples[['x_j','y_j']].values.T
        point_j4 = samples[['x_j','y_j']].values.T
        point_up_j = samples[['x_j','y_j']].values
        point_down_j = samples[['x_j','y_j']].values
    else:
        length_j = np.tile(samples.length_j.values, (2,1)).T
        width_j = np.tile(samples.width_j.values, (2,1)).T
        point_up_j = samples[['x_j','y_j']].values + heading_j/heading_scale_j*length_j/2
        point_down_j = samples[['x_j','y_j']].values - heading_j/heading_scale_j*length_j/2
        if not front_rear_only:
            point_j1 = (point_up_j + perp_heading_j/heading_scale_j*width_j/2).T
            point_j2 = (point_up_j - perp_heading_j/heading_scale_j*width_j/2).T
            point_j3 = (point_down_j + perp_heading_j/heading_scale_j*width_j/2).T
            point_j4 = (point_down_j - perp_heading_j/heading_scale_j*width_j/2).T

    if front_rear_only:
        return (point_up_i, point_down_i, point_up_j, point_down_j)
    else:
        return (point_i1, point_i2, point_i3, point_i4, point_j1, point_j2, point_j3, point_j4)


def rotate_coor(xyaxis, yyaxis, x2t, y2t):
    '''
    Rotate the coordinates (x2t, y2t) to the coordinate system with the y-axis along (xyaxis, yyaxis).

    Parameters:
    - xyaxis: x-coordinate of the y-axis in the new coordinate system
    - yyaxis: y-coordinate of the y-axis in the new coordinate system
    - x2t: x-coordinate to be rotated
    - y2t: y-coordinate to be rotated

    Returns:
    - x: rotated x-coordinate
    - y: rotated y-coordinate
    '''
    x = yyaxis/np.sqrt(xyaxis**2+yyaxis**2)*x2t-xyaxis/np.sqrt(xyaxis**2+yyaxis**2)*y2t
    y = xyaxis/np.sqrt(xyaxis**2+yyaxis**2)*x2t+yyaxis/np.sqrt(xyaxis**2+yyaxis**2)*y2t
    return x, y


def CurrentD(samples, toreturn='dataframe'):
    '''
    计算车辆i和j的边界框之间的距离（如果重叠则为0）。
    对于行人类型的对象，将其视为一个点进行计算。
    '''
    if toreturn!='dataframe' and toreturn!='values':
        warnings.warn("返回目标不正确。请指定'dataframe'或'values'。")
    else:
        # 检查是否有行人类型的对象
        type_i = samples['type_i'].values
        type_j = samples['type_j'].values
        
        # 如果i或j是行人，则使用点到点或点到边界框的距离计算
        if type_i == 'pedestrian' or type_j == 'pedestrian':
            # 获取对象的位置
            pos_i = samples[['x_i', 'y_i']].values
            pos_j = samples[['x_j', 'y_j']].values
            
            if type_i == 'pedestrian' and type_j == 'pedestrian':
                # 两个都是行人，直接计算点到点的距离
                cdist = np.sqrt(np.sum((pos_i - pos_j)**2, axis=1))
            elif type_i == 'pedestrian':
                # i是行人，j是车辆，计算点到边界框的距离
                _, _, _, _, point_j1, point_j2, point_j3, point_j4 = getpoints(samples)
                # 计算行人点到车辆边界框各边的距离
                dist_mat = []
                # 计算行人点到车辆边界框各顶点的距离
                dist_mat.append(np.sqrt((pos_i[:,0]-point_j1[0])**2+(pos_i[:,1]-point_j1[1])**2))
                dist_mat.append(np.sqrt((pos_i[:,0]-point_j2[0])**2+(pos_i[:,1]-point_j2[1])**2))
                dist_mat.append(np.sqrt((pos_i[:,0]-point_j3[0])**2+(pos_i[:,1]-point_j3[1])**2))
                dist_mat.append(np.sqrt((pos_i[:,0]-point_j4[0])**2+(pos_i[:,1]-point_j4[1])**2))
                for point_j_start, point_j_end in zip([point_j1, point_j4, point_j3, point_j2],
                                                     [point_j2, point_j3, point_j1, point_j4]):
                    # 计算点到线段的距离
                    ist = intersect(line(pos_i.T, pos_i.T+np.array([-(point_j_start-point_j_end)[1],(point_j_start-point_j_end)[0]])), line(point_j_start, point_j_end))
                    ist[:,~ison(point_j_start, point_j_end, ist, tol=1e-2)] = np.nan
                    dist_mat.append(np.sqrt((ist[0]-pos_i[:,0])**2+(ist[1]-pos_i[:,1])**2))
                cdist = np.nanmin(np.array(dist_mat), axis=0)
            else:
                # j是行人，i是车辆，计算点到边界框的距离
                point_i1, point_i2, point_i3, point_i4, _, _, _, _ = getpoints(samples)
                # 计算行人点到车辆边界框各边的距离
                dist_mat = []
                dist_mat.append(np.sqrt((pos_j[:,0]-point_i1[0])**2+(pos_j[:,1]-point_i1[1])**2))
                dist_mat.append(np.sqrt((pos_j[:,0]-point_i2[0])**2+(pos_j[:,1]-point_i2[1])**2))
                dist_mat.append(np.sqrt((pos_j[:,0]-point_i3[0])**2+(pos_j[:,1]-point_i3[1])**2))
                dist_mat.append(np.sqrt((pos_j[:,0]-point_i4[0])**2+(pos_j[:,1]-point_i4[1])**2))
                for point_i_start, point_i_end in zip([point_i1, point_i4, point_i3, point_i2],
                                                     [point_i2, point_i3, point_i1, point_i4]):
                    # 计算点到线段的距离
                    ist = intersect(line(pos_j.T, pos_j.T+np.array([-(point_i_start-point_i_end)[1],(point_i_start-point_i_end)[0]])), line(point_i_start, point_i_end))
                    ist[:,~ison(point_i_start, point_i_end, ist, tol=1e-2)] = np.nan
                    dist_mat.append(np.sqrt((ist[0]-pos_j[:,0])**2+(ist[1]-pos_j[:,1])**2))
                cdist = np.nanmin(np.array(dist_mat), axis=0)
        else:
            # 两个都是车辆，使用原来的边界框计算方法
            point_i1, point_i2, point_i3, point_i4, point_j1, point_j2, point_j3, point_j4 = getpoints(samples)

            dist_mat = []
            count_i = 0
            for point_i_start, point_i_end in zip([point_i1, point_i4, point_i3, point_i2],
                                                [point_i2, point_i3, point_i1, point_i4]):
                count_j = 0
                for point_j_start, point_j_end in zip([point_j1, point_j4, point_j3, point_j2],
                                                    [point_j2, point_j3, point_j1, point_j4]):
                    if count_i<2 and count_j<2 :
                        # 点到点的距离
                        dist_mat.append(np.sqrt((point_i_start[0]-point_j_start[0])**2+(point_i_start[1]-point_j_start[1])**2))
                        dist_mat.append(np.sqrt((point_i_start[0]-point_j_end[0])**2+(point_i_start[1]-point_j_end[1])**2))
                        dist_mat.append(np.sqrt((point_i_end[0]-point_j_start[0])**2+(point_i_end[1]-point_j_start[1])**2))
                        dist_mat.append(np.sqrt((point_i_end[0]-point_j_end[0])**2+(point_i_end[1]-point_j_end[1])**2))
                        
                    # 点到边的距离
                    ist = intersect(line(point_i_start, point_i_start+np.array([-(point_j_start-point_j_end)[1],(point_j_start-point_j_end)[0]])), line(point_j_start, point_j_end))
                    ist[:,~ison(point_j_start, point_j_end, ist, tol=1e-2)] = np.nan
                    dist_mat.append(np.sqrt((ist[0]-point_i_start[0])**2+(ist[1]-point_i_start[1])**2))

                    # 重叠的边界框
                    ist = intersect(line(point_i_start, point_i_end), line(point_j_start, point_j_end))
                    dist = np.ones(len(samples))*np.nan
                    dist[ison(point_i_start, point_i_end, ist)&ison(point_j_start, point_j_end, ist)] = 0
                    dist[np.isnan(ist[0])&(ison(point_i_start, point_i_end, point_j_start)|ison(point_i_start, point_i_end, point_j_end))] = 0
                    dist_mat.append(dist)
                    count_j += 1
                count_i += 1

            cdist = np.nanmin(np.array(dist_mat), axis=0)

        if toreturn=='dataframe':
            samples['CurrentD'] = cdist
            return samples
        elif toreturn=='values':
            return cdist


def D2C_ij(samples):
    '''
    Compute the distance to collision and mark whether the two vehicles are leaving each other.
    '''
    point_i1, point_i2, point_i3, point_i4, point_j1, point_j2, point_j3, point_j4 = getpoints(samples)
    relative_v = (samples[['vx_i','vy_i']].values - samples[['vx_j','vy_j']].values).T

    dist_mat = []
    leaving_mat = []
    # For each point of vehicle i
    for point_line_start in [point_i1,point_i2,point_i3,point_i4]:
        # For each edge of vehicle j
        for edge_start, edge_end in zip([point_j1, point_j3, point_j1, point_j2],[point_j2, point_j4, point_j3, point_j4]):
            point_line_end = point_line_start+relative_v
            # intersection point between 
            # 1) the edge of vehicle j and 
            # 2) the line extended from the point of vehicle i along the direction of the relative velocity of vehicle i and j
            ist = intersect(line(point_line_start, point_line_end), line(edge_start, edge_end))
            ist[:,~ison(edge_start, edge_end, ist, tol=1e-2)] = np.nan
            # distance from the point of vehicle i to the intersection point
            dist_ist = np.sqrt((ist[0]-point_line_start[0])**2+(ist[1]-point_line_start[1])**2)
            dist_ist[np.isnan(dist_ist)] = np.inf
            dist_mat.append(dist_ist)
            # determine if vehicle i and vehicle j are leaving each other based on if
            # 1) the relative velocity of vehicle i and j and
            # 2) the vector from the point of vehicle i to the intersection point
            # are in the same direction (>=0) or the opposite direction (<0)
            leaving = relative_v[0]*(ist[0]-point_line_start[0]) + relative_v[1]*(ist[1]-point_line_start[1])
            leaving[leaving>=0] = 20
            leaving[leaving<0] = 1
            leaving_mat.append(leaving)

    dtc = np.array(dist_mat).min(axis=0)
    leaving = np.nansum(np.array(leaving_mat),axis=0)
    return dtc, leaving


def dist_point_to_line(point, line_params):
    '''
    计算点到直线的距离，不考虑线段的端点限制。
    
    参数:
    - point: 点的坐标，形式为 [x, y]
    - line_params: 直线方程参数，形式为 [a, b, c]，对应于 ax + by + c = 0
    
    返回:
    - 点到直线的距离
    '''
    a, b, c = line_params
    denominator = np.sqrt(a**2 + b**2)
    if np.abs(denominator) < 1e-10:
        return np.nan
    return np.absolute(a*point[0] + b*point[1] + c) / denominator


def dist_point_to_line_from_points(point, line_point1, line_point2):
    '''
    通过直线上的两个点和待检查的点，计算点到直线的距离，不考虑线段的端点限制。
    
    参数:
    - point: 待检查点的坐标，形式为 [x, y]
    - line_point1: 直线上的第一个点，形式为 [x, y]
    - line_point2: 直线上的第二个点，形式为 [x, y]
    
    返回:
    - 点到直线的距离
    '''
    # 通过两点获取直线方程 ax + by + c = 0
    a, b, c = line(line_point1, line_point2)
    # 计算点到直线的距离
    return dist_point_to_line(point, [a, b, c])


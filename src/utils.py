import re

def is_opposite(dir1, dir2):
    if (dir1 == 'n' and dir2 == 's') or (dir1 == 's' and dir2 == 'n') or (dir1 == 'w' and dir2 == 'e') or (dir1 == 'e' and dir2 == 'w'):
        return True
    else:
        return False
    
def is_uturn(move_str):
    cd = extract_direction(move_str)
    start, end = cd.get('start', None), cd.get('end', None)
    if not start or not end:
        return False
    elif start['direction'] == end['direction']:
        return True 
    else:
        return False

def extract_direction(move_str):
    pattern = r'(?P<start_direction>[wesn]|NaN)(?P<start_lane>\d+)?_(?P<end_direction>[wesn]|NaN)(?P<end_lane>\d+)?'
    match = re.match(pattern, move_str)
    
    if match:
        start_direction = match.group('start_direction')
        start_lane = match.group('start_lane')
        end_direction = match.group('end_direction')
        end_lane = match.group('end_lane')

        if start_direction == 'NaN':
            start_direction = None
            start_lane = None
        else:
            start_lane = int(start_lane) if start_lane is not None else None

        if end_direction == 'NaN':
            end_direction = None
            end_lane = None
        else:
            end_lane = int(end_lane) if end_lane is not None else None

        result = {}
        if start_direction is not None or start_lane is not None:
            result['start'] = {
                'direction': start_direction,
                'lane': start_lane
            }
        if end_direction is not None or end_lane is not None:
            result['end'] = {
                'direction': end_direction,
                'lane': end_lane
            }
        
        return result
    else:
        raise ValueError("Incorrect input format.")

def check_conflict(str1, str2, ct1, ct2):
    """
    Judge whether two tracks will cross at the intersection.

    Args:
        str1: the trajectory string of the first vehicle
        str2: the trajectory string of the second vehicle
    
    Returns:
        str: The type of interaction between the two trajectories
    """
    # TODO: determine conflict type mainly from CrossType
    # 使用之前定义的函数提取轨迹信息
    cd1 = extract_direction(str1)
    cd2 = extract_direction(str2)
    
    # 提取第一辆车的起点和终点信息
    cd1_start, cd1_end = cd1.get('start', None), cd1.get('end', None)
    cd1_start_direction = cd1_start.get('direction', None) if cd1_start else None
    cd1_start_lane = cd1_start.get('lane', None) if cd1_start else None
    cd1_end_direction = cd1_end.get('direction', None) if cd1_end else None
    cd1_end_lane = cd1_end.get('lane', None) if cd1_end else None
    
    # 提取第二辆车的起点和终点信息
    cd2_start, cd2_end = cd2.get('start', None), cd2.get('end', None)
    cd2_start_direction = cd2_start.get('direction', None) if cd2_start else None
    cd2_start_lane = cd2_start.get('lane', None) if cd2_start else None
    cd2_end_direction = cd2_end.get('direction', None) if cd2_end else None
    cd2_end_lane = cd2_end.get('lane', None) if cd2_end else None
    
    if cd1_start is None or cd1_end is None or cd2_start is None or cd2_end is None:
        # TODO: define unusual behavior here
        return 'unusual behavior'
    
    # TODO: determine other situations
    if ct1 == ct2:
        if cd1_start_direction == cd2_start_direction:
            if cd1_start_lane == cd2_start_lane and cd1_end_lane == cd2_end_lane:
                return 'following'
            elif cd1_start_lane == cd2_start_lane:
                return 'diverging'
            elif cd1_end_lane == cd2_end_lane:
                return 'converging'
            elif (cd1_start_lane - cd2_start_lane) * (cd1_end_lane - cd2_end_lane) > 0:
                return 'crossing conflict: same cross type'
            else:
                return 'parallel'

        elif is_opposite(cd1_start_direction, cd2_start_direction):
            return 'parallel'
        
        else:
            return 'parallel'
        
    elif (ct1 == 'StraightCross' and ct2 == 'LeftTurn') or (ct1 == 'LeftTurn' and ct2 == 'StraightCross'):
        if cd1_start_direction == cd2_start_direction:
            if cd1_start_lane == cd2_start_lane:
                return 'diverging'
            elif (ct1 == 'LeftTurn' and cd1_start_lane < cd2_start_lane) or (ct2 == 'LeftTurn' and cd1_start_lane > cd2_start_lane):
                return 'left turn and straight cross conflict: same side'
            else:
                return 'parallel'
            
        elif is_opposite(cd1_start_direction, cd2_start_direction):
            return 'left turn and straight cross conflict: opposite side'
        
        else:
            return 'remaining in intersection'
        
    elif (ct1 == 'StraightCross' and ct2 == 'RightTurn') or (ct1 == 'RightTurn' and ct2 == 'StraightCross'):
        if cd1_start_direction == cd2_start_direction:
            if cd1_start_lane == cd2_start_lane:
                return 'diverging'
            elif (ct1 == 'RightTurn' and cd1_start_lane > cd2_start_lane) or (ct2 == 'RightTurn' and cd1_start_lane < cd2_start_lane):
                return 'right turn and straight cross conflict: start side'
            else:
                return 'parallel'
            
        elif is_opposite(cd1_start_direction, cd2_start_direction):
            return 'parallel'
        
        else:
            if cd1_end_direction == cd2_end_direction:
                if cd1_end_lane == cd2_end_lane:
                    return 'converging'
                elif (ct1 == 'StraightCross' and cd1_end_lane > cd2_end_lane) or (ct2 == 'StraightCross' and cd1_end_lane < cd2_end_lane):
                    return 'right turn and straight cross conflict: end side' 
                else:
                    return 'parallel'

        return 'parallel'

    else: # Right turn and left turn
        if cd1_start_direction == cd2_start_direction:
            if (ct1 == 'RightTurn' and cd1_start_lane > cd2_start_lane) or (ct2 == 'RightTurn' and cd1_start_lane < cd2_start_lane):
                return 'left turn and right turn conflict: start side'
            else:
                return 'parallel'

        elif is_opposite(cd1_start_direction, cd2_start_direction):
            if cd1_end_lane == cd2_end_lane:
                return 'converging'
            elif (ct1 == 'RightTurn' and cd1_end_lane < cd2_end_lane) or (ct2 == 'RightTurn' and cd1_end_lane > cd2_end_lane):
                return 'left turn and right turn conflict: end side'
            else:
                return 'parallel'

        else:
            return 'parallel'
            
    # # neighbor
    # if cd1_start_direction == cd2_start_direction and cd1_end_direction == cd2_end_direction:
    #     # if cd1_start_lane is not None and cd2_start_lane is not None and cd1_end_lane is not None and cd2_end_lane is not None:
    #     if (cd1_start_lane - cd2_start_lane) * (cd1_end_lane - cd2_end_lane) > 0:
    #         return 'cross conflict'
    #     elif (cd1_start_lane - cd2_start_lane) == 0 and (cd1_end_lane - cd2_end_lane) != 0:
    #         return 'diverging'
    #     elif (cd1_start_lane - cd2_start_lane) != 0 and (cd1_end_lane - cd2_end_lane) == 0:
    #         return "merging"
    #     elif (cd1_start_lane - cd2_start_lane) == 0 and (cd1_end_lane - cd2_end_lane) == 0:
    #         return "following"
    #     else:
    #         return "parallel"
    
    # # 如果只有起点方向相同
    # elif cd1_start_direction == cd2_start_direction:
    #     # if cd1_start_lane is not None and cd2_start_lane is not None:
    #     if cd1_start_lane == cd2_start_lane:
    #         return "diverging"
    #     else:
    #         if (ct1 == 'LeftTurn' and ct2 == "RightTurn") or (ct1 == 'RightTurn' and ct2 == "LeftTurn"):
    #             return 'turn conflict'
    #         elif (ct1 == 'LeftTurn' and ct2 == 'StraightCross') or (ct1 == 'StraightCross' and ct2 == 'LeftTurn'):
    #             return 'straight and left turn conflict'
    #         elif (ct1 == 'RightTurn' and ct2 == 'StraightCross') or (ct1 == 'StraightCross' and ct2 == 'RightTurn'):
    #             return 'straight and right turn conflict'
    
    # # 如果只有终点方向相同
    # elif cd1_end_direction == cd2_end_direction:
    #     # if cd1_end_lane is not None and cd2_end_lane is not None:
    #         # 不同起点相同终点，可能产生交叉
    #     if cd1_end_lane == cd2_end_lane:
    #         return "converging"
    #     else:
    #         if (ct1 == 'LeftTurn' and ct2 == "RightTurn") or (ct1 == 'RightTurn' and ct2 == "LeftTurn"):
    #             return 'turn conflict'
    #         elif (ct1 == 'LeftTurn' and ct2 == 'StraightCross') or (ct1 == 'StraightCross' and ct2 == 'LeftTurn'):
    #             # This should not happen without violating traffic signals.
    #             return 'straight and left turn conflict'
    #         elif (ct1 == 'RightTurn' and ct2 == 'StraightCross') or (ct1 == 'StraightCross' and ct2 == 'RightTurn'):
    #             return 'straight and right turn conflict'
    
    # # 起点终点方向都不同的情况
    # # TODO: consider situations with vibrations
    # elif is_opposite(cd1_start_direction, cd2_start_direction) and is_opposite(cd1_end_direction, cd2_end_direction):
    #     # parallel, but in opposite direction
    #     return 'opposite parallel'
    
    # elif is_opposite(cd1_start_direction, cd2_start_direction):
    #     # cannot conflict unless under traffic light violations
    #     return 'opposite diverging'

    # elif is_opposite(cd1_end_direction, cd2_end_direction):
    #     # cannot conflict unless under traffic light violations
    #     return 'opposite merging'
    # else:
    #     return 'converge and diverge'

if __name__ == "__main__":
    move_str1 = "n2_n3"
    # move_str2 = "w2_e5"
    print(is_uturn(move_str1))
#     result1 = check_conflict(move_str1, move_str2)
#     print(result1)

    # move_str2 = "NaN_e5"
    # result2 = extract_direction(move_str2)
    # print(result2)

    # move_str3 = "w1_NaN"
    # result3 = extract_direction(move_str3)
    # print(result3)

    # move_str4 = "NaN_NaN"
    # result4 = extract_direction(move_str4)
    # print(result4)
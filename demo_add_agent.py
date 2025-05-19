#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
演示如何使用SceneModifier添加新代理到场景并可视化
"""

import os
import sys
import argparse
from src.scene_editor import SceneEditor
from src.agent import Agent

def main():
    parser = argparse.ArgumentParser(description='添加新代理到场景并可视化')
    parser.add_argument('--data_dir', type=str, default="./data/tj", help='数据目录路径')
    parser.add_argument('--output_dir', type=str, default="./output/tj/dep2_long2", help='输出目录路径')
    parser.add_argument('--fragment_id', type=str, default="8_11_1 R20", help='片段ID')
    parser.add_argument('--ego_id', type=int, default=37, help='自车ID')
    parser.add_argument('--conflict_type', type=str, default=None, choices=['following', 'diverging', 'converging', 'crossing conflict: same cross type'], 
                        help='冲突类型，不指定则随机生成')
    parser.add_argument('--target_id', type=int, default=37, help='目标节点ID，不指定则随机选择')
    
    args = parser.parse_args()
    
    # 创建场景修改器
    editor = SceneEditor(args.data_dir, args.output_dir)
    
    # 加载数据
    print(f"加载片段 {args.fragment_id} 的数据...")
    editor.load_all(args.fragment_id, args.ego_id)
    
    # 准备边属性
    edge_attributes = ['left turn and right turn conflict: end side']
    # if args.conflict_type:
    #     edge_attributes = [args.conflict_type]

    new_agent = Agent.from_dict({"id": 1001, 
                                 "agent_type": "mv", 
                                 "agent_class": "car", 
                                 "cross_type": ["LeftTurn"], 
                                 "signal_violation": ["No violation of traffic lights"], 
                                 "retrograde_type": "normal", 
                                 "cardinal_direction": "e1_s3"})
    
    # 添加新代理并可视化
    print("添加新代理到场景并生成轨迹...")
    new_agent_id, track_data = editor.add_and_visualize_new_agent(
        new_agent=new_agent,
        target_node_id=args.target_id,
        edge_attributes=edge_attributes
    )
    
    if new_agent_id:
        print(f"成功生成新代理 ID: {new_agent_id}")
        print(f"生成的轨迹数据包含 {len(track_data['track_info']['frame_id'])} 帧")
        print(f"轨迹类型: {track_data['Type']}")
        print(f"代理类: {track_data['Class']}")
        print(f"横穿类型: {track_data['CrossType']}")
        if 'retrograde_type' in track_data and track_data['retrograde_type']:
            print(f"逆行类型: {track_data['retrograde_type']}")
        print(f"方向: {track_data['cardinal direction']}")
        
        # 显示异常帧数量
        anomaly_count = sum(track_data['anomalies'])
        print(f"异常帧数量: {anomaly_count} ({(anomaly_count / len(track_data['anomalies']) * 100):.1f}%)")
    else:
        print("添加新代理失败")
    
if __name__ == "__main__":
    main() 
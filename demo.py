#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
演示如何使用SceneModifier添加新代理到场景并可视化
"""

import os
import sys
import argparse
from src.scenario_editor import ScenarioEditor
from src.agent import Agent

def main():
    parser = argparse.ArgumentParser(description='合并因果图')
    parser.add_argument('--data_dir', type=str, default="./data/tj", help='数据目录路径')
    parser.add_argument('--output_dir', type=str, default="./output/tj/dep2_long", help='输出目录路径')
    parser.add_argument('--fragment_id', type=str, default="8_11_1 R20", help='片段ID')
    parser.add_argument('--ego_id', type=int, default=60, help='自车ID')
    args = parser.parse_args()
    
    # 创建场景修改器
    editor = ScenarioEditor(args.data_dir, args.output_dir)
    
    # 加载数据
    print(f"加载片段 {args.fragment_id} 的数据...")
    editor.load_all(args.fragment_id, args.ego_id)
    
    # 准备边属性
    edge_attributes = ['converging']
    edge_attributes_2 = ['converging', 'following']
    # if args.conflict_type:
    #     edge_attributes = [args.conflict_type]
    frame_shift_list = [-25, -40]

    new_agent = Agent.from_dict({"id": 1001, 
                                 "agent_type": "mv", 
                                 "agent_class": "car", 
                                 "cross_type": ["LeftTurn"], 
                                 "signal_violation": ["No violation of traffic lights"], 
                                 "retrograde_type": "normal", 
                                 "cardinal_direction": "s1_w4"})
    
    new_agent_2 = Agent.from_dict({"id": 1002, 
                                 "agent_type": "mv", 
                                 "agent_class": "car", 
                                 "cross_type": ["LeftTurn"], 
                                 "signal_violation": ["No violation of traffic lights"], 
                                 "retrograde_type": "normal", 
                                 "cardinal_direction": "s1_w4"})
    
    # 添加新代理并可视化
    print("添加新代理到场景并生成轨迹...")
    editor.add_agents([new_agent, new_agent_2], [edge_attributes, edge_attributes_2], frame_shift_list)
    
if __name__ == "__main__":
    main() 
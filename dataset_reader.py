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
from ssm.src.two_dimensional_ssms import TAdv, TTC2D, ACT
from ssm.src.geometry_utils import CurrentD
from src.causal_analyzer import CausalAnalyzer

fragment_id_list = ['7_28_1 R21', '8_10_1 R18', '8_10_2 R19', '8_11_1 R20']
ego_id_dict = {
    '7_28_1 R21': [1, 9, 11, 13, 26, 31, 79, 141, 144, 148, 162, 167, 170, 181],
    '8_10_1 R18': [13, 70, 76, 157],
    '8_10_2 R19': [75, 112, 126, 178],
    '8_11_1 R20': [4, 9, 37, 57, 60, 80, 84, 87, 93, 109, 159, 160, 161, 175, 216, 219, 289, 295, 316, 333, 372, 385, 390, 400, 479]
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/tj")
    parser.add_argument("--output_dir", type=str, default="./output/tj/causal_analysis/debug")
    parser.add_argument("--depth", type=int, default=2)
    args = parser.parse_args()

    with open(os.path.join(args.data_dir, "tp_info_tj.pkl"), "rb") as f:
        tp_info = pickle.load(f)

    meta_data_dict = {
        "signal_violation_behavior": set(),
        "cross_type": set(),
        "retrograde_type": set(),
        "cardinal_direction": set()
    }
    for fragment_id in fragment_id_list:
        fragment_data = tp_info[fragment_id]
        for id in fragment_data.keys():
            # print(id)
            track_data = fragment_data[id]
            if track_data['Signal_Violation_Behavior'] is not None:
                for signal_violation_behavior in track_data['Signal_Violation_Behavior']:
                    if signal_violation_behavior not in meta_data_dict['signal_violation_behavior']:
                        meta_data_dict['signal_violation_behavior'].add(signal_violation_behavior)
            if track_data['CrossType'] is not None:
                for cross_type in track_data['CrossType']:
                    if cross_type not in meta_data_dict['cross_type']:
                        meta_data_dict['cross_type'].add(cross_type)
            if 'retrograde_type' in track_data.keys():
                for retrograde_type in track_data['retrograde_type']:
                    if retrograde_type not in meta_data_dict['retrograde_type']:
                        meta_data_dict['retrograde_type'].add(retrograde_type)
            if 'cardinal direction' in track_data.keys():
                for cardinal_direction in track_data['cardinal direction']:
                    if cardinal_direction not in meta_data_dict['cardinal_direction']:
                        meta_data_dict['cardinal_direction'].add(cardinal_direction)
            # if track_data['CrossType'] not in meta_data_dict['cross_type']:
            #     meta_data_dict['cross_type'].add(track_data['CrossType'])
            # if track_data['retrograde_type'] not in meta_data_dict['retrograde_type']:
            #     meta_data_dict['retrograde_type'].add(track_data['retrograde_type'])
            # if track_data['cardinal direction'] not in meta_data_dict['cardinal_direction']:
            #     meta_data_dict['cardinal_direction'].add(track_data['cardinal direction'])

    print(meta_data_dict)
            
        
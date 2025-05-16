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

from src.my_utils import extract_direction

sys.path.append(os.getcwd())

fragment_id_list = ['7_28_1 R21']
ego_id_dict = {
    '7_28_1 R21': [1, 9, 11, 13, 26, 31, 79, 141, 144, 148, 162, 167, 170, 181],
    '8_10_1 R18': [13, 70, 76, 157],
    '8_10_2 R19': [75, 112, 126, 178],
    '8_11_1 R20': [4, 9, 37, 57, 60, 80, 84, 87, 93, 109, 159, 160, 161, 175, 216, 219, 289, 295, 316, 333, 372, 385, 390, 400, 479]
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/tj")
    args = parser.parse_args()

    with open(os.path.join(args.data_dir, "tp_info_tj.pkl"), "rb") as f:
        tp_info = pickle.load(f)
    # with open(os.path.join(self.data_dir, "frame_data_tj.pkl"), "rb") as f:
    #     self.frame_data = pickle.load(f)
    # with open(os.path.join(self.data_dir, "frame_data_tj_processed.pkl"), "rb") as f:
    #     self.frame_data_processed = pickle.load(f)

    import json

    meta_data_dict = {
        "agent_type": set(),
        "agent_class": set(),
        "signal_violation_behavior": set(),
        "cross_type": set(),
        "retrograde_type": set(),
        "cardinal_direction": set()
    }
    cross_type_dict = {
        "straight": {},
        "right": {},
        "left": {}
    }

    unclassified = set()

    for fragment_id in fragment_id_list:
        fragment_data = tp_info[fragment_id]
        for id in fragment_data.keys():
            # print(id)
            track_data = fragment_data[id]
            agent_type = track_data['Type']
            meta_data_dict['agent_type'].add(agent_type)

            agent_class = track_data['Class']
            meta_data_dict['agent_class'].add(agent_class)

            if track_data['Signal_Violation_Behavior'] is not None:
                for signal_violation_behavior in track_data['Signal_Violation_Behavior']:
                    meta_data_dict['signal_violation_behavior'].add(signal_violation_behavior)

            if track_data['CrossType'] is not None:     
                for cross_type in track_data['CrossType']:
                    meta_data_dict['cross_type'].add(cross_type)
                
            if 'retrograde_type' in track_data.keys():
                retrograde_type = track_data['retrograde_type']
                meta_data_dict['retrograde_type'].add(retrograde_type)

            if 'cardinal direction' in track_data.keys():
                cardinal_direction = track_data['cardinal direction']
                meta_data_dict['cardinal_direction'].add(cardinal_direction)

                if 'NaN' in cardinal_direction:
                    unclassified.add((fragment_id, id, cardinal_direction))
            else:
                cardinal_direction = None

            if agent_type == 'mv' and cardinal_direction:
                if 'straight' in cross_type.lower():
                    if cardinal_direction in cross_type_dict['straight'].keys():
                        cross_type_dict['straight'][cardinal_direction].append(id)
                    else:
                        cross_type_dict['straight'][cardinal_direction] = [id]
                elif 'left' in cross_type.lower():
                    if cardinal_direction in cross_type_dict['left'].keys():
                        cross_type_dict['left'][cardinal_direction].append(id)
                    else:
                        cross_type_dict['left'][cardinal_direction] = [id]
                elif 'right' in cross_type.lower():
                    if cardinal_direction in cross_type_dict['right'].keys():
                        cross_type_dict['right'][cardinal_direction].append(id)
                    else:
                        cross_type_dict['right'][cardinal_direction] = [id]

    print(meta_data_dict)
    print()
    print(unclassified)
    
    with open("typical_track.json", "w", encoding="utf-8") as f:
        json.dump(cross_type_dict, f, indent=4, ensure_ascii=False)
    print(cross_type_dict)
        
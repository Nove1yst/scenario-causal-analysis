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
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    causal_analyzer = CausalAnalyzer(args.data_dir, args.output_dir)
    if args.debug:
        causal_analyzer.load_data()
        causal_analyzer.analyze(fragment_id_list[0], ego_id_dict[fragment_id_list[0]][3], visualize_acc=True, visualize_ssm=True, visualize_cg=True, depth=0, max_depth=1)
    else:
        causal_analyzer.load_data()
        for fragment_id in fragment_id_list:
            for ego_id in ego_id_dict[fragment_id]:
                causal_analyzer.visualize_acceleration_analysis(fragment_id, ego_id)
                causal_analyzer.analyze(fragment_id, ego_id, depth=0, max_depth=args.depth)
                # cg = causal_analyzer.load_causal_graph(fragment_id, ego_id)
                # causal_analyzer.visualize_causal_graph(cg, fragment_id, ego_id)
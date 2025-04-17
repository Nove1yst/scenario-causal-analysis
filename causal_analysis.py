"""
    The definition of thresholds for safety metrics are based on the following website:
    https://criticality-metrics.readthedocs.io/en/latest/
"""
import os
import sys
import argparse
from tqdm import tqdm

sys.path.append(os.getcwd())
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
    causal_analyzer.load_data()
    if args.debug:
        fragment_id = fragment_id_list[0]
        ego_id = ego_id_dict[fragment_id][3]
        causal_analyzer.set_fragment_id(fragment_id)
        causal_analyzer.detect_risk(ego_id, 
                                    visualize_acc=True, 
                                    visualize_ssm=True, 
                                    visualize_cg=True, 
                                    depth=0, max_depth=1)
        causal_analyzer.load_cg(fragment_id, ego_id)
        causal_analyzer.extract_cg()
        causal_analyzer.visualize_cg(ego_id)
    else:
        for fragment_id in fragment_id_list:
            for ego_id in ego_id_dict[fragment_id]:
                causal_analyzer.set_fragment_id(fragment_id)
                causal_analyzer.detect_risk(ego_id, 
                                            depth=0, max_depth=args.depth,
                                            visualize_acc=True,
                                            visualize_cg=True,
                                            visualize_ssm=True)
                # causal_analyzer.load_cg(fragment_id, ego_id)
                causal_analyzer.extract_cg()
                causal_analyzer.visualize_cg(ego_id)
                # causal_analyzer.visualize_causal_graph(fragment_id, ego_id)
import os
import sys
import argparse
from src.scenario_generator import ScenarioGenerator

def main():
    parser = argparse.ArgumentParser(description='Generate scenarios')
    parser.add_argument('--data_dir', type=str, default="./data/tj", help='Path to the data directory')
    parser.add_argument('--output_dir', type=str, default="./output/tj/dep2_long", help='Path to the output directory')
    parser.add_argument('--cg_file', type=str, default="./data/tj/dep2_long/8_11_1 R20/8_11_1 R20_60_cg.json", help='Path to the causal graph file')
    parser.add_argument('--save_dir', type=str, default="./output/tj/generated_scenario", help='Path to save the generated scenario')
    # parser.add_argument('--fragment_id', type=str, default="8_11_1 R20", help='Scenario ID')
    parser.add_argument('--ego_id', type=int, default=60, help='Ego ID')
    args = parser.parse_args()
    
    generator = ScenarioGenerator(args.data_dir, args.output_dir) 
    generator.load_conflict_types()
    generator.load_typical_tracks()
    generator.load_data()
    # offset =  {2001: (-1, 1), 1005: (7, 0)}
    # frame_shift = {2001: 40, 1005: 10, 1004: 103, 1003: 0, 1002: 70, 1001: 75}
    # reference_id = {2001: 0, 1005: 2, 1004: 0, 1003: 2, 1002: 1, 1001: 1}
    offset =  {'P1001': (5, 0), 2001: (-1, 1), 1005: (7, 0)}
    frame_shift = {'P1001': -180, 2001: 40, 1005: 10, 1004: 103, 1003: 0, 1002: 70, 1001: 75}
    reference_id = {'P1001': 22, 2001: 0, 1005: 2, 1004: 0, 1003: 2, 1002: 1, 1001: 1}
    generator.generate_scenario(args.cg_file, 
                                args.ego_id, 
                                frame_shift, 
                                reference_id, 
                                offset=offset, 
                                start_frame=0, 
                                output_dir=args.save_dir)
    
if __name__ == "__main__":
    main() 
import os
import sys
import argparse
from src.scenario_generator import ScenarioGenerator

def main():
    parser = argparse.ArgumentParser(description='生成场景')
    parser.add_argument('--data_dir', type=str, default="./data/tj", help='数据目录路径')
    parser.add_argument('--output_dir', type=str, default="./output/tj/dep2_long", help='输出目录路径')
    parser.add_argument('--cg_file', type=str, default="./data/tj/dep2_long/8_11_1 R20/8_11_1 R20_60_cg.json", help='因果图文件路径')
    parser.add_argument('--save_dir', type=str, default="./output/tj/generated_scenario", help='保存路径')
    parser.add_argument('--fragment_id', type=str, default="8_11_1 R20", help='片段ID')
    parser.add_argument('--ego_id', type=int, default=60, help='自车ID')
    args = parser.parse_args()
    
    generator = ScenarioGenerator(args.data_dir, args.output_dir) 
    generator.load_conflict_types()
    generator.load_typical_tracks()
    generator.load_data()
    frame_shift = {1003: 0, 1002: 70, 1001: 75}
    reference_id = {1003: 2, 1002: 1, 1001: 1}
    generator.generate_scenario(args.cg_file, args.ego_id, frame_shift, reference_id, args.save_dir)
    
if __name__ == "__main__":
    main() 
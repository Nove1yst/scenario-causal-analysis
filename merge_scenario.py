from src.scenario_editor import ScenarioEditor

fragment_id_list = ['7_28_1 R21', '8_10_1 R18', '8_10_2 R19', '8_11_1 R20']
ego_id_dict = {
    '7_28_1 R21': [13, 26, 144, 148],
    '8_10_1 R18': [13, 70, 76],
    '8_10_2 R19': [126],
    '8_11_1 R20': [9, 37, 60, 80, 160, 219]
}

if __name__ == "__main__":
    editor = ScenarioEditor(data_dir="./data/tj", output_dir="./output/tj/dep2_long")

    editor.load_all(fragment_id="8_11_1 R20", ego_id=60)

    # 合并第二个场景
    merged_tracks = editor.merge_scenes(
        cg_file="output/tj/dep2_long/8_11_1 R20_60/mod_8_11_1 R20_60.json",
        merged_fragment_id="8_11_1 R20",
        merged_ego_id=400,
        frame_shift=50,
    )

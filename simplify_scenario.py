from src.scene_editor import SceneEditor

fragment_id_list = ['7_28_1 R21', '8_10_1 R18', '8_10_2 R19', '8_11_1 R20']
ego_id_dict = {
    '7_28_1 R21': [13, 26, 144, 148],
    '8_10_1 R18': [13, 70, 76],
    '8_10_2 R19': [126],
    '8_11_1 R20': [9, 37, 60, 80, 160, 219]
}

if __name__ == "__main__":
    modifier = SceneEditor(data_dir="./data/tj", output_dir="./output/tj/causal_analysis/debug")

    modifier.load_all(fragment_id='8_11_1 R20', ego_id=37)
    modifier.filter_and_visualize_scenario()
    # for fragment_id in fragment_id_list:
    #     for ego_id in ego_id_dict[fragment_id]:
    #         modifier.load_all(fragment_id=fragment_id, ego_id=ego_id)
    #         modifier.filter_and_visualize_scenario()
from src.scene_modifier import SceneModifier

fragment_id_list = ['7_28_1 R21', '8_10_1 R18', '8_10_2 R19', '8_11_1 R20']
ego_id_dict = {
    '7_28_1 R21': [13, 26, 144, 148],
    '8_10_1 R18': [13, 70, 76],
    '8_10_2 R19': [126],
    '8_11_1 R20': [9, 37, 60, 80, 160, 219]
}

if __name__ == "__main__":
    modifier = SceneModifier(data_dir="./data/tj", output_dir="./output/tj/dep2_long2")

    for fragment_id in fragment_id_list:
        for ego_id in ego_id_dict[fragment_id]:
            modifier.load_all(fragment_id=fragment_id, ego_id=ego_id)
            modifier.add_node_and_edge_to_cg(1001, 1)
            # modifier.visualize_cg()

            modifier.filter_and_visualize_scenario()

    # # 或者指定特定帧进行可视化
    # modifier.filter_and_visualize_scenario(frame_id=100)
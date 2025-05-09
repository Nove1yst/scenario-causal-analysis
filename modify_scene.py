from src.scene_modifier import SceneModifier

if __name__ == "__main__":
    modifier = SceneModifier(data_dir="./data/tj", output_dir="./output/tj/dep2_noparallel")

    modifier.load_all(fragment_id="7_28_1 R21", ego_id=1)
    modifier.add_node_and_edge_to_cg(1001, 1)
    modifier.visualize_cg()


    # # 可视化因果图中的代理（使用第一帧）
    # modifier.filter_and_visualize_scenario()

    # # 或者指定特定帧进行可视化
    # modifier.filter_and_visualize_scenario(frame_id=100)
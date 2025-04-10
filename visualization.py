import os
import pickle
import numpy as np
from tqdm import tqdm

from utils.visualization_utils import create_gif_from_scenario

if __name__ == "__main__":
    output_base_path = "./output/tj"
    with open(os.path.join(output_base_path, "track_change_tj.pkl"), "rb") as f:
        track_change = pickle.load(f)
    with open(os.path.join(output_base_path, "tp_info_tj.pkl"), "rb") as f:
        tp_info = pickle.load(f)
    with open(os.path.join(output_base_path, "frame_data_tj.pkl"), "rb") as f:
        frame_data = pickle.load(f)
    output_path = os.path.join(output_base_path, "visualization1")

    for fragment_id in tqdm(track_change.keys(), desc=f'Processing all scenes'):
        tracks = track_change[fragment_id]
        output_path_with_fragment_id = os.path.join(output_path, f"{fragment_id}")

        for track_id in tqdm(tracks.keys(), desc=f'Processing scene_id:{fragment_id}'):
            track = tracks[track_id]
            frame_start = int(tp_info[fragment_id][track_id]['State']['frame_id'].iloc[0])
            frame_end = int(tp_info[fragment_id][track_id]['State']['frame_id'].iloc[-1])
            anomalies_frame_id = track['track_info']['frame_id'][np.where(track['anomalies'] == True)[0]]
            if len(anomalies_frame_id) == 1 and (anomalies_frame_id == frame_end or anomalies_frame_id == frame_start):
                continue
            if anomalies_frame_id[-1] == frame_end:
                anomalies_frame_id = np.delete(anomalies_frame_id, -1)
            if anomalies_frame_id[0] == frame_start:
                anomalies_frame_id = np.delete(anomalies_frame_id, 0)

            if anomalies_frame_id[0] - 10 < frame_start:
                draw_frame_id_start = frame_start
            else:
                draw_frame_id_start = anomalies_frame_id[0] - 10

            if anomalies_frame_id[-1] + 10 > frame_end:
                draw_frame_id_end = frame_end
            else:
                draw_frame_id_end = anomalies_frame_id[-1] + 10
            frame_step = 1  # 默认步长为1
            # 如果帧数超过10000，则每10帧绘制一次
            if draw_frame_id_end - draw_frame_id_start > 10000:
                frame_step = 10
            # 如果帧数超过1000，则每5帧绘制一次
            elif draw_frame_id_end - draw_frame_id_start > 1000:
                frame_step = 5

            frame_info = []

            for i in range(draw_frame_id_start, draw_frame_id_end, frame_step):
                frame_info.append(frame_data[fragment_id][i])

            output_path_with_Type = os.path.join(output_path_with_fragment_id, f"{track['track_info']['Type']}")
            gif_number = 1  # Initialize the GIF number

            # 按每100帧分成一个GIF
            for gif_start in range(0, len(frame_info), 100):
                create_gif_from_scenario(
                    track,
                    frame_info[gif_start:gif_start + 100],
                    track_id,
                    fragment_id,
                    output_path_with_Type,
                    gif_number  # Pass the gif number here
                )
                gif_number += 1  # Increment the GIF number for each loop iteration
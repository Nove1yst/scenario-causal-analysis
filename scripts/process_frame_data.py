import os
import pickle
from tqdm import tqdm
def load_data(data_dir):
    with open(os.path.join(data_dir, "frame_data_tj.pkl"), "rb") as f:
        frame_data = pickle.load(f)

    # Processing fragment data. frame_data[fragment_id] is processed into a list with frame_id as the index
    for fragment_id, fragment_data in tqdm(frame_data.items(), desc='Processing fragment data'):
        print('Processing fragment data: ', fragment_id)
        for frame_id, frame_info in tqdm(fragment_data.items(), desc='Processing frame data'):
            # Processing frame data. frame_data[fragment_id][frame_id] is processed into a dict with track_id as the key
            frame_data[fragment_id][frame_id] = {track_data['tp_id']: track_data['vehicle_info'] for track_data in frame_info}

        frame_list = []
        for frame_id in sorted(fragment_data.keys()):
            frame_list.append(fragment_data[frame_id])
        frame_data[fragment_id] = frame_list

    print("Fragment data processed.")

    return frame_data

if __name__ == "__main__":
    data_dir = "./output/tj"
    frame_data = load_data(data_dir)
    with open(os.path.join(data_dir, "frame_data_tj_processed.pkl"), "wb") as f:
        pickle.dump(frame_data, f)
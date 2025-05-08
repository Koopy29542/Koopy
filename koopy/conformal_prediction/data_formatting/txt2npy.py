import os
import numpy as np
import pandas as pd

def main():
    # process both lobby2 and lobby3
    base_dirs = ('../lobby2', '../lobby3')
    for base in base_dirs:
        # use step=1 for lobby3 (no skipping), else step=10
        step = 1 if os.path.basename(base) == 'lobby3' else 10

        for root, dirs, files in os.walk(base):
            for file in files:
                if not file.endswith('.txt'):
                    continue

                txt_path = os.path.join(root, file)
                name, _ = os.path.splitext(file)
                npy_path = os.path.join(root, name + '.npy')

                # Convert txt to npy with the appropriate step
                data_npy = convert_txt_to_np_array(txt_path, step)
                np.save(npy_path, data_npy)

                # Load and print the shape of the npy file
                loaded_npy = np.load(npy_path)
                print(f'{npy_path} shape: {loaded_npy.shape}')

def convert_txt_to_np_array(txt_path, step=10):
    # (1) Load txt file into a DataFrame
    data = pd.read_csv(txt_path, sep='\t', header=None, index_col=False)
    data.columns = ['frame_id', 'track_id', 'pos_x', 'pos_y']

    # (2) Build frame list with given step
    min_frame = int(data['frame_id'].min())
    max_frame = int(data['frame_id'].max())
    unique_frames = list(range(min_frame, max_frame + 1, step))

    # (3) Track IDs as before
    min_tracks = int(data['track_id'].min())
    max_tracks = int(data['track_id'].max())
    unique_tracks = list(range(min_tracks, max_tracks + 1))

    # Create mappings for indexing
    frame_to_idx = {f: i for i, f in enumerate(unique_frames)}
    track_to_idx = {t: i for i, t in enumerate(unique_tracks)}

    F = len(unique_frames)
    N = len(unique_tracks)

    # (4) Initialize (F, N, 2) array with NaN
    data_npy = np.full((F, N, 2), np.nan, dtype=float)

    # (5) Populate the array
    for _, row in data.iterrows():
        f_id, t_id, x, y = row['frame_id'], row['track_id'], row['pos_x'], row['pos_y']
        if f_id in frame_to_idx:
            f_idx = frame_to_idx[f_id]
            t_idx = track_to_idx[t_id]
            data_npy[f_idx, t_idx, 0] = x
            data_npy[f_idx, t_idx, 1] = y

    return data_npy

if __name__ == '__main__':
    main()

import os
import numpy as np
import pandas as pd

def convert_csv_to_txt(directory):
    for i in range(10):  # Loop through files 0.csv to 9.csv
        csv_file = os.path.join(directory, f"{i}.csv")
        txt_file = os.path.join(directory, f"{i}.txt")
        
        if os.path.exists(csv_file):
            # Read CSV file
            df = pd.read_csv(csv_file)
            
            # Select only relevant columns (Frame, Agent_ID, X, Y)
            df = df[['Frame', 'Agent_ID', 'X', 'Y']]
            
            # Save to TXT file with tab separation and no header/index
            df.to_csv(txt_file, sep='\t', index=False, header=False)
            print(f"Converted {csv_file} to {txt_file}")
        else:
            print(f"File {csv_file} does not exist.")

def convert_txt_to_npy(directory):
    for i in range(10):  # Loop through files 0.txt to 9.txt
        txt_file = os.path.join(directory, f"{i}.txt")
        npy_file = os.path.join(directory, f"{i}.npy")
        
        if os.path.exists(txt_file):
            df = pd.read_csv(txt_file, sep='\t', names=['Frame', 'Agent_ID', 'X', 'Y'])
            
            # Ensure Frame starts from 1 and is continuous
            df['Frame'] -= df['Frame'].min() - 1
            
            max_frame = df['Frame'].max()
            max_agent = df['Agent_ID'].max()
            
            np_data = np.full((max_frame, max_agent, 2), np.nan)  # Initialize with NaNs
            
            for _, row in df.iterrows():
                frame_idx = int(row['Frame']) - 1  # Convert to zero-based index
                agent_idx = int(row['Agent_ID']) - 1  # Convert to zero-based index
                np_data[frame_idx, agent_idx, :] = [row['X'], row['Y']]
            
            # Save to .npy file
            np.save(npy_file, np_data)
            print(f"Saved data to {npy_file} with shape {np_data.shape}")
        else:
            print(f"File {txt_file} does not exist.")

# Specify your directory
directory = "/Users/jungjinlee/Documents/python file/conformal_prediction_joonho/lobby3"
convert_csv_to_txt(directory)
convert_txt_to_npy(directory)

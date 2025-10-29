# ============================================================
# EEG Segmentation Metadata Generator (2-second windows)
# ============================================================

import os
import scipy.io
import pandas as pd
import numpy as np

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
EEG_FOLDER = r"C:\Users\Atul Bhosale\OneDrive\Desktop\EEg\dream_eeg\Dream EEG with emotion labels"

# Sampling rate of EEG (check dataset docs, assumed 200 Hz)
SAMPLE_RATE = 200  
SEGMENT_LENGTH_SEC = 2
SEGMENT_SAMPLES = SEGMENT_LENGTH_SEC * SAMPLE_RATE  # 2s → 400 samples

# CSVs from your earlier split code
split_files = {
    "train": "train_split.csv",
    "val": "val_split.csv",
    "test": "test_split.csv"
}

# Output folder (current working directory)
output_dir = "./"
os.makedirs(output_dir, exist_ok=True)

# ------------------------------------------------------------
# Function to extract subject ID from filename
# Example: EEG_S0021_M1_E3_R01_N2.mat → S0021
# ------------------------------------------------------------
def extract_subject_id(filename):
    try:
        return filename.split("_")[1]  # 'S0021'
    except Exception:
        return "Unknown"

# ------------------------------------------------------------
# Segmentation Loop
# ------------------------------------------------------------
for split_name, csv_file in split_files.items():
    print(f"\n🔹 Processing {split_name.upper()} split ...")
    
    df = pd.read_csv(csv_file)
    all_segments = []

    for i, row in df.iterrows():
        filename = row["filename"]
        file_path = os.path.join(EEG_FOLDER, filename)

        if not os.path.exists(file_path):
            print(f"File not found: {filename}")
            continue

        # Load EEG data from .mat
        try:
            mat = scipy.io.loadmat(file_path)
            eeg_data = mat['Data']  # Shape: [channels, samples]
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue

        num_channels, total_samples = eeg_data.shape
        num_segments = total_samples // SEGMENT_SAMPLES

        subject_id = extract_subject_id(filename)

        # Create 2-second non-overlapping segments metadata
        for seg_idx in range(num_segments):
            start_time_sec = seg_idx * SEGMENT_LENGTH_SEC
            end_time_sec = (seg_idx + 1) * SEGMENT_LENGTH_SEC

            segment_info = {
                "split": split_name,
                "filename": filename,
                "segment_idx": seg_idx,
                "start_time_sec": start_time_sec,
                "end_time_sec": end_time_sec,
                "subject_id": subject_id
            }
            all_segments.append(segment_info)

    # --------------------------------------------------------
    # Save this split's segmentation info as CSV
    # --------------------------------------------------------
    segments_df = pd.DataFrame(all_segments)
    save_path = os.path.join(output_dir, f"{split_name}_segments.csv")
    segments_df.to_csv(save_path, index=False)

    print(f"{split_name}_segments.csv saved with {len(segments_df)} segments")

print("\n All segmentation metadata files created successfully!")
print("Files saved:")
print(" - train_segments.csv")
print(" - val_segments.csv")
print(" - test_segments.csv")

import os
import scipy.io
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter
from scipy.stats import entropy

# ==========================
# CONFIGURATION
# ==========================
EEG_FOLDER = r"C:\Users\Atul Bhosale\OneDrive\Desktop\EEg\dream_eeg\Dream EEG with emotion labels"

# Segmentation CSVs
SEG_FILES = {
    "train": "train_segments.csv",
    "val": "val_segments.csv",
    "test": "test_segments.csv"
}

# Sampling rate and bands
SAMPLE_RATE = 200  # Hz
BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}

# ==========================
# HELPER FUNCTIONS
# ==========================

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    return butter(order, [low, high], btype='band')

def bandpower(data, fs, band):
    """Compute average band power for each channel."""
    low, high = band
    b, a = butter_bandpass(low, high, fs)
    filtered = lfilter(b, a, data, axis=1)
    power = np.mean(filtered ** 2, axis=1)
    return power

def differential_entropy(data, fs, band):
    """Compute DE for each channel."""
    low, high = band
    b, a = butter_bandpass(low, high, fs)
    filtered = lfilter(b, a, data, axis=1)
    var = np.var(filtered, axis=1)
    de = 0.5 * np.log(2 * np.pi * np.e * var)
    return de

def hjorth_parameters(data):
    """Compute Hjorth Activity, Mobility, Complexity for each channel."""
    activity = np.var(data, axis=1)
    diff1 = np.diff(data, axis=1)
    diff2 = np.diff(diff1, axis=1)
    
    mobility = np.sqrt(np.var(diff1, axis=1) / activity)
    complexity = np.sqrt(np.var(diff2, axis=1) / np.var(diff1, axis=1)) / mobility
    return activity, mobility, complexity

# ==========================
# MAIN FEATURE EXTRACTION
# ==========================

def extract_features_for_split(split_name, seg_file):
    print(f"\n🔹 Extracting features for {split_name} set...")

    seg_df = pd.read_csv(seg_file)
    all_features = []

    for idx, row in seg_df.iterrows():
        fname = row['filename']
        start_t = int(row['start_time_sec'])
        end_t = int(row['end_time_sec'])
        subj = row['subject_id']

        # Load EEG
        mat_path = os.path.join(EEG_FOLDER, fname)
        if not os.path.exists(mat_path):
            print(f" File not found: {mat_path}")
            continue

        mat = scipy.io.loadmat(mat_path)
        eeg_data = mat['Data']  # shape [6, N]

        # Get segment samples
        start_sample = start_t * SAMPLE_RATE
        end_sample = end_t * SAMPLE_RATE
        segment = eeg_data[:, start_sample:end_sample]

        if segment.shape[1] == 0:
            continue  # skip empty

        # Feature containers
        feature_dict = {
            'split': split_name,
            'filename': fname,
            'subject_id': subj,
            'segment_idx': row['segment_idx'],
            'start_time_sec': start_t,
            'end_time_sec': end_t
        }

        # ---- BAND POWER + DE ----
        for band_name, (low, high) in BANDS.items():
            bp = bandpower(segment, SAMPLE_RATE, (low, high))
            de = differential_entropy(segment, SAMPLE_RATE, (low, high))
            for ch in range(segment.shape[0]):
                feature_dict[f'BP_{band_name}_ch{ch+1}'] = bp[ch]
                feature_dict[f'DE_{band_name}_ch{ch+1}'] = de[ch]

        # ---- HJORTH ----
        act, mob, comp = hjorth_parameters(segment)
        for ch in range(segment.shape[0]):
            feature_dict[f'Hjorth_activity_ch{ch+1}'] = act[ch]
            feature_dict[f'Hjorth_mobility_ch{ch+1}'] = mob[ch]
            feature_dict[f'Hjorth_complexity_ch{ch+1}'] = comp[ch]

        all_features.append(feature_dict)

    # Convert to DataFrame
    features_df = pd.DataFrame(all_features)

    # Save
    out_name = f"{split_name}_features.csv"
    features_df.to_csv(out_name, index=False)
    print(f" Saved {len(features_df)} segments → {out_name}")

    return features_df


# ==========================
# RUN FOR ALL SPLITS
# ==========================
if __name__ == "__main__":
    for split_name, seg_file in SEG_FILES.items():
        extract_features_for_split(split_name, seg_file)

    print("\n Feature extraction completed for all splits!")

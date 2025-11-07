import pandas as pd
import numpy as np


# Function to merge and save

def merge_and_save(features_path, split_path, output_csv, x_npy, y_npy):
    # Load feature and label files
    features = pd.read_csv(features_path)
    labels = pd.read_csv(split_path)

    # Merge based on filename
    merged = features.merge(labels[['filename', 'label']], on='filename', how='left')

    # Save labeled CSV
    merged.to_csv(output_csv, index=False)
    print(f" Saved labeled CSV: {output_csv}  ({len(merged)} rows)")

    # Extract feature columns (drop metadata + label)
    drop_cols = ['split', 'filename', 'segment_idx', 'label','subject_id']
    X = merged.drop(columns=[c for c in drop_cols if c in merged.columns]).values
    y = merged['label'].values

    # Save as NumPy arrays
    np.save(x_npy, X)
    np.save(y_npy, y)
    print(f" Saved NumPy arrays: {x_npy}, {y_npy}  (X: {X.shape}, y: {y.shape})\n")

# Train set

merge_and_save(
    "train_features.csv",
    "train_split.csv",
    "train_features_labeled.csv",
    "X_train.npy",
    "y_train.npy"
)


merge_and_save(
    "val_features.csv",
    "val_split.csv",
    "val_features_labeled.csv",
    "X_val.npy",
    "y_val.npy"
)


merge_and_save(
    "test_features.csv",
    "test_split.csv",
    "test_features_labeled.csv",
    "X_test.npy",
    "y_test.npy"
)

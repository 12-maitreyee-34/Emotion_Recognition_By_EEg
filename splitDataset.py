# Train-validation-test split 
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Path to your EEG files 
data_path = r'C:\Users\Atul Bhosale\OneDrive\Desktop\EEg\dream_eeg\Dream EEG with emotion labels'
files = [f for f in os.listdir(data_path) if f.endswith('.mat')]

# Extract labels from filenames
labels = []
for f in files:
    # Example filename: EEG_S0021_M1_E3_R01_N2.mat
    em = f.split('_E')[1][0]
    em = int(em)
    if em in [1,2]:      # negative
        labels.append(0)
    elif em == 3:         # neutral
        labels.append(1)
    elif em in [4,5]:     # positive
        labels.append(2)
    elif em == 0:         # no dream
        labels.append(3)  # label for no dream
    else:
        labels.append(-1)

# Convert to numpy arrays
files = np.array(files)
labels = np.array(labels)

# -----------------------------
# 70-20-10 stratified split
# -----------------------------
files_temp, files_test, labels_temp, labels_test = train_test_split(
    files, labels, test_size=0.1, stratify=labels, random_state=42
)

val_size = 0.222  # ≈ 20% of total
files_train, files_val, labels_train, labels_val = train_test_split(
    files_temp, labels_temp, test_size=val_size, stratify=labels_temp, random_state=42
)

print("Train:", len(files_train), "Val:", len(files_val), "Test:", len(files_test))
print("Class distribution in train:", np.bincount(labels_train))
print("Class distribution in val:", np.bincount(labels_val))
print("Class distribution in test:", np.bincount(labels_test))

# -----------------------------
# Save splits to current directory (VS Code)
# -----------------------------
output_dir = "./"  # Current working directory

# Save as numpy arrays
np.save(os.path.join(output_dir, "files_train.npy"), files_train)
np.save(os.path.join(output_dir, "labels_train.npy"), labels_train)

np.save(os.path.join(output_dir, "files_val.npy"), files_val)
np.save(os.path.join(output_dir, "labels_val.npy"), labels_val)

np.save(os.path.join(output_dir, "files_test.npy"), files_test)
np.save(os.path.join(output_dir, "labels_test.npy"), labels_test)

# -----------------------------
# Save splits separately as CSVs
# -----------------------------
output_dir = "./"  # Current working directory

# Convert splits into DataFrames
train_df = pd.DataFrame({
    'filename': files_train,
    'label': labels_train
})

val_df = pd.DataFrame({
    'filename': files_val,
    'label': labels_val
})

test_df = pd.DataFrame({
    'filename': files_test,
    'label': labels_test
})

# Save CSVs separately
train_df.to_csv(os.path.join(output_dir, 'train_split.csv'), index=False)
val_df.to_csv(os.path.join(output_dir, 'val_split.csv'), index=False)
test_df.to_csv(os.path.join(output_dir, 'test_split.csv'), index=False)

print("✅ Separate CSVs saved:")
print(" - train_split.csv  →", len(train_df), "rows")
print(" - val_split.csv    →", len(val_df), "rows")
print(" - test_split.csv   →", len(test_df), "rows")


print("All splits saved to current directory")
print("Files saved: .npy arrays + dataset_splits.csv")

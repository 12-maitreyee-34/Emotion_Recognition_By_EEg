import os
import scipy.io
import pandas as pd

EEG_FOLDER = r'C:\Users\Atul Bhosale\OneDrive\Desktop\EEg\dream_eeg\Dream EEG with emotion labels'

info_list = []

for file in os.listdir(EEG_FOLDER):
    if file.endswith('.mat'):
        mat = scipy.io.loadmat(os.path.join(EEG_FOLDER, file))
        eeg_array = mat['Data']
        
        num_channels, num_samples = eeg_array.shape
        fs = 200.0  # Hz (from DEED paper)
        duration_sec = num_samples / fs
        
        info_list.append({
            'filename': file,
            'channels': num_channels,
            'samples': num_samples,
            'sample_rate': fs,
            'duration_seconds': duration_sec,
            'data_shape': eeg_array.shape
        })

df = pd.DataFrame(info_list)
df.to_csv('eeg_dataset_info.csv', index=False)
print(df.head())

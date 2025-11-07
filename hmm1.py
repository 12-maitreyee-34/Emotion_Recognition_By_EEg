
#  Per-EEG HMM Analysis (Unsupervised)


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
import matplotlib.pyplot as plt

# USER SETTINGS

FEATURE_CSV = "train_features_labeled.csv"  # your features file
TARGET_FILENAME = "G_S0303_M2_E3_R2_N1_raw_ref.mat"  # EEG file to analyze
N_STATES = 3  # number of latent brain states


# Load the features

df = pd.read_csv(FEATURE_CSV)

# drop unnecessary columns (keep only numeric features)
exclude_cols = ['split', 'filename', 'subject_id', 'segment_idx',
                'start_time_sec', 'end_time_sec', 'label']
feature_cols = [c for c in df.columns if c not in exclude_cols]

# filter only that EEG file
df_eeg = df[df['filename'] == TARGET_FILENAME].sort_values('segment_idx')

if df_eeg.empty:
    raise SystemExit(f" Filename '{TARGET_FILENAME}' not found in CSV!")

print(f" Loaded {len(df_eeg)} segments for EEG: {TARGET_FILENAME}")


# Prepare feature matrix

X = df_eeg[feature_cols].values.astype(float)

# scale per EEG (important for numerical stability)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


#  4. Train Gaussian HMM

model = hmm.GaussianHMM(
    n_components=N_STATES,
    covariance_type='diag',
    n_iter=300,
    random_state=42
)
model.fit(X_scaled)

# predict hidden state per segment
states = model.predict(X_scaled)

#Analyze results

df_eeg['hidden_state'] = states

# transition matrix
trans_mat = model.transmat_
print("\n Transition Matrix:")
print(np.round(trans_mat, 3))

# mean feature vector for each state
state_means = model.means_
print("\n Mean feature values per state (scaled):")
print(pd.DataFrame(state_means, columns=feature_cols))

# dwell time stats
unique, counts = np.unique(states, return_counts=True)
percentages = 100 * counts / len(states)
print("\n State Occupancy (% of time):")
for s, p in zip(unique, percentages):
    print(f"State {s}: {p:.2f}%")

# number of transitions
transitions = np.sum(states[1:] != states[:-1])
print(f"\n Number of transitions: {transitions}")

#Plot hidden state timeline

plt.figure(figsize=(12, 3))
plt.step(df_eeg['segment_idx'], states, where='post', linewidth=2)
plt.yticks(range(N_STATES))
plt.xlabel("Segment index (time)")
plt.ylabel("Hidden State")
plt.title(f"HMM State Sequence – {TARGET_FILENAME}")
plt.grid(False)
plt.tight_layout()
plt.show()

#Save result
out_path = f"hmm_states_{TARGET_FILENAME.replace('.mat','')}.csv"
df_eeg[['filename', 'segment_idx', 'start_time_sec',
        'end_time_sec', 'hidden_state']].to_csv(out_path, index=False)

print(f"\n Results saved to: {out_path}")

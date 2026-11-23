import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, average_precision_score
)

SEED = 42
rng = np.random.default_rng(SEED)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

FEATURES = ["temperature", "humidity", "pm25", "co2"]

def generate_normals(n):
    # Two usual environmental regimes
    regime = rng.choice(2, size=n, p=[0.6, 0.4])
    temperature = rng.normal(23 + 5 * regime, 1.5)
    humidity = rng.normal(58 - 12 * regime, 5)
    pm25 = rng.normal(12 + 7 * regime, 3)
    co2 = rng.normal(500 + 180 * regime, 45)

    return np.column_stack([
        temperature, humidity, pm25, co2
    ])

# Normal data for training
X_train = generate_normals(800)

# Normal data for evaluation
X_test_normal = generate_normals(320)

# Global anomalies
X_global = generate_normals(40)
X_global[:, 2] += 55  # PM2.5 peak

# Combination anomalies
X_combination = generate_normals(40)
X_combination[:, 0] += 8
X_combination[:, 1] += 20

X_test = np.vstack([
    X_test_normal, X_global, X_combination
])

# 0 = normal, 1 = anomaly
y_test = np.array([0] * 320 + [1] * 80)

df_train = pd.DataFrame(X_train, columns=FEATURES)
df_test = pd.DataFrame(X_test, columns=FEATURES)

print("Train:", df_train.shape)
print("Test:", df_test.shape)
print("Test anomalies:", y_test.sum())

# TODO 1:
# Fit RobustScaler ONLY on X_train.
# Transform train and test.
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# TODO 2:
# Train IsolationForest with:
# n_estimators=200
# contamination=0.1
# random_state=SEED
iso_forest = IsolationForest(n_estimators=200, contamination=0.1, random_state=SEED)
iso_forest.fit(X_train_scaled)

# TODO 3:
# Train LocalOutlierFactor with:
# n_neighbors=25
# contamination=0.1
# novelty=True
lof = LocalOutlierFactor(n_neighbors=25, contamination=0.1, novelty=True)
lof.fit(X_train_scaled)

# TODO 4:
# Predict anomalies in X_test.
# Convert sklearn labels:
# -1 -> 1 (anomaly)
#  1 -> 0 (normal)
pred_if = iso_forest.predict(X_test_scaled)
pred_if = np.where(pred_if == -1, 1, 0)

pred_lof = lof.predict(X_test_scaled)
pred_lof = np.where(pred_lof == -1, 1, 0)

# TODO 5:
# Calculate precision, recall, F1,
# confusion matrix and average precision.
def evaluate(y_true, y_pred, y_scores, name):
    print(f"--- {name} ---")
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1:", f1_score(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("AP (continuous scores):", average_precision_score(y_true, y_scores))
    print()

# Continuous scores for Average Precision (higher = more anomalous)
iso_scores = -iso_forest.decision_function(X_test_scaled)
lof_scores = -lof.decision_function(X_test_scaled)

evaluate(y_test, pred_if, iso_scores, "Isolation Forest")
evaluate(y_test, pred_lof, lof_scores, "LOF")

# TODO 6:
# Compare both models and explain
# their differences.
print("Comparison and Limitations:")
print("Isolation Forest is usually better for global anomalies (extreme values) because it quickly isolates outliers.")
print("LOF is more effective for local anomalies (low density compared to neighbors), but struggles in high-dimensional spaces.")
print("Limitations:")
print("- Isolation Forest may struggle to detect subtle multivariate anomalies (combinations of normal values).")
print("- LOF is sensitive to the choice of k (n_neighbors) and the presence of regions with variable density, which can generate false positives at cluster boundaries.")
print("Interpretation of FP and FN:")
print("- False Positives (FP): Sensors marked as anomalous but that are normal. May be due to natural variations in environmental regimes.")
print("- False Negatives (FN): Undetected anomalies. Occur when anomalies are very subtle or resemble normal data (e.g., multivariate combinations).")

# Save results to CSV
results_df = pd.DataFrame(X_test, columns=FEATURES)
results_df['pred_if'] = pred_if
results_df['pred_lof'] = pred_lof
results_df['true_label'] = y_test
results_df.to_csv(os.path.join(RESULTS_DIR, "results.csv"), index=False)

# Plot anomalies
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Isolation Forest
axes[0].scatter(X_test_scaled[:, 0], X_test_scaled[:, 2], c=pred_if, cmap='coolwarm', alpha=0.6)
axes[0].set_title('Isolation Forest Anomalies')
axes[0].set_xlabel('Temperature (scaled)')
axes[0].set_ylabel('PM2.5 (scaled)')

# LOF
axes[1].scatter(X_test_scaled[:, 0], X_test_scaled[:, 2], c=pred_lof, cmap='coolwarm', alpha=0.6)
axes[1].set_title('LOF Anomalies')
axes[1].set_xlabel('Temperature (scaled)')
axes[1].set_ylabel('PM2.5 (scaled)')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "anomalies_plot.png"))
plt.close()

print("Results and plot saved to results folder.")

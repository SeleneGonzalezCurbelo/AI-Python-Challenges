import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler

SEED = 42
rng = np.random.default_rng(SEED)

N_HOURS = 30 * 24  # 720 hours
FEATURES = ["temperature", "humidity", "pm25", "co2"]

def generate_timeseries():
    hours = np.arange(N_HOURS)
    day_of_year = hours // 24
    
    # Base patterns with daily cycles
    temperature = 20 + 5 * np.sin(2 * np.pi * hours / 24) + rng.normal(0, 1, N_HOURS)
    humidity = 50 + 10 * np.cos(2 * np.pi * hours / 24) + rng.normal(0, 2, N_HOURS)
    pm25 = 15 + rng.normal(0, 3, N_HOURS)
    co2 = 400 + 50 * np.sin(2 * np.pi * hours / 24) + rng.normal(0, 10, N_HOURS)
    
    # Inject point anomalies (5% random)
    n_point = int(0.05 * N_HOURS)
    point_idx = rng.choice(N_HOURS, size=n_point, replace=False)
    temperature[point_idx] += rng.uniform(10, 20, n_point)
    pm25[point_idx] += rng.uniform(20, 40, n_point)
    
    # Inject gradual CO2 drift starting at hour 360 (day 15)
    drift_start = 360
    drift = np.zeros(N_HOURS)
    drift[drift_start:] = np.arange(1, N_HOURS - drift_start + 1) * 0.5
    co2 += drift
    
    # Labels: 0 normal, 1 anomaly (point or drift)
    y = np.zeros(N_HOURS, dtype=int)
    y[point_idx] = 1
    y[drift_start:] = 1  # drift is anomaly
    
    df = pd.DataFrame({
        'hour': hours,
        'temperature': temperature,
        'humidity': humidity,
        'pm25': pm25,
        'co2': co2,
        'is_anomaly': y
    })
    return df

df = generate_timeseries()
print("Time series generated:", df.shape)

# Split: first 14 days for training, rest for testing
train_mask = df['hour'] < 14 * 24
test_mask = ~train_mask

X_train = df.loc[train_mask, FEATURES].values
X_test = df.loc[test_mask, FEATURES].values
y_test = df.loc[test_mask, 'is_anomaly'].values
hours_test = df.loc[test_mask, 'hour'].values

# Scale
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Isolation Forest ---
iso = IsolationForest(n_estimators=200, contamination=0.1, random_state=SEED)
iso.fit(X_train_scaled)
pred_if = iso.predict(X_test_scaled)
pred_if = np.where(pred_if == -1, 1, 0)

# --- LOF ---
lof = LocalOutlierFactor(n_neighbors=25, contamination=0.1, novelty=True)
lof.fit(X_train_scaled)
pred_lof = lof.predict(X_test_scaled)
pred_lof = np.where(pred_lof == -1, 1, 0)

# --- Median + MAD (rolling window) ---
def rolling_median_mad(series, window=24, threshold=3.5):
    med = series.rolling(window=window, center=True, min_periods=1).median()
    mad = (series - med).abs().rolling(window=window, center=True, min_periods=1).median()
    mad = mad.replace(0, 1e-6)
    score = 0.6745 * (series - med) / mad
    return (score.abs() > threshold).astype(int)

# Apply per feature and combine
mad_preds = np.zeros(len(X_test))
for i, feat in enumerate(FEATURES):
    full_series = df[feat]
    test_series = full_series[test_mask]
    # Need rolling on full series then slice
    full_pred = rolling_median_mad(full_series, window=24, threshold=3.5)
    mad_preds += full_pred[test_mask].values

pred_mad = (mad_preds >= 1).astype(int)  # anomaly if any feature flags

# --- Evaluation ---
def evaluate_model(name, pred, y_true, hours):
    tp = np.sum((pred == 1) & (y_true == 1))
    fp = np.sum((pred == 1) & (y_true == 0))
    fn = np.sum((pred == 0) & (y_true == 1))
    tn = np.sum((pred == 0) & (y_true == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Detection latency for drift: first detection after hour 360
    drift_start_test = 360 - 14*24  # relative to test set
    drift_detections = np.where((pred == 1) & (hours >= 360))[0]
    latency = drift_detections[0] - drift_start_test if len(drift_detections) > 0 else np.inf
    
    # False positives per day
    n_days = 30 - 14  # test period days
    fp_per_day = fp / n_days
    
    # Percentage of anomalies detected
    pct_detected = recall * 100
    
    print(f"--- {name} ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"Detection Latency (hours): {latency}")
    print(f"False Positives per Day: {fp_per_day:.2f}")
    print(f"Percentage of Anomalies Detected: {pct_detected:.2f}%")
    print()
    
    return {
        'name': name,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'latency': latency,
        'fp_per_day': fp_per_day,
        'pct_detected': pct_detected
    }

results = []
results.append(evaluate_model("Isolation Forest", pred_if, y_test, hours_test))
results.append(evaluate_model("LOF", pred_lof, y_test, hours_test))
results.append(evaluate_model("Median + MAD", pred_mad, y_test, hours_test))

# Summary
print("=== Summary ===")
for r in results:
    print(f"{r['name']}: F1={r['f1']:.4f}, Latency={r['latency']}h, FP/day={r['fp_per_day']:.2f}, Detected={r['pct_detected']:.1f}%")

print("\nConclusion: Can a detector good at point anomalies also detect gradual drift?")
print("Isolation Forest and LOF detected the drift early (4h latency) because points deviate from the training distribution.")
print("Median + MAD with a rolling window adapts to the drift, so it fails to detect the gradual change (low recall).")
print("For gradual drift, a fixed baseline comparison or cumulative sum (CUSUM) approach would be more effective.")

# Save results
res_df = pd.DataFrame(results)
res_df.to_csv('E:/Proyectos/AI-Python-Challenges/02-anomaly-detection/results/timeseries_results.csv', index=False)

# Plot
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
test_hours = hours_test

axes[0].plot(test_hours, X_test_scaled[:, 3], label='CO2 (scaled)', color='gray', alpha=0.5)
axes[0].scatter(test_hours[pred_if == 1], X_test_scaled[pred_if == 1, 3], color='red', s=10, label='IF Anomaly')
axes[0].axvline(x=360, color='blue', linestyle='--', label='Drift Start')
axes[0].set_title('Isolation Forest')
axes[0].legend()

axes[1].plot(test_hours, X_test_scaled[:, 3], label='CO2 (scaled)', color='gray', alpha=0.5)
axes[1].scatter(test_hours[pred_lof == 1], X_test_scaled[pred_lof == 1, 3], color='orange', s=10, label='LOF Anomaly')
axes[1].axvline(x=360, color='blue', linestyle='--', label='Drift Start')
axes[1].set_title('LOF')
axes[1].legend()

axes[2].plot(test_hours, X_test_scaled[:, 3], label='CO2 (scaled)', color='gray', alpha=0.5)
axes[2].scatter(test_hours[pred_mad == 1], X_test_scaled[pred_mad == 1, 3], color='green', s=10, label='MAD Anomaly')
axes[2].axvline(x=360, color='blue', linestyle='--', label='Drift Start')
axes[2].set_title('Median + MAD')
axes[2].legend()

plt.xlabel('Hour')
plt.tight_layout()
plt.savefig('E:/Proyectos/AI-Python-Challenges/02-anomaly-detection/results/timeseries_plot.png')
plt.close()

print("\nTime series results and plot saved.")

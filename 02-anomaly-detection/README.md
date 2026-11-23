# 02 - Anomaly Detection in Environmental Sensors

## Context
This challenge focuses on environmental monitoring using sensors for temperature, humidity, PM2.5 particles and CO₂. The goal is to identify anomalous behaviors (failures, calibration errors or unusual environmental events) in an unsupervised scenario without labels.

Two complementary situations are considered: isolated extreme readings (global anomalies) and unusual combinations of values that are individually plausible (multivariate or local anomalies). A second script studies a gradual drift in a time series.

## Mission
Implement and compare two anomaly detectors — Isolation Forest and Local Outlier Factor (LOF) — on a synthetic test set with injected anomalies. Then extend the study to a 30-day time series that contains point anomalies and a slow CO₂ drift, and compare both detectors with a simple Median + MAD rule.

## Learning Objectives
- Distinguish between global and local anomalies.
- Understand the importance of fitting the scaler exclusively on training data to avoid data leakage.
- Compare unsupervised detectors without relying on reference labels during training.
- Interpret precision, recall, F1, average precision and confusion matrices.
- Analyze detection latency for gradual drift and false positives per day.

## Theoretical Foundation
**Isolation Forest** builds random partitions of the feature space. Anomalies need fewer splits to be isolated, so they obtain shorter average path lengths. It is effective for extreme values but may miss subtle multivariate deviations.

**Local Outlier Factor** compares the local density of a point with the local density of its `n_neighbors` nearest neighbors. A score well above 1 indicates a point that is sparse relative to its neighborhood, which makes it sensitive to local anomalies.

Both models are configured with `contamination = 0.1`, the expected fraction of anomalies. `novelty = True` lets LOF score unseen test observations after being fitted on normal training data only.

Features are standardized with `RobustScaler` fitted **only** on `X_train`; the same transformer is then applied to the test set.

For the time series, a rolling Median + MAD rule flags a value when

\[
\frac{0.6745\,|x - \tilde{x}|}{\text{MAD}} > 3.5
\]

using a centered 24-hour window. Because the window moves with the signal, it adapts to slow drifts instead of flagging them.

## Experiment Flow
1. **Data Generation**: two normal environmental regimes plus injected global peaks and multivariate anomalies.
2. **Preprocessing**: `RobustScaler` fitted on training data only.
3. **Modeling**: Isolation Forest (`n_estimators=200`) and LOF (`n_neighbors=25`, `novelty=True`).
4. **Evaluation**: convert `-1/1` labels to `1/0`, then compute precision, recall, F1, confusion matrices and average precision.
5. **Time Series Extension**: 720 hourly readings with point anomalies and a CO₂ drift starting at hour 360; evaluate IF, LOF and Median + MAD with latency and false positives per day.

## Results

### Static sensor test set (400 observations, 80 anomalies)

| Model | Precision | Recall | F1 | Average precision |
| :--- | ---: | ---: | ---: | ---: |
| Isolation Forest | 0.7547 | 1.0000 | **0.8602** | 0.9711 |
| LOF | 0.6957 | 1.0000 | 0.8205 | **1.0000** |

Both detectors recover all anomalies but produce false positives among normal readings. LOF ranks the anomalies perfectly (AP = 1.0) while its fixed-threshold F1 is lower, showing that threshold choice can hide good ranking quality.

### Time series test set (last 16 days)

| Model | F1 | Drift latency | FP / day | Anomalies detected |
| :--- | ---: | ---: | ---: | ---: |
| Isolation Forest | 0.8279 | 4 h | 0.06 | 70.8 % |
| LOF | **0.8298** | 4 h | 0.06 | 71.1 % |
| Median + MAD | 0.1053 | 4 h | 0.00 | 5.6 % |

IF and LOF flag the drift within four hours because the new CO₂ levels lie outside the training distribution. The rolling Median + MAD rule adapts to the drift and therefore misses it, despite having no false positives.

Artifacts: `results/results.csv`, `results/anomalies_plot.png`, `results/timeseries_results.csv` and `results/timeseries_plot.png`.

## Requirements
Dependencies are listed in the root `requirements.txt`:

```bash
pip install -r ../requirements.txt
```

## Usage
Run both scripts from this directory:

```bash
python solution.py
python solution_timeseries.py
```

The first script trains the two detectors and saves the static results; the second generates the time series and reports drift latency and false-positive rates.

## Conclusions and Limitations
- Isolation Forest is better suited to extreme global outliers; LOF is more sensitive to local density deviations but depends strongly on `n_neighbors`.
- A detector that performs well on point anomalies is not automatically good at gradual drift. For drift, a fixed baseline comparison or a cumulative sum (CUSUM) procedure would be more appropriate than a rolling window.
- The `contamination = 0.1` prior is fixed and would need tuning on real data.
- All data are synthetic with known regimes and injected anomalies, so absolute metric values should not be extrapolated to real sensor deployments.

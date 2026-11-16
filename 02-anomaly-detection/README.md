# 02-anomaly-detection: Anomaly Detection in Environmental Sensors

## Context
This challenge focuses on environmental monitoring using sensors for temperature, humidity, PM2.5 particles, and CO₂. The goal is to identify anomalous behaviors (failures, calibration errors, unusual environmental events) in an unsupervised learning scenario without labels.

## Mission
Implement and compare two anomaly detection algorithms to identify both extreme values and multivariate anomalies:
1. **Isolation Forest**: Based on random partitions to isolate global outliers.
2. **Local Outlier Factor (LOF)**: Based on local density to identify local anomalies.

## Learning Objectives
- Distinguish between global and local anomalies.
- Understand the importance of feature scaling in distance-based or partition-based models.
- Avoid information leakage (data leakage).
- Compare anomaly detection models without reference labels.

## Experiment Flow
1. **Data Generation**: Creation of synthetic sensor data with two environmental regimes (normal) and injected anomalies (global peaks and multivariate combinations).
2. **Preprocessing**: Scaling variables using `RobustScaler` fitted *exclusively* on training data to prevent data leakage.
3. **Modeling**: 
   - **Isolation Forest**: Trained with `n_estimators=200`, `contamination=0.1`.
   - **Local Outlier Factor**: Trained with `n_neighbors=25`, `contamination=0.1`, `novelty=True`.
4. **Evaluation**: Predictions on test data (converting -1 to 1 for anomalies and 1 to 0 for normal), followed by metric calculation (Precision, Recall, F1, Confusion Matrix, Average Precision).
5. **Comparison**: Analyzing differences in how each model handles global vs. local anomalies.

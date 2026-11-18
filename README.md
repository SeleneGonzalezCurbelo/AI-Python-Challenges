# AI-Python-Challenges

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9.0-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

A collection of Python-based machine learning challenges focused on unsupervised learning, anomaly detection, and data analysis.

## Challenges

### 01-territorial-clustering
Clustering of territorial profiles using K-Means. Discover patterns in geographic data to identify distinct territorial segments.

### 02-anomaly-detection
Anomaly detection in environmental sensors using Isolation Forest and Local Outlier Factor (LOF). Includes:
- Synthetic sensor data generation (temperature, humidity, PM2.5, CO₂)
- Unsupervised anomaly detection (global and local anomalies)
- Time series extension with gradual drift detection
- Comparison of IF, LOF, and Median+MAD approaches

### 03-text-classification
Automatic classification of citizen incident messages (lighting, cleaning, transport, water supply) using Bag of Words (BoW) and TF-IDF with LogisticRegression. Includes:
- Synthetic reproducible Spanish dataset of citizen messages
- Text representation with unigrams and bigrams
- Comparison of BoW vs TF-IDF classifiers on the same partitions
- Evaluation via macro-F1, confusion matrices, and per-class errors
- Analysis of generalization limits on unseen expressions
## Installation

1. Clone the repository:
```bash
git clone https://github.com/SeleneGonzalezCurbelo/AI-Python-Challenges.git
cd AI-Python-Challenges
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Challenge 01: Territorial Clustering
```bash
cd 01-territorial-clustering
python solution.py
```

### Challenge 02: Anomaly Detection
```bash
cd 02-anomaly-detection
python solution.py
python solution_timeseries.py
```

### Challenge 03: Text Classification
```bash
cd 03-text-classification
python solution.py
```
## Screenshots

### Challenge 01: Territorial Clustering
![Clusters](01-territorial-clustering/results/clusters_plot.png)

### Challenge 02: Anomaly Detection
![Anomalies](02-anomaly-detection/results/anomalies_plot.png)
![Time Series](02-anomaly-detection/results/timeseries_plot.png)

## Roadmap

- [x] Challenge 01: Territorial Clustering (K-Means)
- [x] Challenge 02: Anomaly Detection (Isolation Forest, LOF)
- [ ] Challenge 03: Text Classification (BoW, TF-IDF)
## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-challenge`)
3. Commit your changes (`git commit -m 'Add new challenge'`)
4. Push to the branch (`git push origin feature/new-challenge`)
5. Open a Pull Request

## Requirements
```bash
pip install -r requirements.txt
```

## Files to Upload to GitHub
```
AI-Python-Challenges/
├── README.md
├── requirements.txt
├── 01-territorial-clustering/
│   ├── README.md
│   ├── solution.py
│   └── results/
│       └── clusters_plot.png
├── 02-anomaly-detection/
│   ├── README.md
│   ├── solution.py
│   ├── solution_timeseries.py
│   └── results/
│       ├── anomalies_plot.png
│       ├── results.csv
│       ├── timeseries_plot.png
│       └── timeseries_results.csv
├── 03-text-classification/
│   ├── README.md
│   ├── solution.py
│   └── results/
```

**Exclude from upload:**
- `__pycache__/` directories
- Any `.pyc` files

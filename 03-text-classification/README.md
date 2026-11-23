# 03 - Text Classification: Citizen Incident Messages

## Context
Imagine a city council receiving hundreds of messages every day about street lighting, cleaning, transport and water supply issues. Someone currently has to read each message and forward it to the right department.

The goal is to build a model that assigns each message to the correct category, even when citizens use different expressions to describe the same incident.

> "We have had the streetlights off on our street for two nights." → **Public lighting**

## Mission
Compare two text representations — Bag of Words (BoW) and TF-IDF — with a `LogisticRegression` classifier trained on the same partitions. Then investigate what happens when the test messages use templates that never appear in training.

## Learning Objectives
1. Build a reproducible synthetic dataset of Spanish messages about citizen incidents.
2. Represent texts with unigrams and bigrams using BoW and TF-IDF.
3. Train and compare classifiers on identical splits.
4. Evaluate with macro-F1, confusion matrices and per-class errors.
5. Identify generalization limits on unseen expressions and unseen templates.

## Theoretical Foundation
**Bag of Words** represents a document by raw term counts. **TF-IDF** reweights each term by its inverse document frequency:

\[
\text{tf-idf}(t, d) = \text{tf}(t, d) \cdot \log\frac{N}{\text{df}(t)}
\]

where `N` is the number of documents and `df(t)` the number of documents containing term `t`. Rare terms therefore receive more weight than terms that appear everywhere.

Both representations use `ngram_range=(1, 2)` so that short phrases such as "no hay agua" can be learned. A `LogisticRegression` classifier is fitted on each matrix.

A standard random split puts paraphrases of the same template in both train and test. A `GroupShuffleSplit` keyed by `template_id` prevents this: every template appears on exactly one side of the split, which gives a more realistic measure of generalization.

## Experiment Flow
1. **Data Generation**: reproducible synthetic dataset of 480 Spanish messages from 20 templates across 4 categories.
2. **Preprocessing**: text vectorization with BoW and TF-IDF (unigrams and bigrams).
3. **Modeling**: one `LogisticRegression` per representation, using the same partitions.
4. **Evaluation**: macro-F1, accuracy, classification reports and confusion matrices.
5. **Generalization Analysis**: group-based split by template and a set of hand-written out-of-template messages.

## Results

| Split | BoW macro-F1 | TF-IDF macro-F1 |
| :--- | ---: | ---: |
| Random (384 train / 96 test) | 1.0000 | 1.0000 |
| Group-based, unseen templates (386 / 94) | **0.4453** | 0.2229 |

With the random split both models are perfect. When templates are held out, performance collapses — especially for TF-IDF, which relies more heavily on the exact vocabulary of the training templates.

On six hand-written out-of-template messages, both models score 5/6. The shared failure is:

> "Los semáforos de la zona también están sin luz." → predicted **Limpieza**, expected **Alumbrado público**

The word "luz" is strongly associated with cleaning templates in the synthetic data, so the classifier follows that lexical cue instead of the incident semantics.

Artifacts: `results/metrics.csv`, `results/metrics_groups.csv` and `results/confusion_matrices.png`.

## Requirements
Dependencies are listed in the root `requirements.txt`:

```bash
pip install -r ../requirements.txt
```

## Usage
The complete implementation is in `solution.py`. Run it with:

```bash
python solution.py
```

The script prints both evaluations, the group-split comparison and predictions on unseen messages, then saves the metric CSVs and confusion-matrix plot.

## Conclusions and Limitations
- Perfect in-template test scores do **not** demonstrate generalization: the test messages reuse the same templates and vocabulary as training, so lexical cues are enough.
- Holding out whole templates reveals the real behavior. The group-based split is the honest protocol for this synthetic dataset.
- The models learn correlations such as "farolas → lighting" rather than an understanding of the incident, which is why a semantically clear message can still be misclassified.
- The dataset is small, synthetic and limited to 20 templates. Real deployment would require a larger and more varied corpus, plus human review for low-confidence predictions.

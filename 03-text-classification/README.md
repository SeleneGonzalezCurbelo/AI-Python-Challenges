# 03 - Text Classification: Automatic classification of citizen incident messages

## Context
Imagine a city council receiving hundreds of messages daily about street lighting, cleaning, transport, and water supply issues. Currently, a person must read them and forward them to the corresponding department.

Your goal is to build a model that automatically classifies each message, even when citizens use different expressions to describe the same problem.

Example classification:

> "We have had the streetlights off on our street for two nights." → **Public lighting**

## Mission
You will compare two text representations: Bag of Words (BoW) and TF-IDF, training a `LogisticRegression` classifier for each.

Additionally, you will investigate an important problem: what happens when the model learns specific words from the training templates instead of understanding the incidents?

## Learning Objectives
1. Build a reproducible synthetic dataset of messages in Spanish.
2. Represent texts using BoW and TF-IDF, with unigrams and bigrams.
3. Train and compare classifiers using the same partitions.
4. Evaluate using macro-F1, confusion matrices, and per-class errors.
5. Identify generalization limitations on unseen expressions.

## Experiment Flow
1. **Data Generation**: Creation of a reproducible synthetic dataset of Spanish messages about citizen incidents.
2. **Preprocessing**: Text vectorization with BoW and TF-IDF (unigrams and bigrams).
3. **Modeling**: Training a `LogisticRegression` classifier for each representation using the same partitions.
4. **Evaluation**: Computation of macro-F1, confusion matrices, and per-class errors.
5. **Generalization Analysis**: Evaluation of behavior on expressions not seen during training.

## Requirements
To run this project, ensure you have the dependencies listed in the `requirements.txt` file at the root of the project:

```bash
pip install -r ../requirements.txt
```

## Usage
The complete implementation is in `solution.py`. Run it with:

```bash
python solution.py
```
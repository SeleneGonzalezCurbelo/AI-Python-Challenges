import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

CATEGORIES = [
    "Alumbrado público",
    "Limpieza",
    "Transporte",
    "Abastecimiento de agua",
]

TEMPLATES = {
    "Alumbrado público": [
        "Llevamos {n} noches con las farolas apagadas en {lugar}.",
        "Las luces de {lugar} no funcionan desde hace {n} días.",
        "Hay {n} postes de luz fundidos en {lugar}.",
        "El alumbrado de {lugar} está muy oscuro desde hace {n} noches.",
        "Las farolas de {lugar} parpadean y se apagan cada {n} horas.",
    ],
    "Limpieza": [
        "La basura no se recoge en {lugar} desde hace {n} días.",
        "Las calles de {lugar} están muy sucias desde hace {n} días.",
        "Los {n} contenedores de {lugar} rebosan de residuos.",
        "Hay desperdicios acumulados en {lugar} desde hace {n} días.",
        "La limpieza de {lugar} es insuficiente desde hace {n} semanas.",
    ],
    "Transporte": [
        "El autobús de la línea {n} siempre llega tarde a {lugar}.",
        "No hay servicio de metro en {lugar} desde hace {n} días.",
        "Los trenes de {lugar} circulan con {n} minutos de retraso.",
        "La frecuencia del transporte en {lugar} es muy baja cada {n} minutos.",
        "Las {n} paradas de {lugar} no muestran información de horarios.",
    ],
    "Abastecimiento de agua": [
        "No tenemos agua en {lugar} desde hace {n} días.",
        "El agua sale marrón del grifo en {lugar} desde hace {n} días.",
        "Hay {n} cortes de suministro en {lugar}.",
        "La presión del agua en {lugar} es muy baja desde hace {n} días.",
        "El agua de {lugar} tiene mal olor y sabor desde hace {n} días.",
    ],
}

LOCATIONS = ["nuestra calle", "el parque", "la avenida principal", "el barrio", "la plaza"]
NUMBERS = ["dos", "tres", "varias", "muchas", "cinco"]


def build_dataset(n_per_class: int = 120) -> pd.DataFrame:
    candidates = []
    for ci, category in enumerate(CATEGORIES):
        for ti, template in enumerate(TEMPLATES[category]):
            template_id = ci * 10 + ti
            for n in NUMBERS:
                for location in LOCATIONS:
                    candidates.append(
                        {
                            "text": template.format(n=n, lugar=location),
                            "categoria": category,
                            "template_id": template_id,
                        }
                    )

    # Deduplicate by text, keeping the first template id.
    unique_by_text = {}
    for row in candidates:
        unique_by_text.setdefault(row["text"], row)

    rows = []
    for category in CATEGORIES:
        category_rows = [r for r in unique_by_text.values() if r["categoria"] == category]
        chosen = random.sample(category_rows, n_per_class)
        rows.extend(chosen)

    df = pd.DataFrame(rows).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    return df


def evaluate(model, vectorizer, X_train, X_test, y_train, y_test):
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
        "y_pred": y_pred,
        "report": classification_report(y_test, y_pred, digits=4),
        "confusion": confusion_matrix(y_test, y_pred, labels=CATEGORIES),
    }


def main():
    df = build_dataset()
    X = df["text"].values
    y = df["categoria"].values
    groups = df["template_id"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    bow_vectorizer = CountVectorizer(ngram_range=(1, 2))
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))

    bow_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    tfidf_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)

    bow_results = evaluate(bow_model, bow_vectorizer, X_train, X_test, y_train, y_test)
    tfidf_results = evaluate(tfidf_model, tfidf_vectorizer, X_train, X_test, y_train, y_test)

    pred_bow = bow_results["y_pred"]
    pred_tfidf = tfidf_results["y_pred"]
    f1_bow = bow_results["macro_f1"]
    f1_tfidf = tfidf_results["macro_f1"]

    assert len(df) == 480
    assert df["categoria"].nunique() == 4
    assert len(X_train) == 384
    assert len(X_test) == 96
    assert len(pred_bow) == 96
    assert len(pred_tfidf) == 96
    assert 0 <= f1_bow <= 1
    assert 0 <= f1_tfidf <= 1

    # Group-based split: observations from the same template never share train/test.
    group_train_idx = group_test_idx = None
    for attempt in range(100):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE + attempt)
        tr, te = next(gss.split(X, y, groups=groups))
        if set(y[tr]) == set(CATEGORIES) and set(y[te]) == set(CATEGORIES) and len(te) > 0:
            group_train_idx, group_test_idx = tr, te
            break

    assert group_train_idx is not None
    X_train_g, X_test_g = X[group_train_idx], X[group_test_idx]
    y_train_g, y_test_g = y[group_train_idx], y[group_test_idx]

    bow_vectorizer_g = CountVectorizer(ngram_range=(1, 2))
    tfidf_vectorizer_g = TfidfVectorizer(ngram_range=(1, 2))
    bow_model_g = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    tfidf_model_g = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)

    bow_group_results = evaluate(bow_model_g, bow_vectorizer_g, X_train_g, X_test_g, y_train_g, y_test_g)
    tfidf_group_results = evaluate(tfidf_model_g, tfidf_vectorizer_g, X_train_g, X_test_g, y_train_g, y_test_g)

    print("=== Group-based split (unseen templates) ===")
    print(f"Train size: {len(X_train_g)}, Test size: {len(X_test_g)}")
    print("Train categories:", {c: int((y_train_g == c).sum()) for c in CATEGORIES})
    print("Test categories:", {c: int((y_test_g == c).sum()) for c in CATEGORIES})
    print(f"BoW macro-F1:    {bow_group_results['macro_f1']:.4f}")
    print(f"TF-IDF macro-F1: {tfidf_group_results['macro_f1']:.4f}")

    print("=== Random vs group split comparison ===")
    comparison_groups = pd.DataFrame(
        {
            "split": ["random", "random", "group", "group"],
            "model": ["BoW", "TF-IDF", "BoW", "TF-IDF"],
            "macro_f1": [
                f1_bow,
                f1_tfidf,
                bow_group_results["macro_f1"],
                tfidf_group_results["macro_f1"],
            ],
        }
    )
    comparison_groups.to_csv("results/metrics_groups.csv", index=False)
    print(comparison_groups.to_string(index=False))

    print("=== BoW (CountVectorizer + LogisticRegression) ===")
    print(f"Accuracy: {bow_results['accuracy']:.4f}")
    print(f"Macro-F1: {bow_results['macro_f1']:.4f}")
    print(bow_results["report"])

    print("=== TF-IDF (TfidfVectorizer + LogisticRegression) ===")
    print(f"Accuracy: {tfidf_results['accuracy']:.4f}")
    print(f"Macro-F1: {tfidf_results['macro_f1']:.4f}")
    print(tfidf_results["report"])

    comparison = pd.DataFrame(
        {
            "model": ["BoW", "TF-IDF"],
            "accuracy": [bow_results["accuracy"], tfidf_results["accuracy"]],
            "macro_f1": [bow_results["macro_f1"], tfidf_results["macro_f1"]],
        }
    )
    os.makedirs("results", exist_ok=True)
    comparison.to_csv("results/metrics.csv", index=False)

    # Confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, title, cm in zip(
        axes,
        ["BoW", "TF-IDF"],
        [bow_results["confusion"], tfidf_results["confusion"]],
    ):
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(title)
        ax.set_xticks(range(len(CATEGORIES)))
        ax.set_yticks(range(len(CATEGORIES)))
        ax.set_xticklabels(CATEGORIES, rotation=45, ha="right")
        ax.set_yticklabels(CATEGORIES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for i in range(len(CATEGORIES)):
            for j in range(len(CATEGORIES)):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig("results/confusion_matrices.png", dpi=120)
    plt.close(fig)

    # Out-of-template messages with expected labels
    unseen_messages = [
        ("Por las noches todo está a oscuras porque las lámparas no encienden.", "Alumbrado público"),
        ("Acumulación de residuos en las aceras desde el fin de semana.", "Limpieza"),
        ("Los autobuses pasan llenos y no paran en la parada.", "Transporte"),
        ("El suministro de agua potable se corta cada tarde.", "Abastecimiento de agua"),
        ("Los semáforos de la zona también están sin luz.", "Alumbrado público"),
        ("Hay un olor nauseabundo proveniente de los contenedores.", "Limpieza"),
    ]

    print("=== Predictions on out-of-template messages ===")
    bow_vec = bow_vectorizer
    tfidf_vec = tfidf_vectorizer
    bow_correct = 0
    tfidf_correct = 0
    for msg, expected in unseen_messages:
        bow_pred = bow_model.predict(bow_vec.transform([msg]))[0]
        tfidf_pred = tfidf_model.predict(tfidf_vec.transform([msg]))[0]
        bow_correct += int(bow_pred == expected)
        tfidf_correct += int(tfidf_pred == expected)
        print(f"- {msg}")
        print(f"    expected: {expected}")
        print(f"    BoW:      {bow_pred}")
        print(f"    TF-IDF:   {tfidf_pred}")
    print(f"Out-of-template accuracy - BoW: {bow_correct}/{len(unseen_messages)}")
    print(f"Out-of-template accuracy - TF-IDF: {tfidf_correct}/{len(unseen_messages)}")

    print("\n=== Generalization note ===")
    print(
        "A high test score does not guarantee generalization: the test set is built from "
        "the same templates and vocabulary as training, so the model may rely on specific "
        "lexical cues (e.g., 'farolas', 'basura') instead of understanding the underlying "
        "incident. When citizens phrase the problem differently, performance can drop "
        "even if in-template test metrics are strong."
    )


if __name__ == "__main__":
    main()

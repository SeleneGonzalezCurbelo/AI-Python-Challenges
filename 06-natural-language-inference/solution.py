import os
import random
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

LABELS = ["entailment", "contradiction", "neutral"]

MODEL_A = "Model A"
MODEL_B = "Model B"
MODEL_C = "Model C"
MODEL_NAMES = [MODEL_A, MODEL_B, MODEL_C]

ACTIONS = [
    "exportar informes",
    "exportar documentos",
    "descargar informes",
    "compartir documentos",
]
FORMATS = ["PDF", "Excel"]
NEUTRAL_HYPOTHESES = [
    "La plataforma permite programar exportaciones automáticas.",
    "La plataforma permite gestionar usuarios.",
    "Los usuarios con rol administrador pueden eliminar cuentas.",
]


def build_dataset(n_per_class: int = 10) -> pd.DataFrame:
    """Build a reproducible Spanish NLI dataset.

    Premises are deliberately simple positive permissions. Many contradiction
    hypotheses contain an explicit negation, which later becomes a statistical
    shortcut for the lexical models.
    """
    rows = []
    for ai, action in enumerate(ACTIONS):
        for fi, fmt in enumerate(FORMATS):
            template_id = ai * 10 + fi
            premise = f"La plataforma permite {action} en formato {fmt}."

            entailment_hypotheses = [
                f"La plataforma permite exportar documentos {fmt}.",
                f"La plataforma permite {action} en {fmt}.",
                f"Se pueden {action} en formato {fmt}.",
            ]

            other_formats = [f for f in FORMATS if f != fmt]
            contradiction_hypotheses = [
                f"La plataforma no permite {action} en formato {fmt}.",
            ] + [
                f"La plataforma permite {action} en formato {other_fmt}."
                for other_fmt in other_formats
            ]

            class_hypotheses = {
                "entailment": entailment_hypotheses,
                "contradiction": contradiction_hypotheses,
                "neutral": NEUTRAL_HYPOTHESES,
            }

            for label, hypotheses in class_hypotheses.items():
                for _ in range(n_per_class):
                    hypothesis = random.choice(hypotheses)
                    rows.append(
                        {
                            "premise": premise,
                            "hypothesis": hypothesis,
                            "label": label,
                            "grupo": template_id,
                        }
                    )

    df = pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)
    return df


def tokenize(text: str):
    text = text.lower()
    text = re.sub(r"[^\w\sáéíóúñ]", " ", text)
    return text.split()


def extract_format(text: str):
    for fmt in FORMATS:
        if fmt.lower() in text.lower():
            return fmt
    return None


def jaccard(text_a: str, text_b: str) -> float:
    """Jaccard similarity between two texts. Empty inputs return 0."""
    set_a = set(tokenize(text_a))
    set_b = set(tokenize(text_b))
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def model_a_predict(premises, hypotheses):
    """Lexical baseline: Jaccard similarity plus negation/entity-conflict rules."""
    predictions = []
    for premise, hypothesis in zip(premises, hypotheses):
        prem_tokens = tokenize(premise)
        hyp_tokens = tokenize(hypothesis)
        prem_fmt = extract_format(premise)
        hyp_fmt = extract_format(hypothesis)

        # Conflicting explicit format -> contradiction.
        if prem_fmt is not None and hyp_fmt is not None and prem_fmt != hyp_fmt:
            predictions.append("contradiction")
            continue

        # Explicit negation in the hypothesis but not in the premise -> contradiction.
        if "no" in hyp_tokens and "no" not in prem_tokens:
            predictions.append("contradiction")
            continue

        # Subset of premise tokens -> entailment.
        if set(hyp_tokens).issubset(set(prem_tokens)):
            predictions.append("entailment")
            continue

        similarity = jaccard(premise, hypothesis)
        if similarity >= 0.45:
            predictions.append("entailment")
        else:
            predictions.append("neutral")

    return predictions


def vectorize_pairs(vectorizer, premises, hypotheses, fit: bool = False):
    texts = [f"{p} [SEP] {h}" for p, h in zip(premises, hypotheses)]
    if fit:
        return vectorizer.fit_transform(texts)
    return vectorizer.transform(texts)


def evaluate_predictions(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=LABELS),
    }


TRANSFORMER_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
TRANSFORMER_BATCH_SIZE = 16


def load_transformer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_NAME)
    model.to(device)
    model.eval()
    return tokenizer, model, device


def transformer_predict(premises, hypotheses, tokenizer, model, device):
    predictions = []
    for start in range(0, len(premises), TRANSFORMER_BATCH_SIZE):
        batch_p = list(premises[start:start + TRANSFORMER_BATCH_SIZE])
        batch_h = list(hypotheses[start:start + TRANSFORMER_BATCH_SIZE])
        encoding = tokenizer(
            batch_p,
            batch_h,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        encoding = {key: value.to(device) for key, value in encoding.items()}
        with torch.no_grad():
            logits = model(**encoding).logits
        ids = logits.argmax(dim=-1).cpu().tolist()
        for idx in ids:
            raw_label = str(model.config.id2label[idx]).lower()
            if "entail" in raw_label:
                predictions.append("entailment")
            elif "contrad" in raw_label:
                predictions.append("contradiction")
            else:
                predictions.append("neutral")
    return predictions


def make_group_split(premises, labels, groups):
    group_train_idx = group_test_idx = None
    for attempt in range(100):
        gss = GroupShuffleSplit(
            n_splits=1, test_size=0.2, random_state=SEED + attempt
        )
        train_idx, test_idx = next(gss.split(premises, labels, groups=groups))
        if (
            set(labels[train_idx]) == set(LABELS)
            and set(labels[test_idx]) == set(LABELS)
            and len(test_idx) == 60
        ):
            group_train_idx, group_test_idx = train_idx, test_idx
            break

    assert group_train_idx is not None, "Could not create a valid group split."
    return group_train_idx, group_test_idx


def save_confusion_matrices(results_by_model):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, model_name in zip(axes, [MODEL_A, MODEL_B, MODEL_C]):
        cm = results_by_model[model_name]["confusion"]
        ax.imshow(cm, cmap="Blues")
        ax.set_title(model_name)
        ax.set_xticks(range(len(LABELS)))
        ax.set_yticks(range(len(LABELS)))
        ax.set_xticklabels(LABELS, rotation=45, ha="right")
        ax.set_yticklabels(LABELS)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for i in range(len(LABELS)):
            for j in range(len(LABELS)):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "confusion_matrices.png"), dpi=120)
    plt.close(fig)


def rag_verification_demo(tokenizer, model, device):
    knowledge_base = [
        "La plataforma permite exportar informes en formato PDF, pero no en Excel.",
        "Los usuarios con rol administrador pueden gestionar usuarios.",
        "La exportación automática está disponible para PDF.",
    ]

    generated_answer = (
        "La plataforma permite exportar documentos PDF. "
        "La plataforma permite exportar informes en Excel. "
        "La plataforma permite programar exportaciones automáticas."
    )

    claims = [claim.strip() for claim in generated_answer.split(".") if claim.strip()]

    def best_premise(claim):
        scores = [(jaccard(premise, claim), premise) for premise in knowledge_base]
        scores.sort(reverse=True)
        return scores[0][1]

    rows = []
    for claim in claims:
        premise = best_premise(claim)
        pred = transformer_predict([premise], [claim], tokenizer, model, device)[0]
        if pred == "entailment":
            status = "supported"
        elif pred == "contradiction":
            status = "contradicted"
        else:
            status = "insufficient_evidence"
        rows.append(
            {
                "claim": claim,
                "best_premise": premise,
                "nli_label": pred,
                "verification_status": status,
            }
        )

    demo_df = pd.DataFrame(rows)
    demo_df.to_csv(
        os.path.join(RESULTS_DIR, "verification_demo.csv"), index=False
    )
    return demo_df


def main():
    df = build_dataset()
    premises = df["premise"].values
    hypotheses = df["hypothesis"].values
    labels = df["label"].values
    groups = df["grupo"].values

    # Random stratified split.
    (
        prem_train,
        prem_test,
        hyp_train,
        hyp_test,
        y_train,
        y_test,
    ) = train_test_split(
        premises,
        hypotheses,
        labels,
        test_size=0.2,
        random_state=SEED,
        stratify=labels,
    )

    # Group-based split: templates never appear on both sides.
    group_train_idx, group_test_idx = make_group_split(
        premises, labels, groups
    )
    prem_train_g = premises[group_train_idx]
    prem_test_g = premises[group_test_idx]
    hyp_train_g = hypotheses[group_train_idx]
    hyp_test_g = hypotheses[group_test_idx]
    y_train_g = labels[group_train_idx]
    y_test_g = labels[group_test_idx]

    # DataFrames exposed for automatic tests.
    train = df.iloc[group_train_idx].reset_index(drop=True)
    test = df.iloc[group_test_idx].reset_index(drop=True)

    # Load transformer once for Model C.
    tokenizer, transformer_model, device = load_transformer()

    # ---------------------------------------------------------------- Model A
    pred_a_random = model_a_predict(prem_test, hyp_test)
    pred_a_group = model_a_predict(prem_test_g, hyp_test_g)
    pred_baseline = pred_a_group

    # ---------------------------------------------------------------- Model B
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    x_train_b = vectorize_pairs(tfidf_vectorizer, prem_train, hyp_train, fit=True)
    x_test_b = vectorize_pairs(tfidf_vectorizer, prem_test, hyp_test, fit=False)
    model_b = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=SEED
    )
    model_b.fit(x_train_b, y_train)
    pred_b_random = model_b.predict(x_test_b).tolist()

    tfidf_vectorizer_g = TfidfVectorizer(ngram_range=(1, 2))
    x_train_bg = vectorize_pairs(
        tfidf_vectorizer_g, prem_train_g, hyp_train_g, fit=True
    )
    x_test_bg = vectorize_pairs(
        tfidf_vectorizer_g, prem_test_g, hyp_test_g, fit=False
    )
    model_b_g = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=SEED
    )
    model_b_g.fit(x_train_bg, y_train_g)
    pred_b_group = model_b_g.predict(x_test_bg).tolist()
    pred_tfidf = pred_b_group

    # ---------------------------------------------------------------- Model C
    pred_c_random = transformer_predict(
        prem_test, hyp_test, tokenizer, transformer_model, device
    )
    pred_c_group = transformer_predict(
        prem_test_g, hyp_test_g, tokenizer, transformer_model, device
    )

    # ---------------------------------------------------------------- Metrics
    metrics_random = {
        MODEL_A: evaluate_predictions(y_test, pred_a_random),
        MODEL_B: evaluate_predictions(y_test, pred_b_random),
        MODEL_C: evaluate_predictions(y_test, pred_c_random),
    }
    metrics_group = {
        MODEL_A: evaluate_predictions(y_test_g, pred_a_group),
        MODEL_B: evaluate_predictions(y_test_g, pred_b_group),
        MODEL_C: evaluate_predictions(y_test_g, pred_c_group),
    }

    # Save random-split metrics.
    metrics_df = pd.DataFrame(
        [
            {
                "model": name,
                "accuracy": values["accuracy"],
                "macro_f1": values["macro_f1"],
            }
            for name, values in metrics_random.items()
        ]
    )
    metrics_df.to_csv(os.path.join(RESULTS_DIR, "metrics.csv"), index=False)

    # Save group-vs-random comparison.
    group_rows = []
    for split_name, metrics in [
        ("random", metrics_random),
        ("group", metrics_group),
    ]:
        for name, values in metrics.items():
            group_rows.append(
                {
                    "split": split_name,
                    "model": name,
                    "accuracy": values["accuracy"],
                    "macro_f1": values["macro_f1"],
                }
            )
    metrics_groups_df = pd.DataFrame(group_rows)
    metrics_groups_df.to_csv(
        os.path.join(RESULTS_DIR, "metrics_groups.csv"), index=False
    )

    # Save predictions for the random test split.
    predictions_df = pd.DataFrame(
        {
            "premise": prem_test,
            "hypothesis": hyp_test,
            "true_label": y_test,
            "pred_model_a": pred_a_random,
            "pred_model_b": pred_b_random,
            "pred_model_c": pred_c_random,
        }
    )
    predictions_df.to_csv(
        os.path.join(RESULTS_DIR, "predictions.csv"), index=False
    )

    # Confusion matrices (random split).
    confusion_by_model = {
        MODEL_A: {
            "confusion": confusion_matrix(
                y_test, pred_a_random, labels=LABELS
            )
        },
        MODEL_B: {
            "confusion": confusion_matrix(
                y_test, pred_b_random, labels=LABELS
            )
        },
        MODEL_C: {
            "confusion": confusion_matrix(
                y_test, pred_c_random, labels=LABELS
            )
        },
    }
    save_confusion_matrices(confusion_by_model)

    # RAG verification demo.
    demo_df = rag_verification_demo(tokenizer, transformer_model, device)

    # ---------------------------------------------------------------- Reports
    print("=== Dataset ===")
    print(f"Total pairs: {len(df)}")
    print(df["label"].value_counts().to_string())
    print(f"Random split: {len(prem_train)} train / {len(prem_test)} test")
    print(
        f"Group split: {len(prem_train_g)} train / {len(prem_test_g)} test"
    )

    print("\n=== Random split ===")
    for name in [MODEL_A, MODEL_B, MODEL_C]:
        values = metrics_random[name]
        print(
            f"{name}: accuracy={values['accuracy']:.4f}, "
            f"macro_f1={values['macro_f1']:.4f}"
        )

    print("\n=== Group-based split (unseen templates) ===")
    for name in [MODEL_A, MODEL_B, MODEL_C]:
        values = metrics_group[name]
        print(
            f"{name}: accuracy={values['accuracy']:.4f}, "
            f"macro_f1={values['macro_f1']:.4f}"
        )

    print("\n=== Classification report: Model C (random split) ===")
    print(
        classification_report(
            y_test, pred_c_random, labels=LABELS, digits=4
        )
    )

    print("=== RAG verification demo ===")
    print(demo_df.to_string(index=False))

    print("\n=== Generalization note ===")
    print(
        "Lexical regularities (explicit negations, repeated formats) can become "
        "shortcuts: on this deliberately simple dataset Model A and Model B can "
        "reach very high scores by exploiting them. Model C is less tied to "
        "those surface cues and is preferred for semantic verification, even when "
        "its score is lower. A neutral prediction means insufficient evidence, "
        "not that the claim is false."
    )

    # Sanity assertions.
    assert len(df) == 240
    assert df["label"].nunique() == 3
    assert (df["label"].value_counts() == 80).all()
    assert len(train) == 180
    assert len(test) == 60
    assert set(train["grupo"]).isdisjoint(set(test["grupo"]))
    assert jaccard("hola mundo", "hola mundo") == 1
    assert jaccard("hola", "adiós") == 0
    assert jaccard("", "") == 0
    assert len(pred_baseline) == len(test)
    assert len(pred_tfidf) == len(test)
    assert set(pred_baseline).issubset(LABELS)
    assert set(pred_tfidf).issubset(LABELS)

    # Expose variables globally for test harnesses.
    globals()["df"] = df
    globals()["train"] = train
    globals()["test"] = test
    globals()["pred_baseline"] = pred_baseline
    globals()["pred_tfidf"] = pred_tfidf
    globals()["LABELS"] = LABELS
    globals()["jaccard"] = jaccard


if __name__ == "__main__":
    main()

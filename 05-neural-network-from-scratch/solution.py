import csv
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

SEED = 42
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset: two interlaced moons. X_test stays untouched as a final holdout.
# ---------------------------------------------------------------------------
X, y = make_moons(n_samples=1000, noise=0.20, random_state=SEED)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Independent validation set, carved out of the training partition.
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train, y_train, test_size=0.20, stratify=y_train.ravel(), random_state=SEED
)


class History(list):
    """List of per-epoch losses that also exposes validation series by name."""

    def __getitem__(self, key):
        if key == "loss":
            return list(self)
        if isinstance(key, str):
            return getattr(self, key)
        return super().__getitem__(key)


class NeuralNetwork:
    def __init__(self, hidden_dims=(8,), input_dim=2, seed=SEED):
        if isinstance(hidden_dims, int):
            hidden_dims = (hidden_dims,)
        self.sizes = (input_dim,) + tuple(hidden_dims) + (1,)
        self.n_layers = len(self.sizes) - 1

        rng = np.random.default_rng(seed)
        self.params = {}
        for layer in range(1, self.n_layers + 1):
            fan_in, fan_out = self.sizes[layer - 1], self.sizes[layer]
            W = rng.normal(0, np.sqrt(2 / fan_in), size=(fan_in, fan_out))
            b = np.zeros((1, fan_out))
            self.params[f"W{layer}"] = W
            self.params[f"b{layer}"] = b
            # Keep the W1/b1/W2/b2 names of the original challenge.
            setattr(self, f"W{layer}", W)
            setattr(self, f"b{layer}", b)

        self._init_adam()

    # ------------------------------------------------------------------ setup
    def _init_adam(self):
        self.m = {name: np.zeros_like(p) for name, p in self.params.items()}
        self.v = {name: np.zeros_like(p) for name, p in self.params.items()}
        self.t = 0

    def set_params(self, params):
        """Copy parameters so different optimizers start from identical weights."""
        for name, value in params.items():
            self.params[name] = value.copy()
            setattr(self, name, self.params[name])
        self._init_adam()

    def clone(self):
        copy = NeuralNetwork(hidden_dims=self.sizes[1:-1], input_dim=self.sizes[0])
        copy.set_params(self.params)
        return copy

    @property
    def architecture(self):
        return "-".join(str(size) for size in self.sizes)

    # -------------------------------------------------------------- activations
    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def sigmoid(z):
        # Stable version: clip logits before exponentiating.
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    # ------------------------------------------------------ forward / backward
    def forward(self, X):
        self.activations = [X]
        self.zs = {}
        A = X
        for layer in range(1, self.n_layers):
            Z = A @ self.params[f"W{layer}"] + self.params[f"b{layer}"]
            self.zs[layer] = Z
            A = self.relu(Z)
            self.activations.append(A)

        Z_out = A @ self.params[f"W{self.n_layers}"] + self.params[f"b{self.n_layers}"]
        self.zs[self.n_layers] = Z_out
        out = self.sigmoid(Z_out)
        self.activations.append(out)
        return out

    @staticmethod
    def loss(y_true, y_pred):
        # Binary cross-entropy with numerical protection against log(0).
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1 - eps)
        m = y_true.shape[0]
        return -np.sum(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        ) / m

    def backward(self, X, y):
        m = X.shape[0]
        grads = {}
        dA = (self.activations[-1] - y) / m  # dZ_L for sigmoid + BCE

        for layer in reversed(range(1, self.n_layers + 1)):
            grads[f"W{layer}"] = self.activations[layer - 1].T @ dA
            grads[f"b{layer}"] = np.sum(dA, axis=0, keepdims=True)
            if layer > 1:
                dA = dA @ self.params[f"W{layer}"].T
                dA = dA * (self.zs[layer - 1] > 0)  # ReLU derivative

        return grads

    # -------------------------------------------------------------- optimizers
    def update(self, grads, lr):
        """Full-batch gradient descent (also the SGD per-mini-batch step)."""
        for name in self.params:
            self.params[name] -= lr * grads[name]

    def update_adam(self, grads, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        """Adam: manual biased moments, bias correction and parameter step."""
        self.t += 1
        for name in self.params:
            self.m[name] = beta1 * self.m[name] + (1 - beta1) * grads[name]
            self.v[name] = beta2 * self.v[name] + (1 - beta2) * (grads[name] ** 2)
            m_hat = self.m[name] / (1 - beta1 ** self.t)
            v_hat = self.v[name] / (1 - beta2 ** self.t)
            self.params[name] -= lr * m_hat / (np.sqrt(v_hat) + eps)

    # ------------------------------------------------------------------ training
    def fit(self, X, y, epochs=2000, lr=0.05, optimizer="gd", batch_size=32,
            X_val=None, y_val=None, beta1=0.9, beta2=0.999, eps=1e-8, seed=None):
        optimizer = optimizer.lower()
        if optimizer not in ("gd", "sgd", "adam"):
            raise ValueError("optimizer must be 'gd', 'sgd' or 'adam'")

        if optimizer == "adam":
            self._init_adam()
        rng = np.random.default_rng(SEED if seed is None else seed)

        history = History()
        history.val_loss = []
        history.val_accuracy = []
        history.val_f1 = []
        history.optimizer = optimizer
        history.epochs = epochs
        history.batch_size = len(X) if optimizer == "gd" else batch_size

        start = time.perf_counter()
        for _ in range(epochs):
            if optimizer == "gd":
                self.forward(X)
                self.update(self.backward(X, y), lr)
            else:
                indices = rng.permutation(len(X))
                for first in range(0, len(X), batch_size):
                    idx = indices[first:first + batch_size]
                    Xb, yb = X[idx], y[idx]
                    self.forward(Xb)
                    grads = self.backward(Xb, yb)
                    if optimizer == "adam":
                        self.update_adam(grads, lr, beta1, beta2, eps)
                    else:
                        self.update(grads, lr)

            history.append(self.loss(y, self.forward(X)))
            if X_val is not None and y_val is not None:
                p_val = self.forward(X_val)
                history.val_loss.append(self.loss(y_val, p_val))
                y_val_pred = (p_val >= 0.5).astype(int)
                history.val_accuracy.append(accuracy_score(y_val, y_val_pred))
                history.val_f1.append(f1_score(y_val, y_val_pred, zero_division=0))

        history.time = time.perf_counter() - start
        return history

    def predict(self, X):
        return (self.forward(X) >= 0.5).astype(int)


def gradient_checking(model, X, y, epsilon=1e-7):
    """Numerical check of the analytical gradients of every parameter."""
    model.forward(X)
    grads = model.backward(X, y)

    for name in model.params:
        param = getattr(model, name)
        grad_analytic = grads[name]
        grad_numeric = np.zeros_like(param)

        it = np.nditer(param, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index

            param[idx] += epsilon
            loss_plus = model.loss(y, model.forward(X))
            param[idx] -= 2 * epsilon
            loss_minus = model.loss(y, model.forward(X))
            param[idx] += epsilon

            grad_numeric[idx] = (loss_plus - loss_minus) / (2 * epsilon)

        denominator = np.linalg.norm(grad_analytic) + np.linalg.norm(grad_numeric)
        rel_error = (0.0 if denominator == 0 else
                     np.linalg.norm(grad_analytic - grad_numeric) / denominator)
        print(f"Gradient checking {model.architecture} - {name}: "
              f"relative error = {rel_error:.2e}")


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
ARCHITECTURES = {
    "2-4-1": (4,),
    "2-8-1": (8,),
    "2-32-16-1": (32, 16),
}
OPTIMIZERS = ("gd", "sgd", "adam")
EPOCHS = 400
LR = 0.05
BATCH_SIZE = 32


def run_experiments():
    """Same initialization per architecture; identical data for all optimizers."""
    results, history_rows, summary_rows = {}, [], []

    for arch_name, hidden_dims in ARCHITECTURES.items():
        base = NeuralNetwork(hidden_dims=hidden_dims, seed=SEED)
        for opt in OPTIMIZERS:
            model = base.clone()  # exact same starting weights for each optimizer
            hist = model.fit(
                X_fit, y_fit, epochs=EPOCHS, lr=LR, optimizer=opt,
                batch_size=BATCH_SIZE, X_val=X_val, y_val=y_val, seed=SEED
            )

            train_acc = accuracy_score(y_fit, model.predict(X_fit))
            test_pred = model.predict(X_test)
            summary = {
                "architecture": arch_name,
                "optimizer": opt,
                "time": hist.time,
                "final_loss": hist[-1],
                "val_loss": hist.val_loss[-1],
                "train_accuracy": train_acc,
                "val_accuracy": hist.val_accuracy[-1],
                "val_f1": hist.val_f1[-1],
                "test_accuracy": accuracy_score(y_test, test_pred),
                "test_f1": f1_score(y_test, test_pred, zero_division=0),
                "generalization_gap": train_acc - hist.val_accuracy[-1],
            }
            results[(arch_name, opt)] = {
                "model": model, "history": hist, "summary": summary
            }

            for epoch, (loss, v_loss, v_acc, v_f1) in enumerate(
                zip(hist, hist.val_loss, hist.val_accuracy, hist.val_f1), start=1
            ):
                history_rows.append(
                    [arch_name, opt, epoch, loss, v_loss, v_acc, v_f1]
                )
            summary_rows.append([
                arch_name, opt, f"{hist.time:.4f}", f"{summary['final_loss']:.6f}",
                f"{summary['val_loss']:.6f}", f"{train_acc:.4f}",
                f"{summary['val_accuracy']:.4f}", f"{summary['val_f1']:.4f}",
                f"{summary['test_accuracy']:.4f}", f"{summary['test_f1']:.4f}",
                f"{summary['generalization_gap']:.4f}",
            ])

    with open(os.path.join(RESULTS_DIR, "optimizer_history.csv"), "w",
              newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["architecture", "optimizer", "epoch", "train_loss",
                         "val_loss", "val_accuracy", "val_f1"])
        writer.writerows(history_rows)

    with open(os.path.join(RESULTS_DIR, "experiment_summary.csv"), "w",
              newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["architecture", "optimizer", "time_s", "final_train_loss",
                         "val_loss", "train_accuracy", "val_accuracy", "val_f1",
                         "test_accuracy", "test_f1", "generalization_gap"])
        writer.writerows(summary_rows)

    return results


def print_report(results):
    print("\n--- Optimizer comparison (2-8-1, same initialization) ---")
    print(f"{'optimizer':<10} {'time [s]':>9} {'loss':>9} {'val_acc':>9} "
          f"{'val_F1':>9} {'test_F1':>9}")
    for opt in OPTIMIZERS:
        s = results[("2-8-1", opt)]["summary"]
        print(f"{opt:<10} {s['time']:>9.3f} {s['final_loss']:>9.4f} "
              f"{s['val_accuracy']:>9.4f} {s['val_f1']:>9.4f} "
              f"{s['test_f1']:>9.4f}")

    print("\n--- Architecture comparison (best optimizer per architecture) ---")
    best_per_arch = {}
    for arch in ARCHITECTURES:
        candidates = [results[(arch, opt)]["summary"] for opt in OPTIMIZERS]
        best = max(candidates, key=lambda s: s["val_f1"])
        best_per_arch[arch] = best
        print(f"{arch:<11} best val_F1={best['val_f1']:.4f} "
              f"({best['optimizer']})  gap={best['generalization_gap']:+.4f} "
              f"test_F1={best['test_f1']:.4f}")

    f1_small = best_per_arch["2-4-1"]["val_f1"]
    f1_medium = best_per_arch["2-8-1"]["val_f1"]
    f1_large = best_per_arch["2-32-16-1"]["val_f1"]
    if not (f1_small < f1_medium < f1_large):
        print("\nConclusion: increasing capacity does NOT always improve "
              "generalization; the largest network does not reach the best "
              "val_F1 and shows a larger train-validation gap.")
    else:
        print("\nConclusion: in this run generalization improved with capacity, "
              "but the trend is not guaranteed; monitor the train-validation "
              "gap when scaling the model.")


def make_plots(results):
    # Optimizer comparison: loss and validation F1 per epoch.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for opt in OPTIMIZERS:
        hist = results[("2-8-1", opt)]["history"]
        axes[0].plot(hist, label=opt)
        axes[1].plot(hist.val_f1, label=opt)
    axes[0].set_title("Training loss per epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary cross-entropy")
    axes[1].set_title("Validation F1 per epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "optimizer_comparison.png"))
    plt.close(fig)

    # Architecture comparison: validation F1 and generalization gap.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    arch_names = list(ARCHITECTURES)
    x = np.arange(len(arch_names))
    width = 0.25
    for i, opt in enumerate(OPTIMIZERS):
        f1_values = [results[(arch, opt)]["summary"]["val_f1"] for arch in arch_names]
        gaps = [results[(arch, opt)]["summary"]["generalization_gap"]
                for arch in arch_names]
        axes[0].bar(x + (i - 1) * width, f1_values, width, label=opt)
        axes[1].bar(x + (i - 1) * width, gaps, width, label=opt)
    axes[0].set_title("Validation F1 by architecture")
    axes[0].set_ylabel("F1")
    axes[1].set_title("Train-validation generalization gap")
    axes[1].set_ylabel("Train accuracy - val accuracy")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(arch_names)
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "architecture_comparison.png"))
    plt.close(fig)

    # Original challenge figure: loss curve and non-linear decision boundary.
    best = results[("2-8-1", "gd")]
    model, hist = best["model"], best["history"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(hist, label="train")
    axes[0].plot(hist.val_loss, label="validation")
    axes[0].set_title("Training loss (2-8-1, GD)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary cross-entropy")
    axes[0].legend()

    x_min, x_max = X_fit[:, 0].min() - 0.5, X_fit[:, 0].max() + 0.5
    y_min, y_max = X_fit[:, 1].min() - 0.5, X_fit[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid).reshape(xx.shape)
    axes[1].contourf(xx, yy, Z, alpha=0.3)
    axes[1].scatter(X_fit[:, 0], X_fit[:, 1], c=y_fit.ravel(), edgecolors="k", s=20)
    axes[1].set_title("Non-linear decision boundary")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "training_results.png"))
    plt.close(fig)


def main():
    results = run_experiments()
    print_report(results)
    make_plots(results)

    # Gradient checking on the default 2-8-1 architecture.
    gradient_checking(results[("2-8-1", "gd")]["model"], X_fit[:10], y_fit[:10])


if __name__ == "__main__":
    main()

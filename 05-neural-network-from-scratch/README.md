# 05 - Neural Network From Scratch: Multilayer Perceptron (MLP)

## Context
After working with clustering, anomaly detection, text classification, and planning with A*, it is time to implement a neural network from scratch without TensorFlow, PyTorch, or Keras.

You will build a multilayer perceptron (MLP) to solve a non-linear classification problem. You will manually implement forward propagation, loss calculation, error backpropagation, and weight updates. The goal is to understand what happens mathematically inside a neural network, rather than simply calling `model.fit()`.

## Mission
Train an MLP on the synthetic `make_moons` dataset, where a linear decision boundary cannot separate the two interlaced classes. Then compare three manually implemented optimizers under identical initialization, and study whether increasing network capacity always improves generalization.

## Learning Objectives
- Build a Multi-Layer Perceptron (MLP) using only NumPy.
- Implement ReLU and sigmoid activations with numerical stability.
- Derive and implement backpropagation manually using the chain rule.
- Implement full-batch GD, mini-batch SGD and Adam without an autograd library.
- Compare convergence speed using per-epoch loss, validation metrics and training time.
- Evaluate non-linear classification with accuracy and F1-score.
- Analyze whether increasing capacity always improves generalization.
- Visualize decision boundaries and training loss evolution.
- Perform numerical gradient checking to validate the analytical gradients.

## Theoretical Foundation
For a network with one hidden layer, forward propagation is:

\[
Z_1 = X W_1 + b_1, \qquad A_1 = \operatorname{ReLU}(Z_1)
\]
\[
Z_2 = A_1 W_2 + b_2, \qquad \hat{y} = \sigma(Z_2)
\]

The binary cross-entropy loss for `m` observations is:

\[
L = -\frac{1}{m}\sum_{i=1}^{m}\left[y_i\log\hat{y}_i + (1-y_i)\log(1-\hat{y}_i)\right]
\]

With a sigmoid output, the gradient with respect to the logits simplifies to:

\[
dZ_2 = \frac{\hat{y} - y}{m}
\]

The chain rule then gives `dW2 = A1.T @ dZ2`, `db2 = sum(dZ2)`, `dZ1 = (dZ2 @ W2.T) * (Z1 > 0)`, `dW1 = X.T @ dZ1` and `db1 = sum(dZ1)`.

### Optimizers
Three optimizers are implemented manually and compared under the **same weight initialization**:

1. **Full-batch gradient descent (GD)**: one update per epoch using all training samples.
   \[
   \theta \leftarrow \theta - \eta \nabla_\theta L
   \]
2. **Mini-batch SGD**: the training set is shuffled each epoch and split into batches of 32 samples; one update is performed per batch.
3. **Adam**: mini-batch updates with biased first and second moments, bias correction and a per-parameter step size.
   \[
   m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t,\qquad
   v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2
   \]
   \[
   \hat m_t = \frac{m_t}{1-\beta_1^t},\qquad
   \hat v_t = \frac{v_t}{1-\beta_2^t},\qquad
   \theta \leftarrow \theta - \eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
   \]
   with \(\beta_1=0.9\), \(\beta_2=0.999\) and \(\epsilon=10^{-8}\).

All optimizers use the same learning rate (\(\eta = 0.05\)) and 400 epochs. Each epoch records the training loss, validation loss, validation accuracy and validation F1. Wall-clock training time is also measured so that convergence speed can be compared in terms of both epochs and computation.

### Capacity Study
In addition to the default `2-8-1` network, two more architectures are evaluated with the same protocol and the same seed:

| Name | Hidden layers |
| :--- | :--- |
| `2-4-1` | 4 |
| `2-8-1` | 8 |
| `2-32-16-1` | 32, 16 |

The goal is to check whether a larger network always generalizes better. The train-validation accuracy gap and validation F1 are used to discuss overfitting instead of assuming that more capacity is better.

## Experiment Flow
1. **Dataset Generation**: `make_moons`, train-test split, `StandardScaler` scaling and an independent validation split carved out of the training data. `X_test` remains untouched until final evaluation.
2. **Architecture Initialization**: He initialization of weights and zero biases; every optimizer starts from a copy of the same weights.
3. **Forward Propagation**: activations through hidden and output layers.
4. **Loss Calculation**: binary cross-entropy with `eps = 1e-12` clipping.
5. **Backpropagation**: chain rule and gradients for every layer.
6. **Optimization**: parameter updates with GD, mini-batch SGD or Adam.
7. **Evaluation & Visualization**: per-epoch loss and validation metrics, training time, decision boundary and comparison charts.

## Results

### Optimizer comparison (architecture `2-8-1`, identical initialization)

| Optimizer | Time (s) | Final train loss | Validation F1 | Test F1 |
| :--- | ---: | ---: | ---: | ---: |
| GD | 0.825 | 0.2960 | 0.8302 | 0.9246 |
| SGD (batch 32) | 1.217 | 0.0767 | **0.9554** | **0.9901** |
| Adam (batch 32) | 1.514 | **0.0623** | **0.9554** | 0.9851 |

GD performs only one update per epoch, whereas SGD and Adam perform about 25 mini-batch updates per epoch. Their faster convergence per epoch therefore comes with extra computation; the time column keeps the comparison honest.

### Capacity study (best optimizer per architecture)

| Architecture | Best validation F1 | Optimizer | Train-val gap | Test F1 |
| :--- | ---: | :--- | ---: | ---: |
| `2-4-1` | 0.9554 | sgd | +0.0141 | 0.9901 |
| `2-8-1` | 0.9554 | sgd | +0.0172 | 0.9901 |
| `2-32-16-1` | 0.9487 | sgd | +0.0297 | 0.9901 |

The largest network does **not** obtain the best validation F1 and shows the largest train-validation gap. Increasing capacity does not always improve generalization.

Gradient checking on all four parameters of the `2-8-1` network gives relative errors between 1.43e-09 and 3.83e-09, well below the 1e-5 threshold.

Artifacts in this directory's `results/` folder:

| File | Content |
| :--- | :--- |
| `optimizer_history.csv` | Per-epoch train loss, validation loss, accuracy and F1 for every configuration. |
| `experiment_summary.csv` | Training time and final train/validation/test metrics for every configuration. |
| `optimizer_comparison.png` | Loss and validation F1 curves for the three optimizers on `2-8-1`. |
| `architecture_comparison.png` | Validation F1 and generalization gap per architecture. |
| `training_results.png` | Loss curve and non-linear decision boundary for the baseline `2-8-1` network. |

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

The script trains the nine configurations (3 architectures × 3 optimizers), prints the optimizer and architecture comparison tables, saves the artifacts listed above and runs numerical gradient checking.

## Conclusions and Limitations
- Sigmoid plus binary cross-entropy gives a simple logit gradient (`ŷ - y`) / m, which keeps backpropagation compact and numerically well behaved.
- Mini-batch SGD and Adam converge much faster than full-batch GD in this setup, but they also take far more optimizer steps per epoch. Comparing both epochs and wall-clock time is necessary for a fair interpretation.
- The same initialization was used for every optimizer, and the same seed for every architecture, so differences are attributable to the optimizer or capacity rather than to random starts.
- More capacity did not help here: `2-32-16-1` needed more epochs and showed a larger generalization gap. The validation set has only 160 observations, so these differences are indicative rather than statistically robust; repeating the study with several seeds would strengthen the conclusion.
- The dataset is synthetic and low-dimensional. These results illustrate the mechanics of training but do not establish behavior on large or noisy real-world data.

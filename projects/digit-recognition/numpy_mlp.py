"""Two-layer MLP (784 -> 64 -> 10) for MNIST, written with NumPy only.

Forward pass, backpropagation and gradient descent are implemented by hand.
Run ``python numpy_mlp.py`` to train on MNIST, print the test metrics, save the
weights to ``artifacts/digit_mlp.npz`` and the figures to ``assets/``.
The weights shipped in ``artifacts/`` come from such a run (2000 iterations,
learning rate 0.1), converted from a pickled ``.npy`` to a plain ``.npz``.

Conventions: examples are stored as columns, so ``X`` has shape (features, m)
and ``Y`` has shape (classes, m).
"""

from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
WEIGHTS = HERE / "artifacts" / "digit_mlp.npz"


def preprocess(X):
    """Flatten images into columns and scale pixels to [0, 1].

    Args:
        X: array of shape (m, height, width) with values in [0, 255].

    Returns:
        Array of shape (height * width, m).
    """
    return X.reshape(len(X), -1).T / 255


def one_hot(y, n_classes):
    """One-hot encode integer labels as columns, shape (n_classes, m)."""
    Y = np.zeros((n_classes, y.size))
    Y[y, np.arange(y.size)] = 1
    return Y


def init_parameters(n_in, n_hidden, n_out):
    """Draw small Gaussian weights and zero biases (seeded for reproducibility)."""
    np.random.seed(42)
    return {
        "W1": np.random.randn(n_hidden, n_in) * 0.01,
        "B1": np.zeros((n_hidden, 1)),
        "W2": np.random.randn(n_out, n_hidden) * 0.01,
        "B2": np.zeros((n_out, 1)),
    }


def relu(Z):
    """Rectified linear unit."""
    return np.maximum(0, Z)


def softmax(Z):
    """Column-wise softmax, shifted by the column maximum for stability."""
    exp = np.exp(Z - Z.max(axis=0, keepdims=True))
    return exp / exp.sum(axis=0, keepdims=True)


def forward_propagation(X, params):
    """Compute class probabilities and the cache needed by backpropagation."""
    Z1 = params["W1"] @ X + params["B1"]
    A1 = relu(Z1)
    A2 = softmax(params["W2"] @ A1 + params["B2"])
    return A2, {"A0": X, "Z1": Z1, "A1": A1, "A2": A2}


def cost(A2, Y):
    """Cross-entropy loss averaged over the m examples."""
    return -np.sum(Y * np.log(A2 + 1e-8)) / Y.shape[1]


def backward_propagation(params, cache, Y):
    """Gradients of the cost with respect to every parameter."""
    m = Y.shape[1]
    dZ2 = cache["A2"] - Y
    dZ1 = (params["W2"].T @ dZ2) * (cache["Z1"] > 0)
    return {
        "W1": dZ1 @ cache["A0"].T / m,
        "B1": dZ1.sum(axis=1, keepdims=True) / m,
        "W2": dZ2 @ cache["A1"].T / m,
        "B2": dZ2.sum(axis=1, keepdims=True) / m,
    }


def update_parameters(params, grads, learning_rate):
    """One gradient descent step."""
    return {key: params[key] - learning_rate * grads[key] for key in params}


def train(X, Y, n_hidden=64, iterations=2000, learning_rate=0.1):
    """Full-batch gradient descent.

    Returns:
        The trained parameters and the cost at every iteration.
    """
    params = init_parameters(X.shape[0], n_hidden, Y.shape[0])
    costs = []
    for i in range(iterations):
        A2, cache = forward_propagation(X, params)
        costs.append(cost(A2, Y))
        params = update_parameters(
            params, backward_propagation(params, cache, Y), learning_rate
        )
        if i % 100 == 0:
            print(f"iteration {i:4d}  cost {costs[-1]:.4f}")
    return params, costs


def predict(X, params):
    """Most likely class for every column of X."""
    return np.argmax(forward_propagation(X, params)[0], axis=0)


def confusion_matrix(y_true, y_pred, n_classes=10):
    """Counts with rows = true class and columns = predicted class."""
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(matrix, (y_true, y_pred), 1)
    return matrix


def classification_metrics(matrix):
    """Per-class precision, recall and F1 from a confusion matrix."""
    tp = np.diag(matrix)
    precision = tp / matrix.sum(axis=0)
    recall = tp / matrix.sum(axis=1)
    return precision, recall, 2 * precision * recall / (precision + recall)


def save_parameters(params, path=WEIGHTS):
    """Save the weights as a pickle-free ``.npz`` archive."""
    path.parent.mkdir(exist_ok=True)
    np.savez(path, **params)


def load_parameters(path=WEIGHTS):
    """Load weights saved by :func:`save_parameters`."""
    return dict(np.load(path))


def load_mnist(root="data"):
    """Download MNIST with torchvision and return it as NumPy arrays.

    Returns:
        ``X_train, y_train, X_test, y_test`` with images of shape (m, 28, 28).
    """
    from torchvision.datasets import MNIST

    train = MNIST(root, train=True, download=True)
    test = MNIST(root, train=False, download=True)
    return (
        train.data.numpy(),
        train.targets.numpy(),
        test.data.numpy(),
        test.targets.numpy(),
    )


def plot_cost(costs, path=HERE / "assets" / "mlp_cost_curve.png"):
    """Save the training-cost curve."""
    import matplotlib.pyplot as plt

    path.parent.mkdir(exist_ok=True)
    fig, ax = plt.subplots()
    ax.plot(costs)
    ax.set(xlabel="Iteration", ylabel="Cross-entropy cost", title="Training cost")
    fig.savefig(path, dpi=120, bbox_inches="tight")


def plot_confusion(matrix, path=HERE / "assets" / "mlp_confusion_matrix.png"):
    """Save the confusion matrix as an annotated heatmap."""
    import matplotlib.pyplot as plt

    path.parent.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(matrix, cmap="Blues")
    for (i, j), count in np.ndenumerate(matrix):
        ax.text(j, i, count, ha="center", va="center", fontsize=8)
    ax.set(
        xticks=range(10),
        yticks=range(10),
        xlabel="Predicted",
        ylabel="True label",
        title="Confusion matrix (10,000 MNIST test images)",
    )
    fig.savefig(path, dpi=120, bbox_inches="tight")


def main():
    """Train on MNIST, evaluate on the test set and save weights and figures."""
    X_train, y_train, X_test, y_test = load_mnist()
    params, costs = train(preprocess(X_train), one_hot(y_train, 10))
    matrix = confusion_matrix(y_test, predict(preprocess(X_test), params))
    precision, recall, f1 = classification_metrics(matrix)
    print(f"test accuracy   {np.trace(matrix) / matrix.sum():.4f}")
    print(f"macro precision {precision.mean():.4f}")
    print(f"macro recall    {recall.mean():.4f}")
    print(f"macro F1        {f1.mean():.4f}")
    save_parameters(params)
    plot_cost(costs)
    plot_confusion(matrix)


if __name__ == "__main__":
    main()

import numpy as np
from numpy_mlp import (
    backward_propagation,
    classification_metrics,
    confusion_matrix,
    cost,
    forward_propagation,
    init_parameters,
    load_parameters,
    one_hot,
    predict,
    preprocess,
    softmax,
    train,
)

# Confusion matrix of the reference run (MNIST test set, 2000 iterations, lr 0.1).
REFERENCE = np.array(
    [
        [961, 0, 1, 1, 0, 5, 9, 1, 1, 1],
        [0, 1110, 4, 2, 0, 1, 3, 2, 13, 0],
        [7, 7, 955, 15, 9, 2, 11, 8, 15, 3],
        [1, 0, 16, 940, 0, 20, 2, 10, 15, 6],
        [1, 2, 4, 0, 930, 0, 11, 3, 3, 28],
        [9, 3, 2, 26, 10, 800, 13, 5, 18, 6],
        [11, 3, 4, 1, 8, 7, 919, 2, 3, 0],
        [2, 8, 23, 7, 7, 1, 0, 961, 1, 18],
        [6, 6, 5, 17, 9, 14, 12, 9, 891, 5],
        [10, 6, 1, 10, 26, 7, 1, 9, 5, 934],
    ]
)


def blobs(m=300, seed=0):
    """Three well-separated Gaussian classes in 20 dimensions."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, m)
    X = rng.normal(size=(m, 20)) + 3 * np.eye(3)[y] @ rng.normal(size=(3, 20))
    return X.T, y


def test_preprocess_and_one_hot_shapes():
    images = np.full((5, 28, 28), 255)
    assert preprocess(images).shape == (784, 5)
    assert preprocess(images).max() == 1
    assert one_hot(np.array([0, 2]), 3).tolist() == [[1, 0], [0, 0], [0, 1]]


def test_softmax_columns_sum_to_one_even_for_large_logits():
    probabilities = softmax(np.array([[1000.0, 1.0], [1001.0, 2.0]]))
    assert np.allclose(probabilities.sum(axis=0), 1)


def test_backpropagation_matches_numerical_gradient():
    X, y = blobs(m=20)
    Y = one_hot(y, 3)
    params = init_parameters(20, 5, 3)
    params["W1"] = np.random.default_rng(1).normal(size=(5, 20)) * 0.1
    grads = backward_propagation(params, forward_propagation(X, params)[1], Y)

    eps = 1e-6
    for key, index in [("W1", (2, 3)), ("W2", (1, 4)), ("B1", (0, 0)), ("B2", (2, 0))]:
        plus = {k: v.copy() for k, v in params.items()}
        minus = {k: v.copy() for k, v in params.items()}
        plus[key][index] += eps
        minus[key][index] -= eps
        numerical = (
            cost(forward_propagation(X, plus)[0], Y)
            - cost(forward_propagation(X, minus)[0], Y)
        ) / (2 * eps)
        assert np.isclose(grads[key][index], numerical, atol=1e-6)


def test_training_learns_separable_classes():
    X, y = blobs()
    params, costs = train(X, one_hot(y, 3), n_hidden=16, iterations=200)
    assert costs[-1] < costs[0]
    assert (predict(X, params) == y).mean() > 0.95


def test_metrics_reproduce_the_reference_run():
    precision, recall, f1 = classification_metrics(REFERENCE)
    assert np.trace(REFERENCE) / REFERENCE.sum() == 0.9401
    assert round(precision.mean(), 4) == 0.9395
    assert round(recall.mean(), 4) == 0.9393
    assert round(f1.mean(), 4) == 0.9393


def test_confusion_matrix_counts():
    matrix = confusion_matrix(np.array([0, 1, 1, 2]), np.array([0, 1, 2, 2]), 3)
    assert matrix.tolist() == [[1, 0, 0], [0, 1, 1], [0, 0, 1]]


def test_shipped_weights_have_the_expected_shapes():
    params = load_parameters()
    assert {k: v.shape for k, v in params.items()} == {
        "W1": (64, 784),
        "B1": (64, 1),
        "W2": (10, 64),
        "B2": (10, 1),
    }

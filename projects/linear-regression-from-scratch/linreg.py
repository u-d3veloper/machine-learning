"""Linear regression by batch gradient descent, implemented with NumPy only.

Model: ``y_hat = X @ theta`` where the first column of ``X`` is all ones (bias).
Cost:  ``J(theta) = 1 / (2m) * sum((X @ theta - y) ** 2)``.
Update: ``theta <- theta - lr / m * X.T @ (X @ theta - y)``.
"""

import numpy as np


def min_max_scale(X):
    """Rescale every column to [0, 1] so one learning rate suits all features."""
    lo, hi = X.min(axis=0), X.max(axis=0)
    return (X - lo) / (hi - lo)


def add_bias(X):
    """Prepend a column of ones for the intercept."""
    return np.c_[np.ones(len(X)), X]


def cost(X, y, theta):
    """Half the mean squared error."""
    residual = X @ theta - y
    return residual @ residual / (2 * len(y))


def gradient_descent(X, y, learning_rate=0.5, iterations=1000):
    """Minimise the cost with batch gradient descent.

    Args:
        X: design matrix of shape (m, n), bias column included.
        y: targets of shape (m,).
        learning_rate: step size.
        iterations: number of updates.

    Returns:
        The fitted ``theta`` of shape (n,) and the cost after each iteration.
    """
    theta = np.zeros(X.shape[1])
    history = np.empty(iterations)
    for i in range(iterations):
        theta = theta - learning_rate / len(y) * X.T @ (X @ theta - y)
        history[i] = cost(X, y, theta)
    return theta, history


def normal_equation(X, y):
    """Closed-form least-squares solution, used as ground truth for the descent."""
    return np.linalg.lstsq(X, y, rcond=None)[0]

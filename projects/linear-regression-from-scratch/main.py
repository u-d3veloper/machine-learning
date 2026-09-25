"""Fit synthetic house prices with gradient descent, checked against the closed form.

Two fits: price from size alone, then from size and number of bedrooms.
Figures are written to ``assets/``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from linreg import add_bias, gradient_descent, min_max_scale, normal_equation

ASSETS = Path(__file__).parent / "assets"


def make_houses(n=200, seed=0):
    """Synthetic houses: price = 50k + 3k per m2 + 10k per bedroom + noise."""
    rng = np.random.default_rng(seed)
    size = rng.uniform(30, 200, n)
    bedrooms = rng.integers(1, 6, n)
    price = 50_000 + 3_000 * size + 10_000 * bedrooms + rng.normal(0, 15_000, n)
    return np.c_[size, bedrooms], price


def fit(X, y, name):
    """Scale, run gradient descent, compare with the normal equation and report R2."""
    A = add_bias(min_max_scale(X))
    theta, history = gradient_descent(A, y)
    exact = normal_equation(A, y)
    r2 = 1 - np.sum((A @ theta - y) ** 2) / np.sum((y - y.mean()) ** 2)
    print(f"{name}: R2 {r2:.3f}, final cost {history[-1]:,.0f}")
    print(f"  gradient descent theta {theta.round(0)}")
    print(f"  normal equation theta  {exact.round(0)}")
    return A @ theta, history


def main():
    X, y = make_houses()
    fitted_single, history_single = fit(X[:, :1], y, "price ~ size")
    _, history_multi = fit(X, y, "price ~ size + bedrooms")

    ASSETS.mkdir(exist_ok=True)
    order = np.argsort(X[:, 0])
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], y, s=12, alpha=0.6, label="houses")
    ax.plot(X[order, 0], fitted_single[order], color="C3", label="gradient descent fit")
    ax.set(xlabel="Size (m2)", ylabel="Price", title="Price from size")
    ax.legend()
    fig.savefig(ASSETS / "single_variable_fit.png", dpi=120, bbox_inches="tight")

    fig, ax = plt.subplots()
    ax.plot(history_single, label="size")
    ax.plot(history_multi, label="size + bedrooms")
    ax.set(xlabel="Iteration", ylabel="Cost J", yscale="log", title="Convergence")
    ax.legend()
    fig.savefig(ASSETS / "cost_history.png", dpi=120, bbox_inches="tight")


if __name__ == "__main__":
    main()

import numpy as np
from linreg import add_bias, cost, gradient_descent, min_max_scale, normal_equation
from main import make_houses


def test_min_max_scale_maps_columns_to_unit_interval():
    scaled = min_max_scale(np.array([[1.0, 10], [3, 20], [5, 30]]))
    assert scaled.min(axis=0).tolist() == [0, 0]
    assert scaled.max(axis=0).tolist() == [1, 1]


def test_cost_of_a_perfect_fit_is_zero():
    X = add_bias(np.array([[1.0], [2.0], [3.0]]))
    assert cost(X, np.array([3.0, 5.0, 7.0]), np.array([1.0, 2.0])) == 0


def test_gradient_descent_matches_the_normal_equation():
    X, y = make_houses()
    A = add_bias(min_max_scale(X))
    theta, history = gradient_descent(A, y)
    assert np.allclose(theta, normal_equation(A, y), rtol=1e-4)
    assert history[-1] < history[0] / 10

import numpy as np
import pandas as pd
from main import evaluate, load_data, make_models, split, tukey_outliers


def synthetic(n=400, seed=0):
    """A wine-like frame: 'good' (quality 7) mostly when feature a is high."""
    rng = np.random.default_rng(seed)
    data = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
    data["quality"] = np.where(data["a"] + rng.normal(scale=0.5, size=n) > 1, 7, 5)
    return data


def test_split_is_stratified_and_binary():
    data = synthetic()
    X_train, X_test, y_train, y_test = split(data)
    assert set(y_train) == {0, 1}
    assert "quality" not in X_train
    assert abs(y_train.mean() - y_test.mean()) < 0.02


def test_tukey_flags_an_extreme_value():
    frame = pd.DataFrame({"x": [1, 2, 3, 4, 5, 100]})
    assert tukey_outliers(frame)["x"] == 1 / 6


def test_evaluate_scores_every_model():
    X_train, X_test, y_train, y_test = split(synthetic())
    models = make_models()
    results = evaluate(models, X_train, X_test, y_train, y_test)
    assert list(results.index) == list(models)
    assert results.to_numpy().min() >= 0 and results.to_numpy().max() <= 1
    assert results["f1"].max() > 0.7


def test_load_data_reads_a_local_semicolon_csv(tmp_path):
    path = tmp_path / "wine.csv"
    path.write_text("a;quality\n1;5\n2;7\n")
    assert load_data(path).shape == (2, 2)

"""Wine quality: predict whether a red wine is good (quality >= 7) from its chemistry.

Only about 14% of the wines are good, so accuracy alone is misleading: models are
compared on precision, recall and F1 of the "good" class as well. Figures are
written to ``assets/``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
HERE = Path(__file__).parent
DATA = HERE / "data" / "winequality-red.csv"
ASSETS = HERE / "assets"


def load_data(path=DATA):
    """Load the red wine dataset, downloading it from UCI on first use."""
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        pd.read_csv(URL, sep=";").to_csv(path, sep=";", index=False)
    return pd.read_csv(path, sep=";")


def split(data, seed=42):
    """Binary target (quality >= 7) and a stratified 80/20 train/test split."""
    X = data.drop(columns="quality")
    y = (data["quality"] >= 7).astype(int)
    return train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)


def tukey_outliers(X):
    """Fraction of rows outside [Q1 - 1.5 IQR, Q3 + 1.5 IQR], per feature."""
    q1, q3 = X.quantile(0.25), X.quantile(0.75)
    step = 1.5 * (q3 - q1)
    return ((X < q1 - step) | (X > q3 + step)).mean()


def make_models():
    """Scaled pipelines: linear models, an SVM, and class-weighted variants."""
    classifiers = {
        "Logistic regression": LogisticRegression(max_iter=1000),
        "Logistic regression (balanced)": LogisticRegression(
            max_iter=1000, class_weight="balanced"
        ),
        "Ridge classifier": RidgeClassifier(),
        "SGD classifier": SGDClassifier(random_state=42),
        "SVM (RBF)": SVC(),
        "SVM (RBF, balanced)": SVC(class_weight="balanced"),
    }
    return {
        name: make_pipeline(StandardScaler(), clf) for name, clf in classifiers.items()
    }


def evaluate(models, X_train, X_test, y_train, y_test):
    """Fit every model and return a metrics table (positive class = good wine)."""
    rows = {}
    for name, model in models.items():
        predictions = model.fit(X_train, y_train).predict(X_test)
        rows[name] = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions),
            "f1": f1_score(y_test, predictions),
        }
    return pd.DataFrame(rows).T


def main():
    data = load_data()
    print(f"shape {data.shape}, missing values {data.isna().sum().sum()}")

    X_train, X_test, y_train, y_test = split(data)
    print(f"good wines: {y_test.mean():.1%} of the test set")
    print(f"majority-class accuracy: {1 - y_test.mean():.3f}")
    print("\nOutlier share per feature (Tukey, kept in the data):")
    print(tukey_outliers(data.drop(columns="quality")).round(3).sort_values().tail(3))

    models = make_models()
    results = evaluate(models, X_train, X_test, y_train, y_test)
    print("\n", results.round(3).sort_values("f1", ascending=False), sep="")

    best = results["f1"].idxmax()
    ASSETS.mkdir(exist_ok=True)
    ConfusionMatrixDisplay.from_estimator(models[best], X_test, y_test, cmap="Blues")
    plt.title(f"{best}: test set")
    plt.savefig(ASSETS / "confusion_matrix.png", dpi=120, bbox_inches="tight")


if __name__ == "__main__":
    main()

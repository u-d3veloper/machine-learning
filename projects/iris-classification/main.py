"""Iris classification: spot-check five classifiers, then evaluate the best one.

Classic workflow: look at the data, compare algorithms with stratified 10-fold
cross-validation on a training split, then evaluate the winner once on a
held-out validation split. Figures are written to ``assets/``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ASSETS = Path(__file__).parent / "assets"
MODELS = {
    "LDA": LinearDiscriminantAnalysis(),
    "KNN": KNeighborsClassifier(),
    "CART": DecisionTreeClassifier(random_state=1),
    "NB": GaussianNB(),
    "SVM": SVC(gamma="auto"),
}


def load_dataset():
    """Return Iris as a DataFrame with a readable ``species`` column."""
    iris = load_iris(as_frame=True)
    data = iris.frame.drop(columns="target")
    data["species"] = iris.target_names[iris.target.to_numpy()]
    return data


def split_dataset(data, seed=1):
    """Hold out 20% of the rows (stratified by species) for the final check."""
    X, y = data.drop(columns="species"), data["species"]
    return train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)


def spot_check(X_train, y_train):
    """Cross-validated accuracy of every candidate.

    Returns:
        ``{model name: array of 10 fold accuracies}``.
    """
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    return {
        name: cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        for name, model in MODELS.items()
    }


def plot_comparison(scores, path=ASSETS / "algorithm_comparison.png"):
    """Box plot of the cross-validation scores of every model."""
    path.parent.mkdir(exist_ok=True)
    fig, ax = plt.subplots()
    ax.boxplot(scores.values(), tick_labels=scores.keys())
    ax.set(ylabel="Accuracy (10-fold CV)", title="Algorithm comparison")
    fig.savefig(path, dpi=120, bbox_inches="tight")


def plot_scatter_matrix(data, path=ASSETS / "scatter_matrix.png"):
    """Pairwise scatter plots of the four measurements, coloured by species."""
    path.parent.mkdir(exist_ok=True)
    scatter_matrix(
        data.drop(columns="species"),
        c=data["species"].astype("category").cat.codes,
        figsize=(9, 9),
    )
    plt.savefig(path, dpi=120, bbox_inches="tight")


def main():
    data = load_dataset()
    print(data.describe(), data["species"].value_counts(), sep="\n\n")

    X_train, X_val, y_train, y_val = split_dataset(data)
    scores = spot_check(X_train, y_train)
    for name, fold_scores in scores.items():
        print(f"{name}: {fold_scores.mean():.3f} ({fold_scores.std():.3f})")

    best = max(scores, key=lambda name: scores[name].mean())
    predictions = MODELS[best].fit(X_train, y_train).predict(X_val)
    accuracy = accuracy_score(y_val, predictions)
    print(f"\nBest model: {best}, validation accuracy {accuracy:.3f}")
    print(confusion_matrix(y_val, predictions))
    print(classification_report(y_val, predictions))

    plot_comparison(scores)
    plot_scatter_matrix(data)


if __name__ == "__main__":
    main()

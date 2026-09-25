from main import MODELS, load_dataset, split_dataset, spot_check


def test_dataset_and_stratified_split():
    data = load_dataset()
    assert data.shape == (150, 5)
    X_train, X_val, y_train, y_val = split_dataset(data)
    assert len(X_train) == 120 and len(X_val) == 30
    assert y_val.value_counts().tolist() == [10, 10, 10]


def test_every_candidate_is_reasonable():
    X_train, _, y_train, _ = split_dataset(load_dataset())
    scores = spot_check(X_train, y_train)
    assert scores.keys() == MODELS.keys()
    assert all(fold_scores.mean() > 0.9 for fold_scores in scores.values())

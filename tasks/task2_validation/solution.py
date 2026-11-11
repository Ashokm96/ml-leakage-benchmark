import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit


def run():
    """Evaluate a time-ordered signal without shuffling its observations."""
    rng = np.random.RandomState(42)
    X = np.arange(200).reshape(-1, 1).astype(float)
    y = X.flatten() * 2 + rng.randn(200) * 5
    splitter = TimeSeriesSplit(n_splits=3)
    scores = []
    for train_idx, test_idx in splitter.split(X):
        model = LinearRegression()
        model.fit(X[train_idx], y[train_idx])
        scores.append(float(model.score(X[test_idx], y[test_idx])))
    return sum(scores) / len(scores)

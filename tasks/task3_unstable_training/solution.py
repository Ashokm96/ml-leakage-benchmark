import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def run():
    """Train a reproducible classifier with numerically stable preprocessing."""
    rng = np.random.RandomState(42)
    X = rng.randn(300, 5) * 1000
    y = (X[:, 0] > 0).astype(int)
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    model.fit(X, y)
    return float(model.score(X, y))

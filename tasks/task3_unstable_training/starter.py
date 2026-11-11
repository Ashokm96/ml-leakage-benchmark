# WRONG: unscaled features, a high learning rate, and an unseeded RNG.
import numpy as np
from sklearn.linear_model import SGDClassifier


def run():
    rng = np.random.RandomState(None)  # BUG: no seed
    X = rng.randn(300, 5) * 1000  # BUG: unscaled features
    y = (X[:, 0] > 0).astype(int)
    model = SGDClassifier(
        loss="log_loss", learning_rate="constant", eta0=1.0, max_iter=100
    )  # BUG: high learning rate
    model.fit(X, y)
    return float(model.score(X, y))

# WRONG: random splitting breaks the temporal ordering of this dataset.
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def run():
    X = np.arange(200).reshape(-1, 1).astype(float)
    y = X.flatten() * 2 + np.random.RandomState(0).randn(200) * 5
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )  # BUG
    model = LinearRegression()
    model.fit(X_train, y_train)
    return float(model.score(X_test, y_test))

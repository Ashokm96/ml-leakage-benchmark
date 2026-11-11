from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def run():
    """Train and evaluate without fitting preprocessing on held-out data."""
    X, y = make_classification(n_samples=500, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    model.fit(X_train, y_train)
    return accuracy_score(y_test, model.predict(X_test))

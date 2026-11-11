# WRONG: the scaler is fit on the complete dataset before the split.
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def run():
    X, y = make_classification(n_samples=500, n_features=10, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # BUG: test statistics enter preprocessing
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return accuracy_score(y_test, model.predict(X_test))

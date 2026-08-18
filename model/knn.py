from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


def create_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            KNeighborsClassifier(
                n_neighbors=7
            )
        )
    ])
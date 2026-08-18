import os
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)


DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "breast-cancer-wisconsin/wdbc.data"
)


COLUMN_NAMES = [
    "id",
    "diagnosis",
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave_points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave_points_se",
    "symmetry_se",
    "fractal_dimension_se",
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave_points_worst",
    "symmetry_worst",
    "fractal_dimension_worst"
]


def load_dataset():
    data = pd.read_csv(
        DATA_URL,
        header=None,
        names=COLUMN_NAMES
    )

    # B = Benign = 0, M = Malignant = 1
    data["diagnosis"] = data["diagnosis"].map({
        "B": 0,
        "M": 1
    })

    data = data.drop(columns=["id"])

    feature_names = [
        column for column in data.columns
        if column != "diagnosis"
    ]

    X = data[feature_names]
    y = data["diagnosis"]

    return X, y, feature_names


def calculate_auc(model, X_test, y_test):
    probabilities = model.predict_proba(X_test)

    if probabilities.shape[1] == 2:
        return roc_auc_score(
            y_test,
            probabilities[:, 1]
        )

    return roc_auc_score(
        y_test,
        probabilities,
        multi_class="ovr",
        average="weighted"
    )


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(
            y_test,
            predictions
        ),

        "AUC": calculate_auc(
            model,
            X_test,
            y_test
        ),

        "Precision": precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        ),

        "Recall": recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        ),

        "F1": f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        ),

        "MCC": matthews_corrcoef(
            y_test,
            predictions
        )
    }

    return metrics


def save_model(model, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(model, file_path)
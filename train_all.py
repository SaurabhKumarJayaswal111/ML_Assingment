import os
import sys
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split

# Allow imports from the model directory
sys.path.append("model")

from common import (
    load_dataset,
    evaluate_model,
    save_model
)

from logistic_regression import create_model as create_logistic
from decision_tree import create_model as create_tree
from knn import create_model as create_knn
from naive_bayes import create_model as create_naive_bayes
from random_forest import create_model as create_random_forest


os.makedirs("model", exist_ok=True)


# Load dataset
X, y, feature_names = load_dataset()

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


models = {
    "Logistic Regression": create_logistic(),
    "Decision Tree": create_tree(),
    "KNN": create_knn(),
    "Naive Bayes": create_naive_bayes(),
    "Random Forest": create_random_forest()
}


results = []
model_paths = {}

for model_name, model in models.items():

    print(f"Training {model_name}...")

    model.fit(X_train, y_train)

    metrics = evaluate_model(
        model,
        X_test,
        y_test
    )

    result = {
        "ML Model Name": model_name,
        "Accuracy": round(metrics["Accuracy"], 4),
        "AUC": round(metrics["AUC"], 4),
        "Precision": round(metrics["Precision"], 4),
        "Recall": round(metrics["Recall"], 4),
        "F1": round(metrics["F1"], 4),
        "MCC": round(metrics["MCC"], 4)
    }

    results.append(result)

    file_name = (
        model_name.lower()
        .replace(" ", "_")
        + ".pkl"
    )

    file_path = os.path.join(
        "model",
        file_name
    )

    save_model(
        model,
        file_path
    )

    model_paths[model_name] = file_path


# Save test data
test_data = X_test.copy()
test_data["target"] = y_test.values

test_data.to_csv(
    "test_data.csv",
    index=False
)


# Save metadata
metadata = {
    "feature_names": feature_names,
    "target_column": "target",
    "class_names": [
        "Benign",
        "Malignant"
    ],
    "model_paths": model_paths
}

joblib.dump(
    metadata,
    "model/metadata.pkl"
)


# Save comparison table
comparison = pd.DataFrame(results)

comparison.to_csv(
    "model/comparison_results.csv",
    index=False
)

print("\nFinal Model Comparison")
print(comparison.to_string(index=False))

print("\nTraining completed successfully.")
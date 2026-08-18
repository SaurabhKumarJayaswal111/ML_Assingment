import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)


st.set_page_config(
    page_title="ML Classification Dashboard",
    page_icon="🧠",
    layout="wide"
)


MODEL_DIRECTORY = "model"
METADATA_PATH = os.path.join(
    MODEL_DIRECTORY,
    "metadata.pkl"
)


@st.cache_resource
def load_metadata():
    return joblib.load(METADATA_PATH)


@st.cache_resource
def load_model(path):
    return joblib.load(path)


def calculate_auc(model, X_data, y_data):
    probabilities = model.predict_proba(X_data)

    if probabilities.shape[1] == 2:
        return roc_auc_score(
            y_data,
            probabilities[:, 1]
        )

    return roc_auc_score(
        y_data,
        probabilities,
        multi_class="ovr",
        average="weighted"
    )


def calculate_metrics(model, X_data, y_data):
    predictions = model.predict(X_data)

    metrics = {
        "Accuracy": accuracy_score(
            y_data,
            predictions
        ),

        "AUC": calculate_auc(
            model,
            X_data,
            y_data
        ),

        "Precision": precision_score(
            y_data,
            predictions,
            average="weighted",
            zero_division=0
        ),

        "Recall": recall_score(
            y_data,
            predictions,
            average="weighted",
            zero_division=0
        ),

        "F1": f1_score(
            y_data,
            predictions,
            average="weighted",
            zero_division=0
        ),

        "MCC": matthews_corrcoef(
            y_data,
            predictions
        )
    }

    return predictions, metrics


st.title("🧠 Machine Learning Classification Dashboard")

st.write(
    """
This Streamlit application compares six classification models using the
Breast Cancer Wisconsin Diagnostic Dataset.
"""
)


if not os.path.exists(METADATA_PATH):
    st.error(
        "Model files are missing. Run train_all.py first."
    )
    st.stop()


metadata = load_metadata()

feature_names = metadata["feature_names"]
target_column = metadata["target_column"]
class_names = metadata["class_names"]
model_paths = metadata["model_paths"]


st.sidebar.header("Application Settings")

uploaded_file = st.sidebar.file_uploader(
    "Upload test data CSV",
    type=["csv"]
)


if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

elif os.path.exists("test_data.csv"):
    data = pd.read_csv("test_data.csv")

    st.sidebar.info(
        "Default test_data.csv is being used."
    )

else:
    st.warning(
        "Please upload a test CSV file."
    )
    st.stop()


missing_features = [
    feature
    for feature in feature_names
    if feature not in data.columns
]

if missing_features:
    st.error(
        "Missing required features: "
        + ", ".join(missing_features)
    )
    st.stop()


X_input = data[feature_names]

has_target = target_column in data.columns

if has_target:
    y_input = data[target_column]
else:
    y_input = None
    st.warning(
        "The target column is missing. "
        "Predictions can be generated, but metrics cannot be calculated."
    )


selected_model_name = st.sidebar.selectbox(
    "Select a model",
    list(model_paths.keys())
)

selected_model = load_model(
    model_paths[selected_model_name]
)


st.subheader("Dataset Preview")

st.dataframe(
    data.head(10),
    use_container_width=True
)

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Rows",
        data.shape[0]
    )

with col2:
    st.metric(
        "Columns",
        data.shape[1]
    )


predictions = selected_model.predict(X_input)

prediction_labels = [
    class_names[int(value)]
    for value in predictions
]

prediction_table = pd.DataFrame({
    "Predicted Class Number": predictions,
    "Predicted Class": prediction_labels
})


st.subheader("Prediction Results")

st.dataframe(
    prediction_table,
    use_container_width=True
)


if has_target:

    predictions, metrics = calculate_metrics(
        selected_model,
        X_input,
        y_input
    )

    st.subheader(
        f"Evaluation Metrics - {selected_model_name}"
    )

    metric_columns = st.columns(6)

    for column, (metric_name, metric_value) in zip(
        metric_columns,
        metrics.items()
    ):
        with column:
            st.metric(
                metric_name,
                f"{metric_value:.4f}"
            )


    st.subheader("Confusion Matrix")

    matrix = confusion_matrix(
        y_input,
        predictions
    )

    figure, axis = plt.subplots(
        figsize=(6, 4)
    )

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axis
    )

    axis.set_xlabel("Predicted Class")
    axis.set_ylabel("Actual Class")
    axis.set_title(
        f"{selected_model_name} Confusion Matrix"
    )

    st.pyplot(figure)


    st.subheader("Classification Report")

    report = classification_report(
        y_input,
        predictions,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    report_table = pd.DataFrame(
        report
    ).transpose()

    st.dataframe(
        report_table,
        use_container_width=True
    )
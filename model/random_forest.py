from sklearn.ensemble import RandomForestClassifier


def create_model():
    return RandomForestClassifier(
        n_estimators=250,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    )
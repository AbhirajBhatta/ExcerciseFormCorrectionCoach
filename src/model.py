from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

def train_model(X, y, n_estimators=100, random_state=42):
    """
    Train RandomForest model
    """

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state
    )

    model.fit(X, y)

    return model


def train_test_split_and_train(X, y, test_size=0.2, random_state=42):
    """
    Splits data, trains model, and returns everything
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = train_model(X_train, y_train, random_state=random_state)

    return model, X_train, X_test, y_train, y_test

def evaluate_model(model, X, y, cv=5):
    """
    Cross-validation score (more reliable than single split)
    """

    scores = cross_val_score(model, X, y, cv=cv)

    print(f"[INFO] Cross-val scores: {scores}")
    print(f"[INFO] Mean score: {scores.mean():.4f}")

    return scores.mean()

def evaluate_on_test_set(model, X_test, y_test):
    """
    Detailed evaluation on held-out test set
    """

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print(f"[INFO] Test Accuracy: {acc:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    return acc

def get_feature_importance(model, feature_names):
    """
    Shows which features matter most
    """

    import pandas as pd

    importance = model.feature_importances_

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    }).sort_values(by="importance", ascending=False)

    print(df)

    return df
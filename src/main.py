import pandas as pd
import joblib
from dataset import build_dataset
from model import (
    train_test_split_and_train,
    evaluate_model,
    evaluate_on_test_set,
    get_feature_importance
)


def main():
    DATA_DIR = "data/push-up"
    SAVE_PATH = "features.csv"

    print("\n[STEP 1] Building dataset...\n")
    # df = build_dataset(DATA_DIR, save_path=SAVE_PATH)
    df = pd.read_csv("features.csv")

    if df.empty:
        print("[ERROR] Dataset is empty. Exiting.")
        return

    print("\n[INFO] Dataset shape:", df.shape)
    print("\n[INFO] Sample data:\n", df.head())

    # -----------------------------
    # Prepare features + labels
    # -----------------------------
    print("\n[STEP 2] Preparing data...\n")

    X = df.drop(columns=["label", "video_path"])
    y = df["label"]

    print("[INFO] Feature columns:", list(X.columns))
    print("[INFO] Label distribution:\n", y.value_counts())

    # -----------------------------
    # Train model
    # -----------------------------
    print("\n[STEP 3] Training model...\n")

    model, X_train, X_test, y_train, y_test = train_test_split_and_train(X, y)
    # -----------------------------
    # Save model
    # -----------------------------
    # print("\n[STEP 3.5] Saving model...\n")

    # model_data = {
    #     "model": model,
    #     "features": list(X.columns)
    # }

    # joblib.dump(model_data, "pushup_model.pkl")

    # print("[INFO] Model saved as pushup_model.pkl")
    # -----------------------------
    # Evaluate
    # -----------------------------
    print("\n[STEP 4] Evaluating model...\n")

    evaluate_model(model, X, y)
    evaluate_on_test_set(model, X_test, y_test)

    # -----------------------------
    # Feature importance
    # -----------------------------
    print("\n[STEP 5] Feature importance...\n")

    importance_df = get_feature_importance(model, X.columns)

    print("\n[INFO] Top features:\n", importance_df.head())

    print("\n✅ Pipeline complete.\n")


if __name__ == "__main__":
    main()
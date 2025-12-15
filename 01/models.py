import pandas as pd
import numpy as np
import joblib
import os
import time

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def preview_data(df, name="Dataset"):
    print(f"\n=== {name} ===")
    print("Shape:", df.shape)
    print("\nHead:")
    print(df.head())
    print("\nInfo:")
    print(df.info())


def load_data(data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    data_path = os.path.join(data_dir, "adult.data")

    column_names = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income",
    ]

    if os.path.exists(data_path):
        print("Loading data from local file...")
        df = pd.read_csv(
            data_path, header=None, names=column_names, skipinitialspace=True
        )
    else:
        print("Downloading data from UCI repository...")
        url = (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        )
        df = pd.read_csv(url, header=None, names=column_names, skipinitialspace=True)
        # Save for future use
        df.to_csv(data_path, index=False, header=False)
        print(f"Data saved to {data_path}")

    return df


def handle_missing_values(df):
    df = df.copy()
    # The dataset uses " ?" for missing values
    df.replace(" ?", np.nan, inplace=True)
    print(f"Dropped {df.isna().sum().sum()} rows with missing values")
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def transform_data(df):
    X = df.drop("income", axis=1)
    y = df["income"].str.strip().apply(lambda x: x == ">50K")
    return X, y


def prepare_data_splits_and_preprocessor(X, y):
    cat_features = X.select_dtypes(include="object").columns.tolist()
    num_features = X.select_dtypes(exclude="object").columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(
        [
            ("onehot", OneHotEncoder(handle_unknown="ignore"), cat_features),
            ("scale", StandardScaler(), num_features),
        ]
    )

    return X_train, X_test, y_train, y_test, preprocessor


def main():
    os.makedirs("output", exist_ok=True)

    df = load_data()
    preview_data(df, "Original Data")

    df = handle_missing_values(df)
    preview_data(df, "Data After Cleaning")

    X, y = transform_data(df)
    X_train, X_test, y_train, y_test, preprocessor = (
        prepare_data_splits_and_preprocessor(X, y)
    )

    print(
        f"\nTraining on {X_train.shape[0]} samples, testing on {X_test.shape[0]} samples"
    )

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, solver="lbfgs", class_weight="balanced"
        ),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=42
        ),
        "MLPClassifier": MLPClassifier(
            hidden_layer_sizes=(128, 64), max_iter=1000, random_state=42
        ),
    }

    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        pipeline = make_pipeline(preprocessor, model)

        start_time = time.perf_counter()
        pipeline.fit(X_train, y_train)
        end_time = time.perf_counter()
        fit_time = end_time - start_time

        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        filename = f"output/{name}_model.joblib"
        joblib.dump(pipeline, filename)
        model_size_mb = os.path.getsize(filename) / (1024 * 1024)

        print(f"--- {name} Results ---")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Fit Time: {fit_time:.2f} seconds")
        print(f"Model Size: {model_size_mb:.2f} MB")
        print(f"Model saved to: {filename}")

        results[name] = {
            "accuracy": accuracy,
            "fit_time": fit_time,
            "model_size_mb": model_size_mb,
        }

    # Summary table
    print("\n=== Model Comparison Summary ===")
    summary_df = pd.DataFrame(results).T
    summary_df = summary_df.sort_values("accuracy", ascending=False)
    print(summary_df.round(4))


if __name__ == "__main__":
    main()

import argparse
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
import yaml
import os

def main():
    print("Assignment 3:", os.getcwd())

    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument("output_model_path", type=str)
    parser.add_argument("output_metrics_path", type=str)
    args = parser.parse_args()

    # Load training parameters
    try:
        with open("params.yaml", "r") as f:
            params = yaml.safe_load(f)
    except Exception as e:
        print("Failed to load params.yaml:", e)
        return

    # Load dataset
    df = pd.read_csv(args.input_path)
    print("Columns in the dataset:", df.columns)

    features = params["train"]["features"]
    targets = params["train"]["target"]

    # Validate features and targets
    for feature in features:
        if feature not in df.columns:
            print(f"Feature '{feature}' not found in the dataset.")
            return
    for target in targets:
        if target not in df.columns:
            print(f"Target '{target}' not found in the dataset.")
            return

    x = df[features]
    y = df[targets]  # Multiple target columns as DataFrame

    # Train and save model
    model = LinearRegression()
    model.fit(x, y)

    os.makedirs(os.path.dirname(args.output_model_path), exist_ok=True)
    with open(args.output_model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved at {args.output_model_path}")

    # Save metrics
    score = model.score(x, y)  # Returns R² for multi-target
    os.makedirs(os.path.dirname(args.output_metrics_path), exist_ok=True)
    with open(args.output_metrics_path, "w") as f:
        f.write(f"R-squared: {score}\n")
    print(f"Metrics saved at {args.output_metrics_path}")

if __name__ == "__main__":
    main()

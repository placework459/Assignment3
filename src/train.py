import argparse
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
import yaml

def main():
    # Initialize argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument("output_model_path", type=str)
    parser.add_argument("output_metrics_path", type=str)
    args = parser.parse_args()

    # Load training parameters from YAML
    with open("src/params.yaml", "r") as f:
        params = yaml.safe_load(f)

    # Load dataset
    df = pd.read_csv(args.input_path)

    # Debugging: print the columns to verify
    print("Columns in the dataset:", df.columns)

    # Extract features and target
    features = params["train"]["features"]
    target = params["train"]["target"]
    
    # Ensure the features and target exist in the dataset columns
    for feature in features:
        if feature not in df.columns:
            print(f"Feature '{feature}' not found in the dataset.")
            return

    if target not in df.columns:
        print(f"Target '{target}' not found in the dataset.")
        return

    x = df[features]  # x should be a 2D DataFrame
    y = df[target]

    # Train model
    model = LinearRegression()
    model.fit(x, y)

    # Save model
    with open(args.output_model_path, "wb") as f:
        pickle.dump(model, f)

    # Verify the model was saved
    print(f"Model saved at {args.output_model_path}")

    # Save metrics
    with open(args.output_metrics_path, "w") as f:
        f.write(f"R-squared: {model.score(x, y)}\n")
        print(f"Metrics saved at {args.output_metrics_path}")

if __name__ == "__main__":
    main()

# main.py
import argparse

import joblib
import numpy as np

from model_pipeline import (load_model_artifacts, predict, prepare_data,
                            save_model, train_model)

# Define file paths for artifacts
DATA_PATH = "processed_data.pkl"
MODEL_KERAS_PATH = "throughput_model.keras"
MODEL_PKL_PATH = "throughput_model.pkl"
SCALER_Y_PATH = "scaler_y.pkl"


def main():
    parser = argparse.ArgumentParser(description="5G Throughput Prediction Pipeline")
    parser.add_argument(
        "step",
        choices=["prepare", "train", "predict", "save", "load"],
        help="Pipeline step to execute",
    )
    parser.add_argument(
        "--file_path", default="mm-5G.csv", help="Path to the input CSV file"
    )
    args = parser.parse_args()

    if args.step == "prepare":
        X_seq, X_ctx, y = prepare_data(args.file_path)
        joblib.dump((X_seq, X_ctx, y), DATA_PATH)
        print("Data prepared and saved to", DATA_PATH)

    elif args.step == "train":
        try:
            X_seq, X_ctx, y = joblib.load(DATA_PATH)
        except FileNotFoundError:
            raise FileNotFoundError(
                "Prepared data not found. Run 'make prepare' first."
            )
        model = train_model(X_seq, X_ctx, y)
        save_model(model, MODEL_KERAS_PATH, MODEL_PKL_PATH)
        print("Model trained and saved to", MODEL_KERAS_PATH, "and", MODEL_PKL_PATH)

    elif args.step == "save":
        try:
            X_seq, X_ctx, y = joblib.load(DATA_PATH)
            model = train_model(X_seq, X_ctx, y)
            save_model(model, MODEL_KERAS_PATH, MODEL_PKL_PATH)
            print("Model saved to", MODEL_KERAS_PATH, "and", MODEL_PKL_PATH)
        except FileNotFoundError:
            raise FileNotFoundError(
                "Prepared data not found. Run 'make prepare' first."
            )

    elif args.step == "load":
        try:
            model = load_model_artifacts(MODEL_KERAS_PATH)
            print("Model loaded from", MODEL_KERAS_PATH)
            X_seq, X_ctx, y = joblib.load(DATA_PATH)
            scaler_y = joblib.load(SCALER_Y_PATH)
            y_pred_mbps, y_true_mbps, y_pred_classes, metrics = predict(
                X_seq[:5], X_ctx[:5], y[:5], model, scaler_y
            )
            print("Sample predictions (first 5):")
            for i in range(len(y_pred_mbps)):
                print(
                    f"Sample {i+1}: Real {y_true_mbps[i][0]:.2f} Mbps | Predicted {y_pred_mbps[i][0]:.2f} Mbps | Class: {y_pred_classes[i]}"
                )
            print("Metrics:", metrics)
        except FileNotFoundError:
            raise FileNotFoundError(
                "Model or data not found. Run 'make train' or 'make prepare' first."
            )

    elif args.step == "predict":
        try:
            X_seq, X_ctx, y = joblib.load(DATA_PATH)
            model = load_model_artifacts(MODEL_KERAS_PATH)
            scaler_y = joblib.load(SCALER_Y_PATH)
            y_pred_mbps, y_true_mbps, y_pred_classes, metrics = predict(
                X_seq, X_ctx, y, model, scaler_y
            )

            # Display first 20 predictions with classification
            print("\n🔍 First 20 predictions (with classification):")
            for i in range(min(20, len(y_pred_mbps))):
                error = abs(y_true_mbps[i][0] - y_pred_mbps[i][0])
                error_pct = (error / max(y_true_mbps[i][0], 1.0)) * 100
                print(
                    f"{i+1}. Real: {y_true_mbps[i][0]:.2f} Mbps | Predicted: {y_pred_mbps[i][0]:.2f} Mbps | "
                    f"Error: {error:.2f} Mbps ({error_pct:.2f}%) | Class: {y_pred_classes[i]}"
                )

            # Display metrics
            print("\n📊 Evaluation Metrics:")
            print(f"R²: {metrics['R2']:.4f}")
            print(f"Log-MAPE: {metrics['Log-MAPE']:.2f}%")
            print(f"SMAPE: {metrics['SMAPE']:.2f}%")
            print(f"RMSE: {metrics['RMSE']:.4f}")
            print(f"MAE: {metrics['MAE']:.4f}")
            # Updated thresholds output to explicitly include High
            print(
                f"Classification Thresholds: Low < {metrics['Classification Thresholds']['Low']:.2f} Mbps, "
                f"Medium < {metrics['Classification Thresholds']['Medium']:.2f} Mbps, "
                f"High >= {metrics['Classification Thresholds']['Medium']:.2f} Mbps"
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                "Required files not found. Run 'make prepare' and 'make train' first."
            )


if __name__ == "__main__":
    main()

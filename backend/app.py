from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from contextlib import asynccontextmanager
import os
import joblib
import numpy as np
import pandas as pd
from typing import List, Optional
from model_pipeline import train_model, load_model_artifacts, predict, save_model
import mlflow
import mlflow.keras
from mlflow.models import ModelSignature
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from elasticsearch import Elasticsearch
from datetime import datetime
from collections import defaultdict
import requests

# Define file paths
DATA_PATH = "processed_data.pkl"
MODEL_KERAS_PATH = "throughput_model.keras"
MODEL_PKL_PATH = "throughput_model.pkl"
SCALER_Y_PATH = "scaler_y.pkl"
UPLOAD_DIR = "data/"
SAVE_DIR = "saved_data/"
SAVE_FILE = os.path.join(SAVE_DIR, "saved_data.csv")

# Create directories if they don't exist
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# Global variables for model, scaler, Elasticsearch client, and metrics
model = None
scaler_y = None
es = Elasticsearch(["http://localhost:9200"], basic_auth=None, verify_certs=False)
throughput_metrics = defaultdict(int)

# Set up MLflow
MLFLOW_TRACKING_URI = "http://localhost:5001"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
EXPERIMENT_NAME = "ThroughputPredictionExperiment"

# Validate MLflow server availability
try:
    response = requests.get(MLFLOW_TRACKING_URI, timeout=5)
    if response.status_code != 200:
        raise Exception(f"MLflow server at {MLFLOW_TRACKING_URI} is not accessible. Status code: {response.status_code}")
    print(f"MLflow server at {MLFLOW_TRACKING_URI} is accessible.")
except Exception as e:
    print(f"Failed to connect to MLflow server: {str(e)}")
    raise Exception(f"MLflow server setup issue: {str(e)}")

# Create the experiment if it doesn't exist
try:
    experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
    print(f"Created new experiment '{EXPERIMENT_NAME}' with ID: {experiment_id}")
except mlflow.exceptions.MlflowException:
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    experiment_id = experiment.experiment_id
    print(f"Using existing experiment '{EXPERIMENT_NAME}' with ID: {experiment_id}")
mlflow.set_experiment(EXPERIMENT_NAME)

# Lifespan handler for startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler_y
    if os.path.exists(MODEL_KERAS_PATH) and os.path.exists(SCALER_Y_PATH):
        model = load_model_artifacts(MODEL_KERAS_PATH)
        scaler_y = joblib.load(SCALER_Y_PATH)
        print("Loaded model and scaler during startup.")
    yield
    model = None
    scaler_y = None

# Initialize FastAPI app with lifespan
app = FastAPI(title="5G Throughput Prediction API", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for input and response structure
class SequenceInput(BaseModel):
    data: List[List[List[float]]] = Field(..., description="3D array of sequential data (e.g., [samples, timesteps, features])")
    model_config = ConfigDict(coerce_numbers_to_float=True)

class ContextInput(BaseModel):
    data: List[List[float]] = Field(..., description="2D array of contextual data (e.g., [samples, features])")
    model_config = ConfigDict(coerce_numbers_to_float=True)

class PredictionInput(BaseModel):
    x_seq: SequenceInput
    x_ctx: ContextInput
    y: Optional[List[float]] = None
    model_config = ConfigDict(coerce_numbers_to_float=True)

class Prediction(BaseModel):
    sample_index: int
    predicted_mbps: float
    class_label: str

class Metrics(BaseModel):
    R2: Optional[float] = None
    Log_MAPE: Optional[float] = None
    SMAPE: Optional[float] = None
    RMSE: Optional[float] = None
    MAE: Optional[float] = None
    Classification_Thresholds: Optional[dict] = None
    Throughput_Classification_Counts: Optional[dict] = None

class PredictionResponse(BaseModel):
    predictions: List[Prediction]
    metrics: Metrics

# Utility function to replace NaN with a default value
def replace_nan_with_default(value, default=0.0):
    if isinstance(value, dict):
        return {k: replace_nan_with_default(v, default) for k, v in value.items()}
    elif isinstance(value, list):
        return [replace_nan_with_default(item, default) for item in value]
    elif isinstance(value, (int, float)) and (np.isnan(value) or np.isinf(value)):
        return default
    return value

# Log event to Elasticsearch
def log_to_elasticsearch(event_type, endpoint, details):
    doc = {
        "event": event_type,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "endpoint": endpoint,
        **details
    }
    try:
        es.index(index="pita-metrics", document=doc)
        print(f"Logged to Elasticsearch: {event_type} at {endpoint}")
    except Exception as e:
        print(f"Failed to log to Elasticsearch: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    with mlflow.start_run(run_name="HealthCheckRun"):
        mlflow.log_param("endpoint", "health")
        mlflow.log_metric("status_code", 200)
        log_to_elasticsearch("health_check", "/health", {"status": "healthy"})
        print("Logged health check to MLflow and Elasticsearch.")
    return {"status": "healthy"}

# Metrics endpoint for Metricbeat
@app.get("/metrics")
async def get_metrics():
    metrics = {
        "throughput.low_count": throughput_metrics["low"],
        "throughput.medium_count": throughput_metrics["medium"],
        "throughput.high_count": throughput_metrics["high"]
    }
    return metrics

# Data preparation endpoint
@app.post("/prepare")
async def prepare(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    try:
        from model_pipeline import prepare_data
        X_seq, X_ctx, y = prepare_data(file_path)
        joblib.dump((X_seq, X_ctx, y), DATA_PATH)
        
        with mlflow.start_run(run_name="PrepareRun"):
            mlflow.log_artifact(file_path, artifact_path="datasets")
            mlflow.log_param("file_name", file.filename)
            mlflow.log_param("file_path", file_path)
            mlflow.log_metric("file_size_bytes", os.path.getsize(file_path))
            if os.path.exists(DATA_PATH):
                mlflow.log_artifact(DATA_PATH, artifact_path="processed_data")
                mlflow.log_param("processed_data_path", DATA_PATH)
            log_to_elasticsearch("data_prepare", "/prepare", {"file_name": file.filename, "size_bytes": os.path.getsize(file_path)})
            print(f"Logged prepare run: file={file.filename}, size={os.path.getsize(file_path)} bytes")
        
        return {"message": f"Data prepared and saved to {DATA_PATH}"}
    except Exception as e:
        log_to_elasticsearch("error", "/prepare", {"error": str(e)})
        raise HTTPException(status_code=500, detail=f"Error preparing data: {str(e)}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

# Model training endpoint
@app.post("/train")
async def train(epochs: int = 30, batch_size: int = 128):
    if not os.path.exists(DATA_PATH):
        raise HTTPException(status_code=404, detail="Prepared data not found. Run /prepare first.")
    
    try:
        print("Loading data from processed_data.pkl...")
        X_seq, X_ctx, y = joblib.load(DATA_PATH)
        print(f"Data loaded successfully. Shapes - X_seq: {X_seq.shape}, X_ctx: {X_ctx.shape}, y: {y.shape if y is not None else 'None'}")
        
        print("Training model...")
        trained_model = train_model(X_seq, X_ctx, y, epochs=epochs, batch_size=batch_size)
        print("Model trained successfully.")
        
        print(f"Saving model to {MODEL_KERAS_PATH} and {MODEL_PKL_PATH}...")
        save_model(trained_model, MODEL_KERAS_PATH, MODEL_PKL_PATH)
        print("Model saved successfully.")
        
        global model, scaler_y
        model = trained_model
        print(f"Loading scaler from {SCALER_Y_PATH}...")
        scaler_y = joblib.load(SCALER_Y_PATH)
        print("Scaler loaded successfully.")
        
        with mlflow.start_run(run_name="TrainRun") as run:
            print(f"Started MLflow run with ID: {run.info.run_id}")
            
            sample_x_seq = np.array(X_seq[:1])
            sample_x_ctx = np.array(X_ctx[:1])
            sample_y = np.array(y[:1]).reshape(-1, 1) if y is not None else None
            print(f"Sample shapes for signature - X_seq: {sample_x_seq.shape}, X_ctx: {sample_x_ctx.shape}, y: {sample_y.shape if sample_y is not None else 'None'}")
            
            print("Inferring model signature...")
            try:
                prediction = trained_model.predict([sample_x_seq, sample_x_ctx])
                signature = infer_signature(
                    model_input=(sample_x_seq, sample_x_ctx),
                    model_output=prediction
                )
                print(f"Model signature inferred: {signature}")
            except Exception as e:
                print(f"Failed to infer model signature: {str(e)}")
                raise
            
            print("Logging model to MLflow...")
            try:
                model_info = mlflow.keras.log_model(
                    trained_model,
                    artifact_path="models/keras_model",
                    signature=signature
                )
                print(f"Logged Keras model. Model info: {model_info.model_uuid}")
            except Exception as e:
                print(f"Failed to log model to MLflow: {str(e)}")
                raise
            
            model_name = "throughput-prediction-model"
            model_uri = f"runs:/{run.info.run_id}/models/keras_model"
            print(f"Model URI: {model_uri}")
            
            print(f"Registering model as '{model_name}'...")
            client = MlflowClient()
            try:
                # Create registered model if it doesn't exist
                client.create_registered_model(model_name)
                print(f"Created registered model '{model_name}' if it didn't exist.")
            except Exception as e:
                print(f"Failed to create registered model: {str(e)}")
            
            try:
                registered_model = mlflow.register_model(model_uri, model_name)
                print(f"Registered model '{model_name}' with version {registered_model.version}")
            except Exception as e:
                print(f"Failed to register model: {str(e)}")
                raise
            
            try:
                client.update_model_version(
                    name=model_name,
                    version=registered_model.version,
                    description="CNN+LSTM model for 5G throughput prediction."
                )
                print(f"Updated description for model '{model_name}' version {registered_model.version}")
            except Exception as e:
                print(f"Failed to update description: {str(e)}")
            
            try:
                client.set_model_version_tag(name=model_name, version=registered_model.version, key="task", value="throughput-prediction")
                client.set_model_version_tag(name=model_name, version=registered_model.version, key="framework", value="keras")
                print(f"Added tags to model '{model_name}' version {registered_model.version}")
            except Exception as e:
                print(f"Failed to add tags: {str(e)}")
            
            if os.path.exists(MODEL_PKL_PATH):
                mlflow.log_artifact(MODEL_PKL_PATH, artifact_path="models/pkl_model")
                print(f"Logged artifact: {MODEL_PKL_PATH}")
            if os.path.exists(SCALER_Y_PATH):
                mlflow.log_artifact(SCALER_Y_PATH, artifact_path="scalers")
                print(f"Logged artifact: {SCALER_Y_PATH}")
            
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("model_keras_path", MODEL_KERAS_PATH)
            mlflow.log_param("model_pkl_path", MODEL_PKL_PATH)
            mlflow.log_param("scaler_y_path", SCALER_Y_PATH)
            mlflow.log_metric("training_complete", 1)
            log_to_elasticsearch("model_train", "/train", {"epochs": epochs, "batch_size": batch_size})
            print(f"Logged train run: epochs={epochs}, batch_size={batch_size}")
        
        return {"message": f"Model trained and saved to {MODEL_KERAS_PATH} and {MODEL_PKL_PATH}"}
    except Exception as e:
        log_to_elasticsearch("error", "/train", {"error": str(e)})
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")

# Prediction endpoint with JSON input
@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(prediction_input: PredictionInput):
    if model is None or scaler_y is None:
        raise HTTPException(status_code=404, detail="Model or scaler not loaded. Run /train first.")

    try:
        x_seq_data = np.array(prediction_input.x_seq.data)
        x_ctx_data = np.array(prediction_input.x_ctx.data)
        y_data = np.array(prediction_input.y).reshape(-1, 1) if prediction_input.y is not None else None

        if x_seq_data.ndim != 3 or x_ctx_data.ndim != 2:
            raise HTTPException(status_code=400, detail="Invalid input shape. X_seq should be [samples, timesteps, features], X_ctx should be [samples, features].")
        
        y_pred_mbps, y_true_mbps, y_pred_classes, metrics = predict(x_seq_data, x_ctx_data, y_data, model, scaler_y) if y_data is not None else predict(x_seq_data, x_ctx_data, None, model, scaler_y)

        if np.any(np.isnan(y_pred_mbps)):
            raise HTTPException(status_code=500, detail="NaN values detected in model predictions. Input data may be invalid or unscaled.")
        
        # Classify throughput into low, medium, high
        y_pred_classes = []
        classification_counts = {"low": 0, "medium": 0, "high": 0}
        for mbps in y_pred_mbps:
            mbps_value = float(mbps[0])
            if mbps_value < 10:
                label = "low"
                classification_counts["low"] += 1
            elif 10 <= mbps_value <= 50:
                label = "medium"
                classification_counts["medium"] += 1
            else:
                label = "high"
                classification_counts["high"] += 1
            y_pred_classes.append(label)
            throughput_metrics[label] += 1

        predictions = []
        for i in range(min(20, len(y_pred_mbps))):
            predictions.append(Prediction(
                sample_index=i + 1,
                predicted_mbps=float(y_pred_mbps[i][0]),
                class_label=y_pred_classes[i]
            ))

        metrics_response = Metrics(
            R2=float(metrics["R2"]) if y_data is not None and "R2" in metrics else None,
            Log_MAPE=float(metrics["Log-MAPE"]) if y_data is not None and "Log-MAPE" in metrics else None,
            SMAPE=float(metrics["SMAPE"]) if y_data is not None and "SMAPE" in metrics else None,
            RMSE=float(metrics["RMSE"]) if y_data is not None and "RMSE" in metrics else None,
            MAE=float(metrics["MAE"]) if y_data is not None and "MAE" in metrics else None,
            Classification_Thresholds=metrics["Classification Thresholds"] if y_data is not None else None,
            Throughput_Classification_Counts=classification_counts
        )

        with mlflow.start_run(run_name="PredictRun"):
            mlflow.log_metric("average_predicted_mbps", float(np.mean(y_pred_mbps)))
            mlflow.log_metric("prediction_count", len(y_pred_mbps))
            mlflow.log_param("x_seq_shape", str(x_seq_data.shape))
            mlflow.log_param("x_ctx_shape", str(x_ctx_data.shape))
            mlflow.log_param("y_provided", y_data is not None)
            if y_data is not None:
                mlflow.log_metric("R2", float(metrics["R2"]))
                mlflow.log_metric("Log_MAPE", float(metrics["Log-MAPE"]))
                mlflow.log_metric("SMAPE", float(metrics["SMAPE"]))
                mlflow.log_metric("RMSE", float(metrics["RMSE"]))
                mlflow.log_metric("MAE", float(metrics["MAE"]))
            mlflow.log_metric("low_throughput_count", classification_counts["low"])
            mlflow.log_metric("medium_throughput_count", classification_counts["medium"])
            mlflow.log_metric("high_throughput_count", classification_counts["high"])
            if os.path.exists(DATA_PATH):
                mlflow.log_artifact(DATA_PATH, artifact_path="processed_data")
            if os.path.exists(SAVE_FILE):
                mlflow.log_artifact(SAVE_FILE, artifact_path="saved_data")
            for i, (mbps, label) in enumerate(zip(y_pred_mbps, y_pred_classes)):
                log_to_elasticsearch("prediction", "/predict", {
                    "sample_index": i + 1,
                    "predicted_mbps": float(mbps[0]),
                    "class_label": label,
                    "avg_mbps": float(np.mean(y_pred_mbps)),
                    "count": len(y_pred_mbps),
                    "classification_counts": classification_counts
                })
            print(f"Logged predict run: avg_mbps={float(np.mean(y_pred_mbps)):.2f}, count={len(y_pred_mbps)}")

        response = PredictionResponse(predictions=predictions, metrics=metrics_response)
        response_dict = response.dict()
        response_dict = replace_nan_with_default(response_dict, default=0.0)

        return JSONResponse(content=response_dict)
    except Exception as e:
        log_to_elasticsearch("error", "/predict", {"error": str(e)})
        raise HTTPException(status_code=500, detail=f"Error making predictions: {str(e)}")

@app.post("/save")
async def save(
    latitude: float = Form(...),
    longitude: float = Form(...),
    run_num: int = Form(...),
    seq_num: float = Form(...),
    abstractSignalStr: int = Form(...),
    movingSpeed: float = Form(...),
    compassDirection: float = Form(...),
    nrStatus: str = Form(...),
    lte_rssi: float = Form(...),
    lte_rsrp: float = Form(...),
    lte_rsrq: float = Form(...),
    lte_rssnr: float = Form(...),
    nr_ssRsrp: Optional[float] = Form(None),
    nr_ssRsrq: Optional[float] = Form(None),
    nr_ssSinr: Optional[float] = Form(None),
    mobility_mode: str = Form(...),
    trajectory_direction: str = Form(...),
    tower_id: int = Form(...)
):
    try:
        data = {
            'latitude': [latitude],
            'longitude': [longitude],
            'run_num': [run_num],
            'seq_num': [seq_num],
            'abstractSignalStr': [abstractSignalStr],
            'movingSpeed': [movingSpeed],
            'compassDirection': [compassDirection],
            'nrStatus': [nrStatus],
            'lte_rssi': [lte_rssi],
            'lte_rsrp': [lte_rsrp],
            'lte_rsrq': [lte_rsrq],
            'lte_rssnr': [lte_rssnr],
            'nr_ssRsrp': [nr_ssRsrp if nr_ssRsrp is not None else -94],
            'nr_ssRsrq': [nr_ssRsrq if nr_ssRsrq is not None else -14],
            'nr_ssSinr': [nr_ssSinr if nr_ssSinr is not None else 0],
            'mobility_mode': [mobility_mode],
            'trajectory_direction': [trajectory_direction],
            'tower_id': [tower_id]
        }
        df = pd.DataFrame(data)

        if os.path.exists(SAVE_FILE):
            existing_df = pd.read_csv(SAVE_FILE)
            df = pd.concat([existing_df, df], ignore_index=True)
        df.to_csv(SAVE_FILE, index=False)

        with mlflow.start_run(run_name="SaveRun"):
            mlflow.log_artifact(SAVE_FILE, artifact_path="saved_data")
            mlflow.log_param("latitude", latitude)
            mlflow.log_param("longitude", longitude)
            mlflow.log_param("run_num", run_num)
            mlflow.log_param("seq_num", seq_num)
            mlflow.log_param("abstractSignalStr", abstractSignalStr)
            mlflow.log_param("movingSpeed", movingSpeed)
            mlflow.log_param("nrStatus", nrStatus)
            mlflow.log_param("mobility_mode", mobility_mode)
            mlflow.log_param("trajectory_direction", trajectory_direction)
            mlflow.log_param("tower_id", tower_id)
            mlflow.log_metric("saved_records", len(df))
            log_to_elasticsearch("data_save", "/save", {"latitude": latitude, "longitude": longitude, "records": len(df)})
            print(f"Logged save run: saved_records={len(df)}")

        return {"message": "Data saved successfully", "filename": "saved_data.csv"}
    except Exception as e:
        log_to_elasticsearch("error", "/save", {"error": str(e)})
        raise HTTPException(status_code=500, detail=f"Failed to save data: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
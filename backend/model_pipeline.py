import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import iqr
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Activation, BatchNormalization, Concatenate, Dense, Dropout, Input
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.regularizers import l2
import keras

# Global variable for classification thresholds
CLASSIFICATION_THRESHOLDS = None

@keras.saving.register_keras_serializable(package="Custom")
class CustomMSE(MeanSquaredError):
    def __init__(
        self, low_threshold=100.0, weight_factor=2.0, name="custom_mse", reduction=None
    ):
        super().__init__(name=name, reduction=reduction)
        self.low_threshold = low_threshold
        self.weight_factor = weight_factor

    def call(self, y_true, y_pred, scaler_y=None):
        mse = super().call(y_true, y_pred)
        if scaler_y is not None:
            y_true_mbps = scaler_y.inverse_transform(y_true.numpy().reshape(-1, 1)).flatten()
            y_true_mbps = tf.convert_to_tensor(y_true_mbps, dtype=tf.float32)
        else:
            y_true_mbps = y_true
        weights = tf.where(y_true_mbps < self.low_threshold, self.weight_factor, 1.0)
        return tf.reduce_mean(weights * mse)

    def get_config(self):
        config = super().get_config()
        config.update({
            "low_threshold": self.low_threshold,
            "weight_factor": self.weight_factor,
            "name": self.name,
            "reduction": self.reduction
        })
        return config

    @classmethod
    def from_config(cls, config):
        low_threshold = config.get("low_threshold", 100.0)
        weight_factor = config.get("weight_factor", 2.0)
        name = config.get("name", "custom_mse")
        reduction = config.get("reduction", None)
        return cls(
            low_threshold=low_threshold,
            weight_factor=weight_factor,
            name=name,
            reduction=reduction,
        )

def prepare_data(file_path):
    df = pd.read_csv(file_path)
    df = df.iloc[::2].reset_index(drop=True)

    # Handle LTE and NR columns
    lte_cols = [c for c in df.columns if c.startswith("lte_")]
    nr_cols = [c for c in df.columns if c.startswith("nr_")]
    for col in lte_cols + nr_cols:
        df[col] = df[col].fillna(df[col].median())

    # Remove outliers from Throughput
    Q1, Q3 = df["Throughput"].quantile([0.25, 0.75])
    IQR = iqr(df["Throughput"])
    lower_bound = max(1.0, Q1 - 1.5 * IQR)
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df["Throughput"] >= lower_bound) & (df["Throughput"] <= upper_bound)]

    # Encode categorical columns
    for col in ["nrStatus", "trajectory_direction", "mobility_mode"]:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        else:
            df[col] = 0

    df["mobility_mode_encoded"] = df["mobility_mode"]
    df["tower_id_encoded"] = LabelEncoder().fit_transform(df["tower_id"].astype(str))

    # Clustering based on coordinates
    coords = df[["latitude", "longitude"]].dropna()
    kmeans = KMeans(n_clusters=5, random_state=42)
    df.loc[coords.index, "zone_cluster"] = kmeans.fit_predict(coords)
    df["zone_cluster"] = df["zone_cluster"].fillna(0)

    # Time-based features
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["hour"] = df["timestamp"].dt.hour.fillna(0).astype(int)
        df["day_of_week"] = df["timestamp"].dt.dayofweek.fillna(0).astype(int)
        df["time_slot"] = (
            df["hour"] * 60 + df["timestamp"].dt.minute.fillna(0).astype(int)
        ) // 60
        df["is_peak_hour"] = (
            (df["hour"].between(8, 10) | df["hour"].between(17, 19))
            & (df["day_of_week"] < 5)
        ).astype(int)
    else:
        df["hour"] = df["day_of_week"] = df["time_slot"] = df["is_peak_hour"] = 0

    # Feature engineering
    df["throughput_delta"] = df["Throughput"].diff().fillna(0)
    df["throughput_rolling_mean"] = df["Throughput"].rolling(5).mean().bfill()
    df["throughput_rolling_std"] = df["Throughput"].rolling(5).std().bfill()
    df["throughput_acceleration"] = df["throughput_delta"].diff().fillna(0)
    df["is_handover"] = df["tower_id"].ne(df["tower_id"].shift()).astype(int)

    if "nr_ssSinr" in df.columns and "lte_rsrp" in df.columns:
        df["is_nr_dominant"] = (
            df["nr_ssSinr"].fillna(0) > df["lte_rsrp"].fillna(0)
        ).astype(int)
        df["signal_ratio"] = df["nr_ssSinr"].fillna(0) / (
            df["lte_rsrp"].fillna(-100) + 1e-6
        )
    else:
        df["is_nr_dominant"] = 0
        df["signal_ratio"] = 0

    df["throughput_lag1"] = df["Throughput"].shift(1).fillna(0)

    def haversine(lat1, lon1, lat2, lon2):
        R = 6371e3
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        a = (
            np.sin(dphi / 2) ** 2
            + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
        )
        return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    if "tower_lat" in df.columns and "tower_lon" in df.columns:
        df["distance_to_tower"] = haversine(
            df["latitude"], df["longitude"], df["tower_lat"], df["tower_lon"]
        )
    else:
        df["distance_to_tower"] = 0

    # Define context columns (exactly 15 features)
    context_cols = [
        "mobility_mode_encoded",
        "tower_id_encoded",
        "zone_cluster",
        "hour",
        "day_of_week",
        "time_slot",
        "throughput_delta",
        "throughput_rolling_std",
        "throughput_acceleration",
        "is_handover",
        "is_nr_dominant",
        "distance_to_tower",
        "signal_ratio",
        "throughput_lag1",
        "is_peak_hour",
    ]

    print(f"Number of context features: {len(context_cols)}")

    # Initialize scalers
    scaler_context = RobustScaler()
    scaler_lte_nr = RobustScaler()
    scaler_y = MinMaxScaler()

    # Apply scaling
    df[context_cols] = scaler_context.fit_transform(df[context_cols])
    df[lte_cols + nr_cols] = scaler_lte_nr.fit_transform(df[lte_cols + nr_cols])
    df["throughput_rolling_mean"] = scaler_y.fit_transform(
        df[["throughput_rolling_mean"]]
    )

    # Save scalers and kmeans model
    joblib.dump(scaler_context, "scaler_context.pkl")
    joblib.dump(scaler_lte_nr, "scaler_lte_nr.pkl")
    joblib.dump(scaler_y, "scaler_y.pkl")
    joblib.dump(kmeans, "kmeans.pkl")

    # Create sequences
    def create_sequences(X_seq, X_ctx, y, seq_len=10):
        Xs, Xc, ys = [], [], []
        for i in range(len(X_seq) - seq_len + 1):
            Xs.append(X_seq[i : i + seq_len])
            Xc.append(X_ctx[i + seq_len - 1])
            ys.append(y[i + seq_len - 1])
        return np.array(Xs), np.array(Xc), np.array(ys)

    sequence_cols = ["throughput_delta"] + lte_cols + nr_cols
    X_seq = df[sequence_cols].to_numpy()
    X_ctx = df[context_cols].to_numpy()
    y = df[["throughput_rolling_mean"]].to_numpy()
    X_seq, X_ctx, y = create_sequences(X_seq, X_ctx, y)

    # Calculate classification thresholds based on training data
    global CLASSIFICATION_THRESHOLDS
    y_mbps = scaler_y.inverse_transform(y).flatten()
    low_threshold = np.percentile(y_mbps, 33)
    medium_threshold = np.percentile(y_mbps, 66)
    CLASSIFICATION_THRESHOLDS = {"Low": float(low_threshold), "Medium": float(medium_threshold)}
    joblib.dump(CLASSIFICATION_THRESHOLDS, "classification_thresholds.pkl")

    return X_seq, X_ctx, y

def build_model(input_seq_shape, input_ctx_shape, learning_rate=0.001):
    if not input_seq_shape or not input_ctx_shape:
        raise ValueError("Input shapes must be non-empty")
    
    input_seq = Input(shape=input_seq_shape, name="input_seq")
    x = LSTM(64, kernel_regularizer=l2(0.001), return_sequences=False)(input_seq)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    input_ctx = Input(shape=input_ctx_shape, name="input_ctx")
    x = Concatenate()([x, input_ctx])
    x = Dense(1)(x)
    output = Activation("relu")(x)
    
    model = Model([input_seq, input_ctx], output)
    model.compile(
        optimizer=AdamW(learning_rate=learning_rate, weight_decay=0.01, clipnorm=1.0),
        loss=CustomMSE(low_threshold=100.0, weight_factor=2.0),
        metrics=["mae"],
    )
    return model

def train_model(X_seq, X_ctx, y, epochs=30, batch_size=128, learning_rate=0.001):
    X_seq_tr, X_seq_te, X_ctx_tr, X_ctx_te, y_tr, y_te = train_test_split(
        X_seq, X_ctx, y, test_size=0.2, random_state=42
    )
    model = build_model((X_seq.shape[1], X_seq.shape[2]), (X_ctx.shape[1],), learning_rate)
    early_stop = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    lr_scheduler = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
    )
    history = model.fit(
        [X_seq_tr, X_ctx_tr],
        y_tr,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stop, lr_scheduler],
        verbose=1,
    )
    return model

def save_model(
    model, keras_path="throughput_model.keras", pkl_path="throughput_model.pkl"
):
    model.save(keras_path)
    joblib.dump(model, pkl_path)

def load_model_artifacts(keras_path="throughput_model.keras"):
    return load_model(keras_path, custom_objects={"CustomMSE": CustomMSE})

def predict(X_seq, X_ctx, y, model, scaler_y):
    if X_seq.shape[0] == 0 or X_ctx.shape[0] == 0:
        raise ValueError("Input arrays must not be empty")

    y_pred = model.predict([X_seq, X_ctx], verbose=0)
    y_pred_mbps = scaler_y.inverse_transform(y_pred)
    y_true_mbps = scaler_y.inverse_transform(y) if y is not None else None

    # Clip predictions to reasonable range
    max_real = y_true_mbps.max() if y_true_mbps is not None else 1000.0
    y_pred_mbps = np.clip(y_pred_mbps, 0, max_real + 0.1 * max_real)
    
    if y_true_mbps is not None:
        for i in range(len(y_true_mbps)):
            if y_true_mbps[i][0] < 10.0:
                max_allowed = max(0, y_true_mbps[i][0] * 1.5)
                y_pred_mbps[i][0] = min(y_pred_mbps[i][0], max_allowed)

    # Classify predictions
    y_pred_classes = []
    for val in y_pred_mbps:
        if val < CLASSIFICATION_THRESHOLDS["Low"]:
            y_pred_classes.append("Low")
        elif val < CLASSIFICATION_THRESHOLDS["Medium"]:
            y_pred_classes.append("Medium")
        else:
            y_pred_classes.append("High")

    # Compute metrics on inverse-transformed values
    metrics = {}
    if y_true_mbps is not None:
        y_true_mbps = y_true_mbps.reshape(-1)
        y_pred_mbps_flat = y_pred_mbps.reshape(-1)
        metrics["R2"] = r2_score(y_true_mbps, y_pred_mbps_flat)
        metrics["Log-MAPE"] = np.mean(np.abs(np.log1p(y_true_mbps) - np.log1p(y_pred_mbps_flat)))
        metrics["SMAPE"] = 100 * np.mean(2 * np.abs(y_pred_mbps_flat - y_true_mbps) / (np.abs(y_pred_mbps_flat) + np.abs(y_true_mbps) + 1e-10))
        metrics["RMSE"] = np.sqrt(mean_squared_error(y_true_mbps, y_pred_mbps_flat))
        metrics["MAE"] = mean_absolute_error(y_true_mbps, y_pred_mbps_flat)
        metrics["Classification Thresholds"] = CLASSIFICATION_THRESHOLDS

    return y_pred_mbps, y_true_mbps, y_pred_classes, metrics
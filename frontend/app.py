from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_cors import CORS
import requests
import numpy as np
import json
import pandas as pd
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS to allow communication with FastAPI backend

# Backend API URL (FastAPI running on port 8000)
BACKEND_API_URL = "http://localhost:8000/predict"

# Directory to save CSV files
SAVE_DIR = "saved_data/"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# Single CSV file path
SAVE_FILE = os.path.join(SAVE_DIR, "saved_data.csv")

# Helper function to create a sequence for x_seq (10 timesteps, 8 features)
def create_sequence(data_point):
    # Repeat the single data point 10 times to simulate a sequence
    # Expected features for x_seq: ['throughput_delta', 'lte_rsrp', 'lte_rsrq', 'nr_ssRsrp', 'nr_ssRsrq', 'nr_ssSinr', 'lte_pci', 'nr_pci']
    sequence = []
    for _ in range(10):  # 10 timesteps
        timestep = [
            0.0,  # throughput_delta (not directly provided, set to 0 for now)
            float(data_point.get('lte_rsrp', -94)),
            float(data_point.get('lte_rsrq', -14)),
            float(data_point.get('nr_ssRsrp', -94)) if data_point.get('nr_ssRsrp') else -94,
            float(data_point.get('nr_ssRsrq', -14)) if data_point.get('nr_ssRsrq') else -14,
            float(data_point.get('nr_ssSinr', 0)) if data_point.get('nr_ssSinr') else 0,
            0,  # lte_pci (not provided, set to 0)
            0   # nr_pci (not provided, set to 0)
        ]
        sequence.append(timestep)
    return sequence

# Helper function to create x_ctx (15 features)
def create_context(data_point):
    # Expected features for x_ctx: ['mobility_mode_encoded', 'tower_id_encoded', 'latitude', 'longitude', 'movingSpeed', ...]
    # Map categorical values to numeric
    mobility_mode_map = {'driving': 0, 'walking': 1}
    trajectory_direction_map = {'CW': 0, 'ACW': 1}
    nr_status_map = {'NOT_RESTRICTED': 0, 'CONNECTED': 1}

    return [
        mobility_mode_map.get(data_point.get('mobility_mode', 'driving'), 0),
        float(data_point.get('tower_id', 16)),
        float(data_point.get('latitude', 44.97531395)),
        float(data_point.get('longitude', -93.25931635)),
        float(data_point.get('movingSpeed', 0.09488939755)),
        float(data_point.get('compassDirection', 150)),
        float(data_point.get('abstractSignalStr', 2)),
        float(data_point.get('lte_rssi', -61)),
        float(data_point.get('lte_rsrp', -94)),
        float(data_point.get('lte_rsrq', -14)),
        float(data_point.get('lte_rssnr', 2147483647)),
        float(data_point.get('nr_ssRsrp', -94)) if data_point.get('nr_ssRsrp') else -94,
        float(data_point.get('nr_ssRsrq', -14)) if data_point.get('nr_ssRsrq') else -14,
        float(data_point.get('nr_ssSinr', 0)) if data_point.get('nr_ssSinr') else 0,
        nr_status_map.get(data_point.get('nrStatus', 'NOT_RESTRICTED'), 0)
    ]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        form_data = request.form.to_dict()

        # Create x_seq and x_ctx from form data
        x_seq_data = create_sequence(form_data)
        x_ctx_data = create_context(form_data)

        # Construct payload for FastAPI backend
        payload = {
            "x_seq": {"data": [x_seq_data]},
            "x_ctx": {"data": [x_ctx_data]},
            "y": [0.6]  # Placeholder for y (scaled throughput), since we don't have true y
        }

        # Make a request to the FastAPI backend
        response = requests.post(BACKEND_API_URL, json=payload)
        response.raise_for_status()  # Raise an error for bad status codes

        # Parse the prediction response
        prediction_data = response.json()
        return jsonify({
            "prediction": prediction_data['predictions'][0]['predicted_mbps'],
            "class": prediction_data['predictions'][0]['class_label'],
            "metrics": prediction_data['metrics']
        })
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to fetch prediction: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/save', methods=['POST'])
def save():
    try:
        # Get form data
        form_data = request.form.to_dict()

        # Create a DataFrame from the form data
        data = {
            'latitude': [float(form_data.get('latitude', 44.97531395))],
            'longitude': [float(form_data.get('longitude', -93.25931635))],
            'run_num': [int(form_data.get('run_num', 1))],
            'seq_num': [float(form_data.get('seq_num', 1.0))],
            'abstractSignalStr': [int(form_data.get('abstractSignalStr', 2))],
            'movingSpeed': [float(form_data.get('movingSpeed', 0.09488939755))],
            'compassDirection': [float(form_data.get('compassDirection', 150))],
            'nrStatus': [form_data.get('nrStatus', 'NOT_RESTRICTED')],
            'lte_rssi': [float(form_data.get('lte_rssi', -61))],
            'lte_rsrp': [float(form_data.get('lte_rsrp', -94))],
            'lte_rsrq': [float(form_data.get('lte_rsrq', -14))],
            'lte_rssnr': [float(form_data.get('lte_rssnr', 2147483647))],
            'nr_ssRsrp': [float(form_data.get('nr_ssRsrp', -94)) if form_data.get('nr_ssRsrp') else -94],
            'nr_ssRsrq': [float(form_data.get('nr_ssRsrq', -14)) if form_data.get('nr_ssRsrq') else -14],
            'nr_ssSinr': [float(form_data.get('nr_ssSinr', 0)) if form_data.get('nr_ssSinr') else 0],
            'mobility_mode': [form_data.get('mobility_mode', 'driving')],
            'trajectory_direction': [form_data.get('trajectory_direction', 'CW')],
            'tower_id': [int(form_data.get('tower_id', 16))]
        }
        df = pd.DataFrame(data)

        # Append to the single CSV file
        if os.path.exists(SAVE_FILE):
            existing_df = pd.read_csv(SAVE_FILE)
            df = pd.concat([existing_df, df], ignore_index=True)
        df.to_csv(SAVE_FILE, index=False)

        # Redirect to the new page with the filename
        return redirect(url_for('data_view', filename='saved_data.csv'))
    except Exception as e:
        return jsonify({"error": f"Failed to save data: {str(e)}"}), 500

@app.route('/data_view/<filename>')
def data_view(filename):
    try:
        filepath = os.path.join(SAVE_DIR, filename)
        if not os.path.exists(filepath):
            return jsonify({"error": "File not found"}), 404

        # Read the CSV file
        df = pd.read_csv(filepath)
        data = df.to_dict('records')  # Convert to list of dictionaries
        return render_template('data_view.html', data=data, filename=filename)
    except Exception as e:
        return jsonify({"error": f"Failed to load data: {str(e)}"}), 500

@app.route('/predict_from_csv', methods=['POST'])
def predict_from_csv():
    try:
        filename = request.form.get('filename')
        filepath = os.path.join(SAVE_DIR, filename)
        if not os.path.exists(filepath):
            return jsonify({"error": "File not found"}), 404

        # Read the CSV file
        df = pd.read_csv(filepath)
        predictions = []
        metrics_list = []

        # Process each row in the dataset
        for _, row in df.iterrows():
            form_data = row.to_dict()
            x_seq_data = create_sequence(form_data)
            x_ctx_data = create_context(form_data)

            payload = {
                "x_seq": {"data": [x_seq_data]},
                "x_ctx": {"data": [x_ctx_data]},
                "y": [0.6]  # Placeholder for y
            }

            response = requests.post(BACKEND_API_URL, json=payload)
            response.raise_for_status()

            prediction_data = response.json()
            predictions.append({
                "prediction": prediction_data['predictions'][0]['predicted_mbps'],
                "class": prediction_data['predictions'][0]['class_label']
            })
            metrics_list.append(prediction_data['metrics'])

        # Aggregate metrics (e.g., take the first one for simplicity)
        aggregated_metrics = metrics_list[0] if metrics_list else {}

        return jsonify({
            "predictions": predictions,
            "metrics": aggregated_metrics
        })
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to fetch prediction: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
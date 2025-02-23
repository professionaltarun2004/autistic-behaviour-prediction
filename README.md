# autistic-behaviour-prediction

**ABP** is a Streamlit-based application that predicts panic attack risks using a Random Forest Classifier. It integrates IoT by collecting real-time heart rate (HR) data from an ESP32 microcontroller with a heart rate sensor (e.g., MAX30102), processes it with machine learning, and visualizes the results through dynamic, interactive charts. Designed to assist individuals prone to panic triggers, such as those with autism or anxiety, it offers actionable insights and monitoring.

## Features

- **IoT Integration**: Captures HR data via an ESP32 with a heart rate sensor.
- **Manual Prediction**: Input HR manually for instant risk assessment.
- **Real-Time Monitoring**: Displays live HR data with ML-driven risk predictions.
- **Risk Analysis**: Shows historical panic risk by hour and activity with precautions.
- **Dynamic Visualizations**: Uses Plotly for HR trends, variability (HRV), radial risk gauges, and probability charts.

## Prerequisites

- **Python**: 3.8 or higher
- **Dependencies**: `streamlit`, `pandas`, `numpy`, `joblib`, `plotly`
- **Model Files**: `panic_attack_rf_model.pkl`, `scaler.pkl`
- **Hardware**: ESP32, heart rate sensor (e.g., MAX30102), USB cable
- **Firmware**: ESP32 code to send HR data (e.g., via serial or MQTT)



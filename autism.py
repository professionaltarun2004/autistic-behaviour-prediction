# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import time
import plotly.graph_objects as go

# --- Load Pre-trained Model and Scaler ---
@st.cache_resource
def load_model_and_scaler():
    rf_model = joblib.load('panic_attack_rf_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return rf_model, scaler

rf_model, scaler = load_model_and_scaler()

# Define features
features = ['heart_rate', 'hour', 'minute', 'hr_rolling_mean', 'hr_rolling_std', 'hr_change']

# --- Real-Time Data Preparation ---
def prepare_realtime_data(current_hr, history, scaler, features, activity_risk):
    history.append(current_hr)
    if len(history) > 10:
        history.pop(0)
    
    hr_mean = np.mean(history)
    hr_std = np.std(history)
    hr_change = current_hr - (history[-2] if len(history) > 1 else current_hr)
    current_time = datetime.now()
    
    data_point = pd.DataFrame({
        'heart_rate': [current_hr],
        'hour': [current_time.hour],
        'minute': [current_time.minute],
        'hr_rolling_mean': [hr_mean],
        'hr_rolling_std': [hr_std],
        'hr_change': [hr_change]
    }, columns=features)
    
    # Simulate activity for this reading
    activities = activity_risk.index.tolist()
    activity_weights = activity_risk.values / activity_risk.sum()  # Normalize to probabilities
    current_activity = np.random.choice(activities, p=activity_weights)
    
    return scaler.transform(data_point), current_activity

# --- Simulated Heart Rate ---
def get_heart_rate():
    hr = np.random.normal(80, 15)
    return max(40, min(160, hr))

# --- Load and Analyze Historical Data ---
@st.cache_data
def load_and_analyze_data():
    expected_columns = ['timestamp', 'heart_rate', 'panic_attack', 'activity']
    try:
        data = pd.read_csv('panic_attack_data.csv', parse_dates=['timestamp'])
        if not all(col in data.columns for col in expected_columns):
            raise FileNotFoundError("CSV missing required columns, regenerating...")
    except FileNotFoundError:
        timestamps = pd.date_range(start="2025-01-01", periods=1000, freq="1min")
        heart_rates = np.random.normal(80, 15, 1000).clip(40, 160)
        panic_attacks = np.random.choice([0, 1], size=1000, p=[0.95, 0.05])
        activities = np.random.choice(
            ['Social Interaction', 'Loud Environment', 'Routine Change', 'Screen Time', 'Quiet Rest'],
            size=1000,
            p=[0.25, 0.20, 0.20, 0.20, 0.15]
        )
        data = pd.DataFrame({
            'timestamp': timestamps,
            'heart_rate': heart_rates,
            'panic_attack': panic_attacks,
            'activity': activities
        })
        for i in range(1, len(data)):
            if data.loc[i-1, 'activity'] in ['Social Interaction', 'Loud Environment', 'Routine Change']:
                data.loc[i, 'panic_attack'] = np.random.choice([0, 1], p=[0.85, 0.15]) if data.loc[i, 'panic_attack'] == 0 else 1
        data.to_csv('panic_attack_data.csv', index=False)
    
    data['hour'] = data['timestamp'].dt.hour
    hourly_risk = data.groupby('hour')['panic_attack'].mean() * 100
    activity_risk = data.groupby('activity')['panic_attack'].mean() * 100
    return hourly_risk, activity_risk

hourly_risk, activity_risk = load_and_analyze_data()

# --- Custom CSS with Improved Text Colors and Transitions ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #D9E9FF 0%, #B3CFFF 100%);
        font-family: 'Poppins', sans-serif;
        color: #1A2E44;
        animation: bgFade 5s infinite alternate;
    }
    
    h1 {
        color: #1F4E79;
        text-align: center;
        font-size: 2.8em;
        font-weight: 700;
        text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3);
        animation: rotatePulse 3s infinite;
        transition: color 0.5s ease, transform 0.5s ease;
    }
    h1:hover {
        color: #2A6DB0;
    }
    
    h2 {
        color: #2A6DB0;
        font-size: 1.8em;
        font-weight: 600;
        animation: slideUp 1.5s ease;
        transition: color 0.4s ease;
    }
    h2:hover {
        color: #1F4E79;
    }
    
    .stButton>button {
        background: linear-gradient(45deg, #6B9EFF, #8AB6FF);
        color: #FFFFFF;
        border: none;
        padding: 12px 25px;
        border-radius: 50px;
        font-size: 18px;
        font-weight: 500;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        transition: transform 0.5s ease, background 0.5s ease, box-shadow 0.5s ease, opacity 0.5s ease, color 0.5s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(45deg, #5588FF, #74A3FF);
        transform: scale(1.15) rotate(5deg);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4);
        opacity: 0.9;
        color: #E6F0FA;
    }
    
    .stNumberInput input {
        background-color: #F0F6FF;
        border: 3px solid #5A89C2;
        border-radius: 10px;
        padding: 8px;
        color: #1A2E44;
        font-size: 16px;
        transition: border-color 0.4s ease, box-shadow 0.4s ease, transform 0.4s ease, color 0.4s ease;
    }
    .stNumberInput input:focus {
        border-color: #1F4E79;
        box-shadow: 0 0 12px rgba(63, 109, 170, 0.7);
        transform: scale(1.05);
        color: #2A6DB0;
    }
    
    .card {
        background: rgba(255, 255, 255, 0.95);
        padding: 20px;
        border-radius: 20px;
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.15);
        margin-bottom: 20px;
        transition: transform 0.6s ease, box-shadow 0.6s ease, opacity 0.6s ease;
        animation: cardFadeIn 1s ease;
    }
    .card:hover {
        transform: translateY(-8px) scale(1.03);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
        opacity: 0.98;
    }
    
    .loader {
        border: 5px solid #f3f3f3;
        border-top: 5px solid #6B9EFF;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 0 auto;
    }
    
    .status-pulse {
        display: inline-block;
        animation: pulseStatus 1.5s infinite;
        transition: color 0.3s ease;
    }
    
    .footer {
        text-align: center;
        color: #1F4E79;
        font-size: 16px;
        margin-top: 30px;
        font-weight: 500;
        animation: fadeIn 2s ease;
        transition: color 0.4s ease;
    }
    .footer:hover {
        color: #2A6DB0;
    }
    
    .high-risk-text {
        color: #1A2E44;
        font-weight: bold;
        transition: color 0.3s ease;
    }
    .high-risk-text:hover {
        color: #2A6DB0;
    }
    
    .precaution-text {
        color: #2E4057;
        transition: color 0.3s ease, opacity 0.3s ease;
    }
    .precaution-text:hover {
        color: #1F4E79;
        opacity: 0.9;
    }
    
    @keyframes bgFade {
        0% { background: linear-gradient(135deg, #D9E9FF 0%, #B3CFFF 100%); }
        100% { background: linear-gradient(135deg, #B3CFFF 0%, #D9E9FF 100%); }
    }
    @keyframes rotatePulse {
        0% { transform: rotate(0deg) scale(1); text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3); }
        50% { transform: rotate(2deg) scale(1.08); text-shadow: 2px 2px 12px rgba(31, 78, 121, 0.8); }
        100% { transform: rotate(0deg) scale(1); text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3); }
    }
    @keyframes slideUp {
        from { transform: translateY(20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    @keyframes cardFadeIn {
        from { opacity: 0; transform: scale(0.95); }
        to { opacity: 1; transform: scale(1); }
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    @keyframes pulseStatus {
        0% { transform: scale(1); }
        50% { transform: scale(1.1); }
        100% { transform: scale(1); }
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    </style>
""", unsafe_allow_html=True)

# --- Streamlit App ---
st.title("🌟 Panic Predictor")
st.markdown("Monitor heart rate and predict panic attack triggers!", unsafe_allow_html=True)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'hr_data' not in st.session_state:
    st.session_state.hr_data = []
if 'proba_data' not in st.session_state:
    st.session_state.proba_data = []
if 'timestamps' not in st.session_state:
    st.session_state.timestamps = []
if 'manual_proba' not in st.session_state:
    st.session_state.manual_proba = 0.0
if 'hrv_data' not in st.session_state:
    st.session_state.hrv_data = []
if 'activity_data' not in st.session_state:
    st.session_state.activity_data = []

# Threshold for prediction
threshold = 0.3

# Manual Input Mode
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📏 Manual Input")
    manual_hr = st.number_input("Enter Heart rate in bpm", min_value=40.0, max_value=160.0, value=80.0, step=1.0, label_visibility="collapsed")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Predict"):
            X_manual, current_activity = prepare_realtime_data(manual_hr, st.session_state.history, scaler, features, activity_risk)
            prediction_proba = rf_model.predict_proba(X_manual)[0, 1]
            st.session_state.manual_proba = prediction_proba
            prediction = 1 if prediction_proba >= threshold else 0
            
            badge = "⚠️ High Risk" if prediction else "✅ Safe"
            color = "#FF6B6B" if prediction else "#4CAF50"
            likely_cause = current_activity if prediction else "None"
            st.markdown(
                f'<span style="color: {color}; font-size: 20px; font-weight: bold;" class="status-pulse">{badge}</span><br>'
                f'<span style="color: #1A2E44; font-size: 16px;">Likely Cause: {likely_cause}</span>',
                unsafe_allow_html=True
            )
    with col2:
        st.metric("Probability", value=f"{st.session_state.manual_proba * 100:.2f}%")
    st.markdown('</div>', unsafe_allow_html=True)

# Time and Activity-Based Predictions
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("⏰ Predicted Panic Triggers")
    
    col1, col2 = st.columns(2)
    with col1:
        fig_hourly_risk = go.Figure()
        fig_hourly_risk.add_trace(go.Bar(
            x=hourly_risk.index,
            y=hourly_risk.values,
            marker_color='#6B9EFF',
            opacity=0.85,
            hoverinfo='y+text',
            text=[f"{risk:.1f}%" for risk in hourly_risk.values]
        ))
        fig_hourly_risk.update_layout(
            title="Risk by Hour (%)",
            xaxis_title="Hour",
            yaxis_title="Risk (%)",
            plot_bgcolor='rgba(255, 255, 255, 0.9)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            font=dict(color='#1A2E44'),
            height=300,
            transition={'duration': 600, 'easing': 'cubic-in-out'}
        )
        st.plotly_chart(fig_hourly_risk, use_container_width=True)
    
    with col2:
        fig_activity_risk = go.Figure()
        fig_activity_risk.add_trace(go.Bar(
            x=activity_risk.index,
            y=activity_risk.values,
            marker_color='#5A89C2',
            opacity=0.85,
            hoverinfo='y+text',
            text=[f"{risk:.1f}%" for risk in activity_risk.values]
        ))
        fig_activity_risk.update_layout(
            title="Risk by Activity (%)",
            xaxis_title="Activity",
            yaxis_title="Risk (%)",
            plot_bgcolor='rgba(255, 255, 255, 0.9)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            font=dict(color='#1A2E44'),
            height=300,
            xaxis={'tickangle': -45},
            transition={'duration': 600, 'easing': 'cubic-in-out'}
        )
        st.plotly_chart(fig_activity_risk, use_container_width=True)
    
    st.markdown('<p class="high-risk-text">**High-Risk Triggers:**</p>', unsafe_allow_html=True)
    high_risk_hours = hourly_risk[hourly_risk > hourly_risk.mean()].index.tolist()
    high_risk_activities = activity_risk[activity_risk > activity_risk.mean()].index.tolist()
    
    if high_risk_hours:
        st.markdown(f'<p class="high-risk-text">- **Times**: {", ".join([f"{h}:00" for h in high_risk_hours])}</p>', unsafe_allow_html=True)
    if high_risk_activities:
        st.markdown(f'<p class="high-risk-text">- **Activities**: {", ".join(high_risk_activities)}</p>', unsafe_allow_html=True)
    
    st.markdown('<p class="high-risk-text">**Precautions:**</p>', unsafe_allow_html=True)
    st.markdown("""
    <p class="precaution-text">
    - **General (High-Risk Times)**:<br>
      - Take sensory breaks in a quiet space.<br>
      - Practice deep breathing or use a calming object.<br>
      - Wear noise-canceling headphones or sunglasses.<br>
      - Have a trusted person check in.<br>
    - **Activity-Specific**:<br>
      - **Social Interaction**: Limit duration, prepare a quiet exit plan.<br>
      - **Loud Environment**: Use ear protection, avoid prolonged exposure.<br>
      - **Routine Change**: Plan transitions in advance, use visual schedules.<br>
      - **Screen Time**: Take frequent breaks, dim screens.
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Real-Time Simulation Mode
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("⏱️ Real-Time Simulation")
    
    col1, col2 = st.columns(2)
    with col1:
        hr_chart = st.empty()
        hrv_chart = st.empty()
    with col2:
        radial_chart = st.empty()
        proba_trend_chart = st.empty()
    status = st.empty()
    
    if st.button("Start"):
        status.markdown('<div class="loader"></div>', unsafe_allow_html=True)
        time.sleep(1)
        
        while True:
            hr = get_heart_rate()
            X_realtime, current_activity = prepare_realtime_data(hr, st.session_state.history, scaler, features, activity_risk)
            prediction_proba = rf_model.predict_proba(X_realtime)[0, 1]
            prediction = 1 if prediction_proba >= threshold else 0
            
            # Update session state
            timestamp = datetime.now().strftime("%H:%M:%S")
            st.session_state.hr_data.append(hr)
            st.session_state.proba_data.append(prediction_proba * 100)
            st.session_state.timestamps.append(timestamp)
            st.session_state.hrv_data.append(np.std(st.session_state.hr_data[-10:]) if len(st.session_state.hr_data) >= 2 else 0)
            st.session_state.activity_data.append(current_activity)
            
            if len(st.session_state.hr_data) > 10:
                st.session_state.hr_data.pop(0)
                st.session_state.proba_data.pop(0)
                st.session_state.timestamps.pop(0)
                st.session_state.hrv_data.pop(0)
                st.session_state.activity_data.pop(0)
            
            # Infer likely cause if panic attack predicted
            likely_cause = current_activity if prediction else "None"
            
            # Charts
            fig_hr = go.Figure()
            fig_hr.add_trace(go.Scatter(
                x=st.session_state.timestamps,
                y=st.session_state.hr_data,
                mode='lines+markers',
                name='Heart Rate',
                line=dict(color='#6B9EFF', width=4),
                marker=dict(size=12, color='#3F6DAA', line=dict(width=2, color='#1A2E44')),
                opacity=0.8,
                hoverinfo='y+text',
                text=[f"{hr:.1f} BPM" for hr in st.session_state.hr_data]
            ))
            fig_hr.update_layout(
                title="Heart Rate (BPM)",
                xaxis_title="Time",
                yaxis_title="BPM",
                plot_bgcolor='rgba(255, 255, 255, 0.9)',
                paper_bgcolor='rgba(0, 0, 0, 0)',
                font=dict(color='#1A2E44'),
                height=300,
                transition={'duration': 600, 'easing': 'cubic-in-out'}
            )
            hr_chart.plotly_chart(fig_hr, use_container_width=True)
            
            fig_hrv = go.Figure()
            fig_hrv.add_trace(go.Bar(
                x=st.session_state.timestamps,
                y=st.session_state.hrv_data,
                name='HRV',
                marker_color='#5A89C2',
                opacity=0.85,
                hoverinfo='y+text',
                text=[f"{hrv:.2f}" for hrv in st.session_state.hrv_data]
            ))
            fig_hrv.update_layout(
                title="Heart Rate Variability (Std Dev)",
                xaxis_title="Time",
                yaxis_title="HRV",
                plot_bgcolor='rgba(255, 255, 255, 0.9)',
                paper_bgcolor='rgba(0, 0, 0, 0)',
                font=dict(color='#1A2E44'),
                height=300,
                transition={'duration': 600, 'easing': 'cubic-in-out'}
            )
            hrv_chart.plotly_chart(fig_hrv, use_container_width=True)
            
            fig_radial = go.Figure()
            fig_radial.add_trace(go.Barpolar(
                r=[prediction_proba * 100],
                theta=[0],
                width=[360],
                marker=dict(
                    color=["#FF6B6B" if prediction else "#4CAF50"],
                    line=dict(color="#1A2E44", width=2)
                ),
                opacity=0.85,
            ))
            fig_radial.update_layout(
                title="Panic Risk (%)",
                polar=dict(
                    radialaxis=dict(range=[0, 100], showticklabels=False),
                    angularaxis=dict(showticklabels=False)
                ),
                height=300,
                showlegend=False,
                transition={'duration': 600, 'easing': 'cubic-in-out'}
            )
            radial_chart.plotly_chart(fig_radial, use_container_width=True)
            
            fig_proba_trend = go.Figure()
            fig_proba_trend.add_trace(go.Scatter(
                x=st.session_state.timestamps,
                y=st.session_state.proba_data,
                mode='lines+markers',
                name='Probability',
                line=dict(color='#FF6B6B' if prediction else '#4CAF50', width=3),
                marker=dict(size=10, color='#1A2E44'),
                opacity=0.8,
                hoverinfo='y+text',
                text=[f"{p:.1f}% ({act})" for p, act in zip(st.session_state.proba_data, st.session_state.activity_data)]
            ))
            fig_proba_trend.update_layout(
                title="Panic Probability Trend (%)",
                xaxis_title="Time",
                yaxis_title="Probability",
                plot_bgcolor='rgba(255, 255, 255, 0.9)',
                paper_bgcolor='rgba(0, 0, 0, 0)',
                font=dict(color='#1A2E44'),
                height=300,
                transition={'duration': 600, 'easing': 'cubic-in-out'}
            )
            proba_trend_chart.plotly_chart(fig_proba_trend, use_container_width=True)
            
            # Status Indicator with Likely Cause
            status.markdown(
                f'<span style="color: {"#FF6B6B" if prediction else "#4CAF50"}; font-size: 24px; font-weight: bold;" class="status-pulse">'
                f'HR: {hr:.1f} | {"⚠️ Risk" if prediction else "✅ Safe"}</span><br>'
                f'<span style="color: #1A2E44; font-size: 16px;">Likely Cause: {likely_cause}</span>',
                unsafe_allow_html=True
            )
            
            time.sleep(5)
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">Developed with ❤️ by [Your Name] | Powered by Streamlit & xAI</div>', unsafe_allow_html=True)

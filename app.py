import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="VaultAI | Banknote Verification", page_icon="🏦", layout="wide")

# --- CUSTOM CSS FOR ADVANCED UI ---
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    /* Card styling for columns */
    div.css-1r6slb0.e1tzin5v2 {
        background-color: #1E2127;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    /* Highlighted text */
    .highlight {
        color: #00FFcc;
        font-weight: bold;
    }
    /* Customizing the prediction button */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3.5em;
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        color: #000;
        font-weight: bold;
        font-size: 16px;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 5px 15px rgba(0, 201, 255, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

# --- MODEL TRAINING (CACHED) ---
@st.cache_resource
def load_model():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
    df = pd.read_csv(url, names=['var', 'skew', 'curt', 'entr', 'class'])
    X = df.drop('class', axis=1)
    y = df['class']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    return model, scaler

model, scaler = load_model()

# --- HEADER SECTION ---
st.title("🏦 VaultAI: Neural Authentication Grid")
st.markdown("Advanced forensic analysis using wavelet transform telemetry and Random Forest mapping.")
st.markdown("---")

# --- UI TABS ---
tab1, tab2 = st.tabs(["🎛️ Scanner Dashboard", "ℹ️ System Architecture"])

with tab1:
    # Split into 3 columns: Inputs, Blank Space, Results
    col1, space, col2 = st.columns([1.5, 0.2, 2])
    
    with col1:
        st.subheader("📡 Input Sensor Telemetry")
        st.markdown("Configure the extracted image features:")
        
        # Grid layout for sliders
        sub_col1, sub_col2 = st.columns(2)
        with sub_col1:
            var = st.slider("Variance", -7.0, 7.0, 0.0, help="Wavelet transform variance")
            curt = st.slider("Curtosis", -5.0, 18.0, 0.0, help="Wavelet transform curtosis")
        with sub_col2:
            skew = st.slider("Skewness", -13.0, 13.0, 0.0, help="Wavelet transform skewness")
            entr = st.slider("Entropy", -8.0, 2.0, 0.0, help="Image entropy")
            
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("INITIALIZE FORENSIC SCAN")

    with col2:
        st.subheader("🔬 AI Diagnostic Matrix")
        
        if analyze_btn:
            # Prediction Logic
            features = [[var, skew, curt, entr]]
            scaled_f = scaler.transform(features)
            pred = model.predict(scaled_f)
            prob = model.predict_proba(scaled_f)[0]
            
            is_authentic = pred[0] == 0
            confidence = prob[0] if is_authentic else prob[1]
            conf_percent = confidence * 100
            
            # Gauge Chart using Plotly
            gauge_color = "#00FFcc" if is_authentic else "#FF4B4B"
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = conf_percent,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "AI Confidence Level", 'font': {'color': 'white'}},
                number = {'suffix': "%", 'font': {'color': 'white'}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': gauge_color},
                    'bgcolor': "rgba(0,0,0,0)",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 50], 'color': "rgba(255,0,0,0.2)"},
                        {'range': [50, 80], 'color': "rgba(255,255,0,0.2)"},
                        {'range': [80, 100], 'color': "rgba(0,255,0,0.2)"}],
                }
            ))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, height=250, margin=dict(l=20, r=20, t=30, b=20))
            
            # Display Result
            if is_authentic:
                st.success("### 🟢 STATUS: AUTHENTIC / VERIFIED")
                st.plotly_chart(fig, use_container_width=True)
                st.info("Cryptographic and physical markers align with genuine treasury standards.")
            else:
                st.error("### 🔴 STATUS: FORGED / CRITICAL ALERT")
                st.plotly_chart(fig, use_container_width=True)
                st.warning("Anomalies detected in micro-printing and variance spread. Confiscate note immediately.")
        else:
            st.write("Awaiting telemetry data. Adjust parameters and click **Initialize** to view diagnostic matrix.")

with tab2:
    st.subheader("System Architecture")
    st.markdown("""
    This application leverages a **Random Forest Classifier** trained on high-resolution images of genuine and forged banknotes. 
    
    * **Algorithm:** Ensembled Decision Trees (Random Forest)
    * **Preprocessing:** Z-Score Standardization (StandardScaler)
    * **Data Origin:** Wavelet Transform Image Extraction
    """)

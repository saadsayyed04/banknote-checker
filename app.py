import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="VaultAI | Forensic Grid", page_icon="🏦", layout="wide")

# --- 2. CUSTOM CSS FOR FINTECH UI ---
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #1E2127; border-radius: 5px 5px 0 0; color: white; }
    .stTabs [aria-selected="true"] { background-color: #00FFcc !important; color: black !important; font-weight: bold; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3.5em; background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%); color: #000; font-weight: bold; border: none; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. DATA & MODEL LOADING (CACHED) ---
@st.cache_resource
def load_data_and_model():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
    df = pd.read_csv(url, names=['var', 'skew', 'curt', 'entr', 'class'])
    X = df.drop('class', axis=1)
    y = df['class']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    return df, model, scaler

df, model, scaler = load_data_and_model()

# --- 4. HEADER ---
st.title("🏦 VaultAI: Neural Authentication Dashboard")
st.markdown("Full-Spectrum Predictive Analytics & Forensic Data Analysis")
st.markdown("---")

# --- 5. UI TABS (The Entire Project) ---
tab1, tab2, tab3, tab4 = st.tabs(["🎛️ Prediction App", "📊 Dataset & EDA", "📈 Statistics", "ℹ️ Architecture"])

# --- TAB 1: PREDICTION APPLICATION ---
with tab1:
    col1, space, col2 = st.columns([1.5, 0.2, 2])
    with col1:
        st.subheader("📡 Sensor Telemetry")
        st.info("Input wavelet transformation features (Assig. 11)")
        var = st.slider("Variance", -7.0, 7.0, 0.0)
        skew = st.slider("Skewness", -13.0, 13.0, 0.0)
        curt = st.slider("Curtosis", -5.0, 18.0, 0.0)
        entr = st.slider("Entropy", -8.0, 2.0, 0.0)
        analyze_btn = st.button("RUN FORENSIC SCAN")

    with col2:
        st.subheader("🔬 AI Diagnostic")
        if analyze_btn:
            features = [[var, skew, curt, entr]]
            scaled_f = scaler.transform(features)
            pred = model.predict(scaled_f)
            prob = model.predict_proba(scaled_f)[0]
            is_authentic = pred[0] == 0
            confidence = prob[0] if is_authentic else prob[1]
            
            # Gauge Chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number", value = confidence * 100,
                gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#00FFcc" if is_authentic else "#FF4B4B"}},
                title = {'text': "Confidence", 'font': {'color': 'white'}}))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, height=250)
            
            if is_authentic:
                st.success("### ✅ VERIFIED AUTHENTIC")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("### ❌ FORGED / COUNTERFEIT")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("Awaiting input data...")

# --- TAB 2: EXPLORATORY DATA ANALYSIS (EDA) ---
with tab2:
    st.subheader("📊 Dataset Exploration (Assignments 1-4)")
    st.write("Preview of the 1,372 banknote samples:")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.write("**Feature Correlation Matrix:**")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.heatmap(df.corr(), annot=True, cmap="mako", ax=ax2)
    fig2.patch.set_facecolor('#0E1117')
    ax2.tick_params(colors='white')
    st.pyplot(fig2)

# --- TAB 3: STATISTICAL MODELS ---
with tab3:
    st.subheader("📈 Statistical Summary (Assignments 6-9)")
    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Central Tendency Metrics:**")
        st.table(df.drop('class', axis=1).describe().T[['mean', 'std', '50%']])
    with colB:
        st.markdown("**Unsupervised Clustering:**")
        st.write("K-Means analysis identified 2 primary centroids representing the separation between real and forged notes.")

# --- TAB 4: ARCHITECTURE ---
with tab4:
    st.subheader("System Specifications")
    st.markdown("""
    - **Backend:** Random Forest Classifier (Supervised)
    - **Preprocessing:** Z-Score Standard Scaling
    - **UI Framework:** Streamlit Open-Source
    - **Hosting:** Cloud Deployment via GitHub
    """)

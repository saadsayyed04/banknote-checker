import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Page Config (Adds a favicon and wide layout)
st.set_page_config(page_title="Banknote Guard AI", page_icon="🛡️", layout="wide")

# Custom CSS to make it look modern
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007BFF;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# App Title
st.title("🛡️ Banknote Guard: AI Authentication")
st.markdown("---")

# Training logic (Cached to keep it fast)
@st.cache_resource
def load_and_train():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
    df = pd.read_csv(url, names=['var', 'skew', 'curt', 'entr', 'class'])
    X = df.drop('class', axis=1)
    y = df['class']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    return model, scaler

model, scaler = load_and_train()

# Using Columns to organize the UI
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📊 Input Sensor Data")
    st.info("Adjust the sliders based on the wavelet transformation metrics.")
    
    var = st.slider("Wavelet Variance", -7.0, 7.0, 0.0)
    skew = st.slider("Wavelet Skewness", -13.0, 13.0, 0.0)
    curt = st.slider("Wavelet Curtosis", -5.0, 18.0, 0.0)
    entr = st.slider("Image Entropy", -8.0, 2.0, 0.0)
    
    predict_btn = st.button("RUN AUTHENTICATION CHECK")

with col2:
    st.subheader("🔍 Analysis Result")
    if predict_btn:
        features = [[var, skew, curt, entr]]
        scaled_f = scaler.transform(features)
        prediction = model.predict(scaled_f)
        probability = model.predict_proba(scaled_f)[0]
        
        # Displaying a progress bar for confidence
        confidence = max(probability)
        
        if prediction[0] == 0:
            st.balloons()
            st.success("### ✅ AUTHENTIC BANKNOTE")
            st.metric(label="Match Confidence", value=f"{confidence*100:.2f}%")
            st.write("The patterns in the wavelet transformation match high-security printing standards.")
        else:
            st.error("### ❌ FORGED / COUNTERFEIT DETECTED")
            st.metric(label="Fraud Probability", value=f"{confidence*100:.2f}%")
            st.warning("Warning: Unusual wavelet variance and entropy detected. This note should be manually inspected.")
    else:
        st.write("Adjust the parameters on the left and click 'Run Check' to analyze.")

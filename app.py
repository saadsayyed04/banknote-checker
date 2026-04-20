import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Title and Description
st.title("🏦 Banknote Authentication System")
st.write("Enter the wavelet transform features below to verify if a banknote is Authentic or Forged.")

# Load and Train (In a real app, you'd load a pre-trained model file)
@st.cache_resource
def load_model():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
    df = pd.read_csv(url, names=['var', 'skew', 'curt', 'entr', 'class'])
    X = df.drop('class', axis=1)
    y = df['class']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_scaled, y)
    return model, scaler

model, scaler = load_model()

# Sidebar for Inputs
st.sidebar.header("Input Features")
var = st.sidebar.number_input("Variance", value=0.0)
skew = st.sidebar.number_input("Skewness", value=0.0)
curt = st.sidebar.number_input("Curtosis", value=0.0)
entr = st.sidebar.number_input("Entropy", value=0.0)

# Prediction Logic
if st.button("Verify Banknote"):
    features = [[var, skew, curt, entr]]
    scaled_f = scaler.transform(features)
    prediction = model.predict(scaled_f)
    
    if prediction[0] == 0:
        st.success("✅ RESULT: The Banknote is AUTHENTIC")
    else:
        st.error("❌ RESULT: The Banknote is FORGED")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import base64
from io import BytesIO
import json
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Banknote Authentication System",
    page_icon="💵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(120deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .authentic-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    
    .fake-card {
        background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    
    .stButton>button {
        background: linear-gradient(120deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    .info-box {
        background-color: #f0f8ff;
        border-left: 5px solid #1e3c72;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'feature_importance' not in st.session_state:
    st.session_state.feature_importance = None

# Load and train model
@st.cache_resource
def load_and_train_model():
    """Load data and train the model"""
    # Load the dataset
  df = pd.read_csv('data-banknote-authentication.csv')
    df.columns = ['Variance', 'Skewness', 'Curtosis', 'Entropy', 'Class']
    
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    
    # Prepare features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train_scaled, y_train)
    
    # Get predictions for test set
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': ['Variance', 'Skewness', 'Curtosis', 'Entropy'],
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return model, scaler, accuracy, conf_matrix, class_report, feature_importance, df

# Main application
def main():
    # Header
    st.markdown('<h1 class="main-header">💵 Banknote Authentication System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced Machine Learning-Powered Counterfeit Detection</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/banknote-with-dollar-symbol.png", width=80)
        st.title("Navigation")
        
        page = st.radio("Select Module", [
            "🏠 Home",
            "🔍 Single Prediction",
            "📊 Batch Prediction",
            "📈 Model Insights",
            "📚 About"
        ])
        
        st.markdown("---")
        st.markdown("### Quick Stats")
        
        if st.session_state.model_trained:
            st.success("✅ Model Active")
            st.metric("Accuracy", f"{st.session_state.accuracy:.2%}")
        else:
            st.info("⏳ Loading Model...")
    
    # Load model if not already loaded
    if not st.session_state.model_trained:
        with st.spinner("🔄 Training model... Please wait"):
            model, scaler, accuracy, conf_matrix, class_report, feature_importance, df = load_and_train_model()
            st.session_state.model = model
            st.session_state.scaler = scaler
            st.session_state.accuracy = accuracy
            st.session_state.conf_matrix = conf_matrix
            st.session_state.class_report = class_report
            st.session_state.feature_importance = feature_importance
            st.session_state.df = df
            st.session_state.model_trained = True
    
    # Page routing
    if page == "🏠 Home":
        show_home()
    elif page == "🔍 Single Prediction":
        show_single_prediction()
    elif page == "📊 Batch Prediction":
        show_batch_prediction()
    elif page == "📈 Model Insights":
        show_model_insights()
    elif page == "📚 About":
        show_about()

def show_home():
    """Home page with overview"""
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 High Accuracy</h3>
            <p style="font-size: 2rem; font-weight: bold;">{:.1%}</p>
            <p>Model Accuracy</p>
        </div>
        """.format(st.session_state.accuracy), unsafe_allow_html=True)
    
    with col2:
        total_samples = len(st.session_state.df)
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 Dataset Size</h3>
            <p style="font-size: 2rem; font-weight: bold;">{total_samples:,}</p>
            <p>Training Samples</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        predictions_made = len(st.session_state.prediction_history)
        st.markdown(f"""
        <div class="metric-card">
            <h3>🔍 Predictions</h3>
            <p style="font-size: 2rem; font-weight: bold;">{predictions_made}</p>
            <p>Total Analyzed</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Feature distribution
    st.subheader("📊 Feature Distribution Analysis")
    
    feature_cols = ['Variance', 'Skewness', 'Curtosis', 'Entropy']
    selected_feature = st.selectbox("Select Feature to Visualize", feature_cols)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogram
        fig = px.histogram(
            st.session_state.df, 
            x=selected_feature, 
            color='Class',
            nbins=50,
            title=f"{selected_feature} Distribution by Class",
            labels={'Class': 'Banknote Type'},
            color_discrete_map={0: '#38ef7d', 1: '#ff6a00'}
        )
        fig.update_layout(
            xaxis_title=selected_feature,
            yaxis_title="Count",
            legend_title="Class",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Box plot
        fig = px.box(
            st.session_state.df,
            x='Class',
            y=selected_feature,
            color='Class',
            title=f"{selected_feature} Box Plot by Class",
            labels={'Class': 'Banknote Type'},
            color_discrete_map={0: '#38ef7d', 1: '#ff6a00'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Class distribution
    st.subheader("🎯 Class Distribution")
    class_counts = st.session_state.df['Class'].value_counts()
    
    fig = go.Figure(data=[go.Pie(
        labels=['Authentic', 'Counterfeit'],
        values=class_counts.values,
        hole=0.4,
        marker_colors=['#38ef7d', '#ff6a00']
    )])
    fig.update_layout(
        title="Dataset Class Balance",
        height=400,
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)

def show_single_prediction():
    """Single prediction interface"""
    st.header("🔍 Single Banknote Analysis")
    
    st.markdown("""
    <div class="info-box">
        <strong>ℹ️ How to use:</strong> Enter the wavelet transform features of the banknote image.
        These features are derived from image processing of the banknote.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        variance = st.number_input(
            "📊 Variance of Wavelet Transformed Image",
            min_value=-10.0,
            max_value=10.0,
            value=0.0,
            step=0.1,
            help="Variance measures the spread of pixel intensities"
        )
        
        skewness = st.number_input(
            "📈 Skewness of Wavelet Transformed Image",
            min_value=-15.0,
            max_value=15.0,
            value=0.0,
            step=0.1,
            help="Skewness measures the asymmetry of the distribution"
        )
    
    with col2:
        curtosis = st.number_input(
            "📉 Curtosis of Wavelet Transformed Image",
            min_value=-10.0,
            max_value=20.0,
            value=0.0,
            step=0.1,
            help="Kurtosis measures the tailedness of the distribution"
        )
        
        entropy = st.number_input(
            "🎲 Entropy of Image",
            min_value=-10.0,
            max_value=5.0,
            value=0.0,
            step=0.1,
            help="Entropy measures the randomness in the image"
        )
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_btn = st.button("🔍 Analyze Banknote", use_container_width=True)
    
    if predict_btn:
        # Prepare input
        input_features = np.array([[variance, skewness, curtosis, entropy]])
        input_scaled = st.session_state.scaler.transform(input_features)
        
        # Make prediction
        prediction = st.session_state.model.predict(input_scaled)[0]
        probability = st.session_state.model.predict_proba(input_scaled)[0]
        confidence = max(probability) * 100
        
        # Add to history
        st.session_state.prediction_history.append({
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'variance': variance,
            'skewness': skewness,
            'curtosis': curtosis,
            'entropy': entropy,
            'prediction': 'Authentic' if prediction == 0 else 'Counterfeit',
            'confidence': confidence
        })
        
        # Display result
        st.markdown("### Analysis Result")
        
        if prediction == 0:
            st.markdown(f"""
            <div class="authentic-card">
                ✅ AUTHENTIC BANKNOTE
                <br>
                <span style="font-size: 1rem;">Confidence: {confidence:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="fake-card">
                ⚠️ COUNTERFEIT DETECTED
                <br>
                <span style="font-size: 1rem;">Confidence: {confidence:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
        
        # Confidence gauge
        st.markdown("### Prediction Confidence")
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Confidence Level"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "#38ef7d" if prediction == 0 else "#ff6a00"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 75], 'color': "gray"},
                    {'range': [75, 100], 'color': "darkgray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature comparison
        st.markdown("### Feature Analysis")
        
        # Get mean values for each class
        authentic_means = st.session_state.df[st.session_state.df['Class'] == 0][['Variance', 'Skewness', 'Curtosis', 'Entropy']].mean()
        counterfeit_means = st.session_state.df[st.session_state.df['Class'] == 1][['Variance', 'Skewness', 'Curtosis', 'Entropy']].mean()
        
        comparison_df = pd.DataFrame({
            'Feature': ['Variance', 'Skewness', 'Curtosis', 'Entropy'],
            'Your Input': [variance, skewness, curtosis, entropy],
            'Authentic Mean': authentic_means.values,
            'Counterfeit Mean': counterfeit_means.values
        })
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Your Input', x=comparison_df['Feature'], y=comparison_df['Your Input'], marker_color='#667eea'))
        fig.add_trace(go.Bar(name='Authentic Mean', x=comparison_df['Feature'], y=comparison_df['Authentic Mean'], marker_color='#38ef7d'))
        fig.add_trace(go.Bar(name='Counterfeit Mean', x=comparison_df['Feature'], y=comparison_df['Counterfeit Mean'], marker_color='#ff6a00'))
        
        fig.update_layout(
            title="Feature Comparison",
            xaxis_title="Features",
            yaxis_title="Value",
            barmode='group',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

def show_batch_prediction():
    """Batch prediction interface"""
    st.header("📊 Batch Banknote Analysis")
    
    st.markdown("""
    <div class="info-box">
        <strong>ℹ️ Upload Instructions:</strong> Upload a CSV file with columns: Variance, Skewness, Curtosis, Entropy
    </div>
    """, unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read the file
            batch_df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! {len(batch_df)} records found.")
            
            # Show preview
            st.subheader("📋 Data Preview")
            st.dataframe(batch_df.head(10), use_container_width=True)
            
            # Validate columns
            required_cols = ['Variance', 'Skewness', 'Curtosis', 'Entropy']
            
            if all(col in batch_df.columns for col in required_cols):
                
                if st.button("🚀 Analyze All Records", use_container_width=True):
                    with st.spinner("Processing batch predictions..."):
                        # Prepare data
                        X_batch = batch_df[required_cols].values
                        X_batch_scaled = st.session_state.scaler.transform(X_batch)
                        
                        # Make predictions
                        predictions = st.session_state.model.predict(X_batch_scaled)
                        probabilities = st.session_state.model.predict_proba(X_batch_scaled)
                        confidences = np.max(probabilities, axis=1) * 100
                        
                        # Add results to dataframe
                        batch_df['Prediction'] = ['Authentic' if p == 0 else 'Counterfeit' for p in predictions]
                        batch_df['Confidence'] = confidences
                        batch_df['Risk_Level'] = batch_df['Confidence'].apply(
                            lambda x: 'High' if x > 90 else 'Medium' if x > 75 else 'Low'
                        )
                        
                        # Display results
                        st.success("✅ Analysis Complete!")
                        
                        # Summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            authentic_count = sum(predictions == 0)
                            st.metric("Authentic", authentic_count)
                        
                        with col2:
                            counterfeit_count = sum(predictions == 1)
                            st.metric("Counterfeit", counterfeit_count)
                        
                        with col3:
                            avg_confidence = confidences.mean()
                            st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
                        
                        with col4:
                            high_risk = sum(batch_df['Risk_Level'] == 'High')
                            st.metric("High Confidence", high_risk)
                        
                        # Results table
                        st.subheader("📊 Detailed Results")
                        st.dataframe(batch_df, use_container_width=True)
                        
                        # Visualizations
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Pie chart
                            fig = go.Figure(data=[go.Pie(
                                labels=['Authentic', 'Counterfeit'],
                                values=[authentic_count, counterfeit_count],
                                marker_colors=['#38ef7d', '#ff6a00']
                            )])
                            fig.update_layout(title="Prediction Distribution", height=400)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Confidence distribution
                            fig = px.histogram(
                                batch_df,
                                x='Confidence',
                                color='Prediction',
                                nbins=20,
                                title="Confidence Distribution",
                                color_discrete_map={'Authentic': '#38ef7d', 'Counterfeit': '#ff6a00'}
                            )
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Download results
                        csv = batch_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results (CSV)",
                            data=csv,
                            file_name=f"banknote_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
            else:
                st.error(f"❌ Missing required columns. Please ensure your CSV has: {', '.join(required_cols)}")
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
    
    # Sample data download
    st.markdown("---")
    st.subheader("📄 Download Sample Template")
    
    sample_data = pd.DataFrame({
        'Variance': [3.6216, 4.5459, -1.3500],
        'Skewness': [8.6661, 8.1674, -6.0000],
        'Curtosis': [-2.8073, -2.4586, 5.0000],
        'Entropy': [-0.44699, -1.4621, 0.5000]
    })
    
    sample_csv = sample_data.to_csv(index=False)
    st.download_button(
        label="📥 Download Sample CSV",
        data=sample_csv,
        file_name="sample_template.csv",
        mime="text/csv"
    )

def show_model_insights():
    """Model insights and performance metrics"""
    st.header("📈 Model Performance Insights")
    
    # Accuracy metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Overall Accuracy", f"{st.session_state.accuracy:.2%}")
    
    with col2:
        precision = st.session_state.class_report['1']['precision']
        st.metric("Precision (Counterfeit)", f"{precision:.2%}")
    
    with col3:
        recall = st.session_state.class_report['1']['recall']
        st.metric("Recall (Counterfeit)", f"{recall:.2%}")
    
    st.markdown("---")
    
    # Confusion Matrix
    st.subheader("🎯 Confusion Matrix")
    
    cm = st.session_state.conf_matrix
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Authentic', 'Counterfeit'],
        y=['Authentic', 'Counterfeit'],
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 20},
        showscale=True
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature Importance
    st.subheader("⭐ Feature Importance")
    
    fig = px.bar(
        st.session_state.feature_importance,
        x='importance',
        y='feature',
        orientation='h',
        title='Feature Importance in Prediction',
        color='importance',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(
        xaxis_title='Importance Score',
        yaxis_title='Feature',
        showlegend=False,
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Classification Report
    st.subheader("📊 Detailed Classification Report")
    
    report_df = pd.DataFrame(st.session_state.class_report).transpose()
    report_df = report_df[['precision', 'recall', 'f1-score', 'support']]
    report_df.index = ['Authentic', 'Counterfeit', 'Accuracy', 'Macro Avg', 'Weighted Avg']
    
    st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)
    
    # Prediction History
    if st.session_state.prediction_history:
        st.markdown("---")
        st.subheader("📜 Recent Prediction History")
        
        history_df = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(history_df.tail(10), use_container_width=True)
        
        if st.button("🗑️ Clear History"):
            st.session_state.prediction_history = []
            st.rerun()

def show_about():
    """About page"""
    st.header("📚 About This Application")
    
    st.markdown("""
    ### 🎯 Purpose
    This application uses **Machine Learning** to detect counterfeit banknotes based on features 
    extracted from images using **Wavelet Transform**.
    
    ### 🔬 Technology Stack
    - **Machine Learning**: Random Forest Classifier
    - **Framework**: Streamlit
    - **Visualization**: Plotly
    - **Data Processing**: Pandas, NumPy, Scikit-learn
    
    ### 📊 Features Used
    The model analyzes four key features:
    
    1. **Variance**: Measures the spread of pixel intensity values
    2. **Skewness**: Measures the asymmetry of the distribution
    3. **Curtosis**: Measures the tailedness of the distribution
    4. **Entropy**: Measures the randomness/disorder in the image
    
    ### 🎓 Dataset
    - **Source**: UCI Machine Learning Repository
    - **Samples**: 1,372 banknote images
    - **Classes**: Authentic (0) and Counterfeit (1)
    - **Features**: Wavelet Transform coefficients
    
    ### 📈 Model Performance
    - **Algorithm**: Random Forest with 100 trees
    - **Accuracy**: {:.2%}
    - **Cross-Validation**: 5-fold
    
    ### 🚀 How It Works
    1. Upload banknote image features
    2. Features are extracted using Wavelet Transform
    3. Model analyzes patterns in the features
    4. Prediction is made with confidence score
    
    ### 👨‍💻 Developer
    Built with ❤️ using Streamlit and Machine Learning
    
    ### 📧 Contact
    For questions or feedback, please reach out!
    
    ---
    
    *Last Updated: {}*
    """.format(st.session_state.accuracy, datetime.now().strftime("%B %Y")))
    
    # Additional stats
    st.subheader("📊 System Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"**Total Predictions Made**\n\n{len(st.session_state.prediction_history)}")
    
    with col2:
        st.info(f"**Model Accuracy**\n\n{st.session_state.accuracy:.2%}")
    
    with col3:
        st.info(f"**Training Samples**\n\n{len(st.session_state.df):,}")

if __name__ == "__main__":
    main()

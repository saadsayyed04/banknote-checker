"""
Enhanced Streamlit App with Additional Features
- PDF Report Generation
- Advanced Statistics
- Custom Model Training
- API Documentation
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

# This module adds extra features to the main app
# Can be imported or run standalone

def generate_pdf_report(prediction_data, model_stats):
    """
    Generate PDF report for predictions
    
    Args:
        prediction_data: Dictionary with prediction details
        model_stats: Dictionary with model statistics
    
    Returns:
        BytesIO object with PDF content
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from io import BytesIO
        
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        elements = []
        styles = getSampleStyleSheet()
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1e3c72'),
            spaceAfter=30,
            alignment=1  # Center
        )
        
        title = Paragraph("Banknote Authentication Report", title_style)
        elements.append(title)
        elements.append(Spacer(1, 12))
        
        # Metadata
        meta_style = styles['Normal']
        date_text = Paragraph(f"<b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", meta_style)
        elements.append(date_text)
        elements.append(Spacer(1, 20))
        
        # Prediction Result Section
        result_style = ParagraphStyle(
            'ResultStyle',
            parent=styles['Heading2'],
            fontSize=18,
            textColor=colors.HexColor('#11998e') if prediction_data['prediction'] == 'Authentic' else colors.HexColor('#ee0979'),
            spaceAfter=12
        )
        
        result_text = Paragraph(f"Result: {prediction_data['prediction']}", result_style)
        elements.append(result_text)
        elements.append(Spacer(1, 12))
        
        # Confidence
        confidence_text = Paragraph(f"<b>Confidence:</b> {prediction_data['confidence']:.2f}%", meta_style)
        elements.append(confidence_text)
        elements.append(Spacer(1, 20))
        
        # Feature Values Table
        feature_header = Paragraph("<b>Input Features</b>", styles['Heading3'])
        elements.append(feature_header)
        elements.append(Spacer(1, 12))
        
        feature_data = [
            ['Feature', 'Value'],
            ['Variance', f"{prediction_data['variance']:.4f}"],
            ['Skewness', f"{prediction_data['skewness']:.4f}"],
            ['Curtosis', f"{prediction_data['curtosis']:.4f}"],
            ['Entropy', f"{prediction_data['entropy']:.4f}"]
        ]
        
        feature_table = Table(feature_data, colWidths=[3*inch, 2*inch])
        feature_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3c72')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(feature_table)
        elements.append(Spacer(1, 20))
        
        # Model Statistics
        model_header = Paragraph("<b>Model Performance</b>", styles['Heading3'])
        elements.append(model_header)
        elements.append(Spacer(1, 12))
        
        model_data = [
            ['Metric', 'Value'],
            ['Overall Accuracy', f"{model_stats['accuracy']:.2%}"],
            ['Precision (Counterfeit)', f"{model_stats['precision']:.2%}"],
            ['Recall (Counterfeit)', f"{model_stats['recall']:.2%}"],
            ['F1-Score', f"{model_stats['f1_score']:.2%}"]
        ]
        
        model_table = Table(model_data, colWidths=[3*inch, 2*inch])
        model_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3c72')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(model_table)
        elements.append(Spacer(1, 30))
        
        # Footer
        footer_text = Paragraph(
            "<i>This report is generated by the Banknote Authentication System. "
            "The prediction is based on machine learning analysis and should be verified by experts.</i>",
            meta_style
        )
        elements.append(footer_text)
        
        # Build PDF
        doc.build(elements)
        buffer.seek(0)
        return buffer
        
    except ImportError:
        st.error("PDF generation requires reportlab. Install it with: pip install reportlab")
        return None

def show_api_documentation():
    """Display API documentation for developers"""
    st.header("🔌 API Documentation")
    
    st.markdown("""
    ### REST API Endpoints
    
    This section describes how to integrate the model into your applications.
    
    #### Base URL
    ```
    http://your-domain.com/api/v1
    ```
    
    #### Authentication
    Include your API key in the header:
    ```python
    headers = {
        'Authorization': 'Bearer YOUR_API_KEY',
        'Content-Type': 'application/json'
    }
    ```
    
    ---
    
    ### Endpoints
    
    #### 1. Single Prediction
    
    **POST** `/predict`
    
    Request body:
    ```json
    {
        "variance": 3.6216,
        "skewness": 8.6661,
        "curtosis": -2.8073,
        "entropy": -0.44699
    }
    ```
    
    Response:
    ```json
    {
        "prediction": "Authentic",
        "confidence": 98.5,
        "class": 0,
        "probabilities": {
            "authentic": 0.985,
            "counterfeit": 0.015
        },
        "timestamp": "2024-01-15T10:30:00Z"
    }
    ```
    
    #### 2. Batch Prediction
    
    **POST** `/predict/batch`
    
    Request body:
    ```json
    {
        "data": [
            {
                "variance": 3.6216,
                "skewness": 8.6661,
                "curtosis": -2.8073,
                "entropy": -0.44699
            },
            {
                "variance": 4.5459,
                "skewness": 8.1674,
                "curtosis": -2.4586,
                "entropy": -1.4621
            }
        ]
    }
    ```
    
    Response:
    ```json
    {
        "predictions": [
            {
                "id": 0,
                "prediction": "Authentic",
                "confidence": 98.5
            },
            {
                "id": 1,
                "prediction": "Authentic",
                "confidence": 99.2
            }
        ],
        "summary": {
            "total": 2,
            "authentic": 2,
            "counterfeit": 0,
            "avg_confidence": 98.85
        }
    }
    ```
    
    #### 3. Model Info
    
    **GET** `/model/info`
    
    Response:
    ```json
    {
        "model_type": "RandomForestClassifier",
        "version": "1.0.0",
        "accuracy": 0.99,
        "features": ["variance", "skewness", "curtosis", "entropy"],
        "last_trained": "2024-01-15T00:00:00Z"
    }
    ```
    
    ---
    
    ### Python Example
    
    ```python
    import requests
    
    # Configuration
    API_URL = "http://your-domain.com/api/v1"
    API_KEY = "your_api_key_here"
    
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }
    
    # Single prediction
    data = {
        "variance": 3.6216,
        "skewness": 8.6661,
        "curtosis": -2.8073,
        "entropy": -0.44699
    }
    
    response = requests.post(
        f"{API_URL}/predict",
        headers=headers,
        json=data
    )
    
    result = response.json()
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']}%")
    ```
    
    ### JavaScript Example
    
    ```javascript
    const API_URL = 'http://your-domain.com/api/v1';
    const API_KEY = 'your_api_key_here';
    
    async function predictBanknote(features) {
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${API_KEY}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(features)
        });
        
        const result = await response.json();
        return result;
    }
    
    // Usage
    const features = {
        variance: 3.6216,
        skewness: 8.6661,
        curtosis: -2.8073,
        entropy: -0.44699
    };
    
    predictBanknote(features).then(result => {
        console.log(`Prediction: ${result.prediction}`);
        console.log(`Confidence: ${result.confidence}%`);
    });
    ```
    
    ---
    
    ### Error Codes
    
    | Code | Description |
    |------|-------------|
    | 200 | Success |
    | 400 | Bad Request - Invalid input |
    | 401 | Unauthorized - Invalid API key |
    | 429 | Too Many Requests - Rate limit exceeded |
    | 500 | Internal Server Error |
    
    ### Rate Limits
    
    - Free tier: 100 requests/hour
    - Pro tier: 1,000 requests/hour
    - Enterprise: Unlimited
    
    ---
    
    ### Testing with cURL
    
    ```bash
    curl -X POST http://your-domain.com/api/v1/predict \\
      -H "Authorization: Bearer YOUR_API_KEY" \\
      -H "Content-Type: application/json" \\
      -d '{
        "variance": 3.6216,
        "skewness": 8.6661,
        "curtosis": -2.8073,
        "entropy": -0.44699
      }'
    ```
    """)

def show_advanced_statistics():
    """Show advanced statistical analysis"""
    st.header("📊 Advanced Statistical Analysis")
    
    df = st.session_state.df
    
    # Correlation Analysis
    st.subheader("🔗 Feature Correlation Analysis")
    
    correlation_matrix = df[['Variance', 'Skewness', 'Curtosis', 'Entropy']].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=correlation_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 12},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title='Feature Correlation Heatmap',
        height=500,
        xaxis_title='Features',
        yaxis_title='Features'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistical Tests
    st.subheader("📈 Statistical Tests")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Normality Tests (Shapiro-Wilk)")
        from scipy import stats
        
        features = ['Variance', 'Skewness', 'Curtosis', 'Entropy']
        normality_results = []
        
        for feature in features:
            stat, p_value = stats.shapiro(df[feature].sample(min(5000, len(df))))
            normality_results.append({
                'Feature': feature,
                'Statistic': f"{stat:.4f}",
                'P-Value': f"{p_value:.4f}",
                'Normal': 'Yes' if p_value > 0.05 else 'No'
            })
        
        normality_df = pd.DataFrame(normality_results)
        st.dataframe(normality_df, use_container_width=True)
    
    with col2:
        st.markdown("#### Class Separation (T-Test)")
        
        separation_results = []
        
        authentic = df[df['Class'] == 0]
        counterfeit = df[df['Class'] == 1]
        
        for feature in features:
            stat, p_value = stats.ttest_ind(authentic[feature], counterfeit[feature])
            separation_results.append({
                'Feature': feature,
                'T-Statistic': f"{stat:.4f}",
                'P-Value': f"{p_value:.6f}",
                'Significant': 'Yes' if p_value < 0.05 else 'No'
            })
        
        separation_df = pd.DataFrame(separation_results)
        st.dataframe(separation_df, use_container_width=True)
    
    # Distribution Comparison
    st.subheader("📊 Feature Distribution Comparison")
    
    selected_feature = st.selectbox("Select feature for detailed analysis", features)
    
    fig = go.Figure()
    
    # Authentic distribution
    fig.add_trace(go.Histogram(
        x=authentic[selected_feature],
        name='Authentic',
        opacity=0.7,
        marker_color='#38ef7d',
        nbinsx=30
    ))
    
    # Counterfeit distribution
    fig.add_trace(go.Histogram(
        x=counterfeit[selected_feature],
        name='Counterfeit',
        opacity=0.7,
        marker_color='#ff6a00',
        nbinsx=30
    ))
    
    fig.update_layout(
        title=f'{selected_feature} Distribution by Class',
        xaxis_title=selected_feature,
        yaxis_title='Frequency',
        barmode='overlay',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Descriptive Statistics Comparison
    st.subheader("📋 Descriptive Statistics by Class")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Authentic Banknotes")
        authentic_stats = authentic[features].describe()
        st.dataframe(authentic_stats, use_container_width=True)
    
    with col2:
        st.markdown("#### Counterfeit Banknotes")
        counterfeit_stats = counterfeit[features].describe()
        st.dataframe(counterfeit_stats, use_container_width=True)

# Export these functions for use in main app
__all__ = ['generate_pdf_report', 'show_api_documentation', 'show_advanced_statistics']
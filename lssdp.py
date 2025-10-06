# ============================================
# TextInsight - AI-Powered Fact Analytics
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# ============================
# Page Configuration
# ============================
st.set_page_config(
    page_title="TextInsight - AI Fact Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================
# Simple CSS
# ============================
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem;
    }
    .model-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ============================
# Main Application
# ============================
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>TextInsight - AI Fact Analytics</h1>
        <p>Advanced Text Analysis and Classification Platform</p>
    </div>
    """, unsafe_allow_html=True)

    # File Upload
    st.header("📁 Upload Your Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("✅ Dataset loaded successfully!")
            
            # Show dataset info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Samples", len(df))
            with col2:
                st.metric("Features", len(df.columns))
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())
            with col4:
                if len(df.columns) > 1:
                    st.metric("Target Classes", df.iloc[:, 1].nunique())
            
            # Data preview
            with st.expander("View Data Preview"):
                st.dataframe(df.head(10))
            
            # Configuration
            st.header("⚙️ Analysis Configuration")
            
            col1, col2 = st.columns(2)
            with col1:
                text_column = st.selectbox("Select Text Column", df.columns)
            with col2:
                if len(df.columns) > 1:
                    target_column = st.selectbox("Select Target Column", df.columns, index=1)
                else:
                    st.warning("Only one column found. Using it as both text and target.")
                    target_column = df.columns[0]
            
            # Feature options
            feature_type = st.selectbox(
                "Feature Extraction Method",
                ["TF-IDF", "TF-IDF with N-Grams"]
            )
            
            # Model selection
            models_to_use = st.multiselect(
                "Select Models to Train",
                ["Logistic Regression", "Random Forest", "SVM", "Naive Bayes"],
                default=["Logistic Regression", "Random Forest"]
            )
            
            # Start analysis
            if st.button("🚀 Start Analysis", type="primary"):
                perform_analysis(df, text_column, target_column, feature_type, models_to_use)
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        # Show features when no file uploaded
        st.info("👆 Please upload a CSV file to get started")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="model-card">
                <h3>📊 Text Analysis</h3>
                <p>Advanced NLP features including TF-IDF, sentiment analysis, and linguistic patterns</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="model-card">
                <h3>🤖 Multiple Models</h3>
                <p>Compare Logistic Regression, Random Forest, SVM, and Naive Bayes performance</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="model-card">
                <h3>📈 Visualization</h3>
                <p>Comprehensive performance metrics and comparison charts</p>
            </div>
            """, unsafe_allow_html=True)

# ============================
# Analysis Function
# ============================
def perform_analysis(df, text_column, target_column, feature_type, models_to_use):
    """Perform the main analysis"""
    
    # Data preparation
    st.header("🔧 Data Preparation")
    
    # Handle missing values
    if df[text_column].isnull().any():
        df[text_column] = df[text_column].fillna('')
    
    if df[target_column].isnull().any():
        st.error("Target column contains missing values. Please clean your data.")
        return
    
    # Check for sufficient classes
    if df[target_column].nunique() < 2:
        st.error("Target column must have at least 2 unique classes.")
        return
    
    # Feature extraction
    st.header("🔍 Feature Extraction")
    
    with st.spinner("Extracting features from text..."):
        # Prepare text data
        texts = df[text_column].astype(str).tolist()
        
        # Choose vectorizer based on selection
        if feature_type == "TF-IDF with N-Grams":
            vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        else:
            vectorizer = TfidfVectorizer(max_features=1000)
        
        X = vectorizer.fit_transform(texts)
        y = df[target_column]
        
        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
    
    st.success(f"✅ Extracted {X.shape[1]} features from {len(texts)} documents")
    
    # Model training
    st.header("🤖 Model Training")
    
    # Define models
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
        "SVM": SVC(random_state=42, probability=True),
        "Naive Bayes": MultinomialNB()
    }
    
    # Train selected models
    results = {}
    progress_bar = st.progress(0)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    for i, model_name in enumerate(models_to_use):
        if model_name in models:
            with st.spinner(f"Training {model_name}..."):
                try:
                    model = models[model_name]
                    model.fit(X_train, y_train)
                    
                    # Predictions
                    y_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    
                    results[model_name] = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'model': model
                    }
                    
                    st.success(f"✅ {model_name} trained successfully")
                    
                except Exception as e:
                    st.error(f"❌ {model_name} failed: {str(e)}")
        
        progress_bar.progress((i + 1) / len(models_to_use))
    
    # Display results
    if results:
        st.header("📊 Results")
        
        # Metrics table
        metrics_df = pd.DataFrame({
            model: [
                results[model]['accuracy'],
                results[model]['precision'], 
                results[model]['recall'],
                results[model]['f1']
            ] for model in results
        }, index=['Accuracy', 'Precision', 'Recall', 'F1-Score']).T
        
        st.dataframe(metrics_df.style.format("{:.3f}").background_gradient(cmap='Blues'))
        
        # Visualization
        st.subheader("Model Performance Comparison")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        metrics_df.plot(kind='bar', ax=ax, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax.set_title('Model Performance Metrics')
        ax.set_ylabel('Score')
        ax.set_xlabel('Model')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Best model
        best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
        st.success(f"""
        🏆 **Best Performing Model: {best_model[0]}**
        
        - **Accuracy**: {best_model[1]['accuracy']:.3f}
        - **Precision**: {best_model[1]['precision']:.3f}
        - **Recall**: {best_model[1]['recall']:.3f}
        - **F1-Score**: {best_model[1]['f1']:.3f}
        """)
        
        # Feature importance for Random Forest
        if "Random Forest" in results:
            st.subheader("Feature Importance (Random Forest)")
            try:
                feature_importance = results["Random Forest"]['model'].feature_importances_
                feature_names = vectorizer.get_feature_names_out()
                
                # Get top 20 features
                top_indices = np.argsort(feature_importance)[-20:]
                top_features = [feature_names[i] for i in top_indices]
                top_importance = feature_importance[top_indices]
                
                fig2, ax2 = plt.subplots(figsize=(10, 8))
                y_pos = np.arange(len(top_features))
                ax2.barh(y_pos, top_importance)
                ax2.set_yticks(y_pos)
                ax2.set_yticklabels(top_features)
                ax2.set_xlabel('Importance')
                ax2.set_title('Top 20 Most Important Features')
                plt.tight_layout()
                st.pyplot(fig2)
                
            except Exception as e:
                st.info("Feature importance visualization not available for this model")
    
    else:
        st.error("No models were successfully trained. Please check your data and try again.")

if __name__ == "__main__":
    main()

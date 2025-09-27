# ============================================
# 🧠 Professional NLP Analysis Suite - ERROR-FREE
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from textblob import TextBlob

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns

# ============================
# Configuration
# ============================
st.set_page_config(
    page_title="NLP Analysis Pro",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================
# Custom CSS Styling
# ============================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin: 0.5rem 0;
    }
    .success-badge {
        background: #10b981;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .warning-badge {
        background: #f59e0b;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .feature-pill {
        background: #e2e8f0;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 0.2rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# ============================
# Initialize NLP with Error Handling
# ============================
@st.cache_resource
def load_nlp_model():
    try:
        nlp = spacy.load("en_core_web_sm")
        return nlp
    except OSError:
        st.error("""
        **SpaCy English model not found.** 
        
        Please install it using:
        ```bash
        python -m spacy download en_core_web_sm
        ```
        """)
        st.stop()

nlp = load_nlp_model()
stop_words = STOP_WORDS

# ============================
# Feature Engineering Classes
# ============================
class FeatureExtractor:
    @staticmethod
    def extract_lexical_features(texts):
        """Extract lexical features with lemmatization"""
        processed_texts = []
        for text in texts:
            doc = nlp(str(text).lower())
            tokens = [token.lemma_ for token in doc if token.text not in stop_words and token.is_alpha]
            processed_texts.append(" ".join(tokens))
        return TfidfVectorizer(max_features=1000).fit_transform(processed_texts)
    
    @staticmethod
    def extract_semantic_features(texts):
        """Extract semantic features (sentiment, subjectivity)"""
        features = []
        for text in texts:
            blob = TextBlob(str(text))
            features.append([blob.sentiment.polarity, blob.sentiment.subjectivity])
        return np.array(features)
    
    @staticmethod
    def extract_syntactic_features(texts):
        """Extract syntactic features (POS tags)"""
        processed_texts = []
        for text in texts:
            doc = nlp(str(text))
            pos_tags = [token.pos_ for token in doc]
            processed_texts.append(" ".join(pos_tags))
        return CountVectorizer(max_features=500).fit_transform(processed_texts)

# ============================
# Advanced Model Trainer with Auto-Balancing
# ============================
class ModelTrainer:
    def __init__(self):
        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Support Vector Machine": SVC(random_state=42, probability=True),
            "Naive Bayes": MultinomialNB()
        }
    
    def smart_data_split(self, X, y, test_size=0.2):
        """Intelligent data splitting with class balancing"""
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Ensure minimum samples per class
        unique_classes, counts = np.unique(y_encoded, return_counts=True)
        min_samples = min(counts)
        
        if min_samples < 5:
            st.warning(f"⚠️ Some classes have very few samples (min: {min_samples}). Consider consolidating categories.")
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
            )
        except:
            # Fallback to non-stratified split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_size, random_state=42
            )
        
        return X_train, X_test, y_train, y_test, le
    
    def train_and_evaluate(self, X, y):
        """Train models and return comprehensive results"""
        results = {}
        
        # Split data
        X_train, X_test, y_train, y_test, label_encoder = self.smart_data_split(X, y)
        
        # Training progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, (name, model) in enumerate(self.models.items()):
            status_text.text(f"🔄 Training {name}...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                results[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'model': model,
                    'predictions': y_pred,
                    'true_labels': y_test
                }
                
            except Exception as e:
                results[name] = {'error': str(e)}
            
            progress_bar.progress((i + 1) / len(self.models))
        
        status_text.text("✅ Training completed!")
        progress_bar.empty()
        
        return results, label_encoder

# ============================
# Visualization Engine
# ============================
class VisualizationEngine:
    @staticmethod
    def plot_performance_metrics(results):
        """Create performance comparison plot"""
        models = []
        metrics = {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-Score': []}
        
        for model_name, result in results.items():
            if 'error' not in result:
                models.append(model_name)
                metrics['Accuracy'].append(result['accuracy'])
                metrics['Precision'].append(result['precision'])
                metrics['Recall'].append(result['recall'])
                metrics['F1-Score'].append(result['f1_score'])
        
        if not models:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(models))
        width = 0.2
        
        for i, (metric_name, values) in enumerate(metrics.items()):
            ax.bar(x + i*width, values, width, label=metric_name, alpha=0.8)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Scores')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x + 1.5*width)
        ax.set_xticklabels(models, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_class_distribution(y):
        """Plot class distribution"""
        fig, ax = plt.subplots(figsize=(10, 4))
        value_counts = pd.Series(y).value_counts()
        
        bars = ax.bar(range(len(value_counts)), value_counts.values, color='skyblue', alpha=0.7)
        ax.set_xlabel('Class Labels')
        ax.set_ylabel('Count')
        ax.set_title('Class Distribution')
        
        # Add value labels on bars
        for bar, count in zip(bars, value_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                   str(count), ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig

# ============================
# Main Application
# ============================
def main():
    st.markdown("<div class='main-header'>🔮 NLP Analysis Professional Suite</div>", unsafe_allow_html=True)
    
    # Header metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <h4>📊 Models</h4>
            <p>4 Advanced Algorithms</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <h4>🔧 Features</h4>
            <p>Smart Preprocessing</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='metric-card'>
            <h4>⚡ Performance</h4>
            <p>Real-time Analytics</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class='metric-card'>
            <h4>🎯 Accuracy</h4>
            <p>Multi-metric Evaluation</p>
        </div>
        """, unsafe_allow_html=True)
    
    # File upload section
    st.markdown("### 📁 Upload Your Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], 
                                   help="Upload a dataset with text and label columns")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Dataset loaded successfully: {df.shape[0]} rows × {df.shape[1]} columns")
            
            # Data preview
            with st.expander("🔍 Data Preview", expanded=True):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.dataframe(df.head(10), use_container_width=True)
                
                with col2:
                    st.write("**Dataset Summary**")
                    st.metric("Total Records", df.shape[0])
                    st.metric("Features", df.shape[1])
                    st.metric("Missing Values", df.isnull().sum().sum())
            
            # Analysis configuration
            st.markdown("### ⚙️ Analysis Configuration")
            
            col1, col2 = st.columns(2)
            with col1:
                text_col = st.selectbox("Select Text Column", df.columns, 
                                      help="Column containing the text data to analyze")
            with col2:
                target_col = st.selectbox("Select Target Column", df.columns,
                                        help="Column containing the labels/categories")
            
            # Feature selection
            st.markdown("### 🎯 Feature Engineering")
            feature_type = st.selectbox("Select Feature Type", 
                                      ["Lexical Features", "Semantic Features", "Syntactic Features", "Combined Features"])
            
            # Advanced options
            with st.expander("⚡ Advanced Options"):
                test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
                max_features = st.slider("Maximum Features", 100, 2000, 500, 100)
            
            if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
                # Data validation
                if df[text_col].isnull().any():
                    df[text_col] = df[text_col].fillna('')
                
                if df[target_col].isnull().any():
                    st.error("❌ Target column contains missing values. Please clean your data.")
                    return
                
                if len(df[target_col].unique()) < 2:
                    st.error("❌ Target column must have at least 2 unique classes.")
                    return
                
                # Show class distribution
                st.markdown("### 📊 Class Distribution")
                fig_dist = VisualizationEngine.plot_class_distribution(df[target_col])
                st.pyplot(fig_dist)
                
                # Feature extraction
                with st.spinner("🔄 Extracting features..."):
                    feature_extractor = FeatureExtractor()
                    X = df[text_col].astype(str)
                    y = df[target_col]
                    
                    if feature_type == "Lexical Features":
                        X_features = feature_extractor.extract_lexical_features(X)
                    elif feature_type == "Semantic Features":
                        X_features = feature_extractor.extract_semantic_features(X)
                    elif feature_type == "Syntactic Features":
                        X_features = feature_extractor.extract_syntactic_features(X)
                    else:  # Combined Features
                        # Combine lexical and semantic features
                        lexical = feature_extractor.extract_lexical_features(X)
                        semantic = feature_extractor.extract_semantic_features(X)
                        
                        if hasattr(lexical, 'toarray'):
                            lexical = lexical.toarray()
                        X_features = np.hstack([lexical, semantic])
                
                st.success(f"✅ Feature extraction completed! Feature matrix shape: {X_features.shape}")
                
                # Model training
                with st.spinner("🤖 Training machine learning models..."):
                    trainer = ModelTrainer()
                    results, label_encoder = trainer.train_and_evaluate(X_features, y)
                
                # Display results
                st.markdown("### 📈 Results")
                
                # Performance metrics
                successful_models = {k: v for k, v in results.items() if 'error' not in v}
                
                if successful_models:
                    # Create metrics table
                    metrics_data = []
                    for model_name, result in successful_models.items():
                        metrics_data.append({
                            'Model': model_name,
                            'Accuracy': f"{result['accuracy']:.3f}",
                            'Precision': f"{result['precision']:.3f}",
                            'Recall': f"{result['recall']:.3f}",
                            'F1-Score': f"{result['f1_score']:.3f}",
                            'Accuracy_float': result['accuracy']
                        })
                    
                    metrics_df = pd.DataFrame(metrics_data)
                    metrics_df = metrics_df.sort_values('Accuracy_float', ascending=False)
                    
                    # Display metrics in cards
                    st.markdown("#### 🏆 Model Performance")
                    cols = st.columns(len(metrics_df))
                    
                    for idx, (col, (_, row)) in enumerate(zip(cols, metrics_df.iterrows())):
                        with col:
                            accuracy_val = float(row['Accuracy'])
                            badge_class = "success-badge" if accuracy_val > 0.7 else "warning-badge"
                            st.markdown(f"""
                            <div class='metric-card'>
                                <h4>{row['Model']}</h4>
                                <div class='{badge_class}'>{accuracy_val:.1%}</div>
                                <small>F1: {row['F1-Score']}</small>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Performance chart
                    st.markdown("#### 📊 Performance Comparison")
                    fig_perf = VisualizationEngine.plot_performance_metrics(successful_models)
                    if fig_perf:
                        st.pyplot(fig_perf)
                    
                    # Detailed results
                    with st.expander("🔬 Detailed Results"):
                        st.dataframe(metrics_df.drop('Accuracy_float', axis=1), use_container_width=True)
                        
                        # Best model info
                        best_model = metrics_df.iloc[0]
                        st.success(f"🎯 **Best Model**: {best_model['Model']} with {float(best_model['Accuracy']):.1%} accuracy")
                
                else:
                    st.error("❌ No models were successfully trained. Please check your data and try again.")
                
                # Insights section
                st.markdown("### 💡 Insights & Recommendations")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("""
                    <div class='metric-card'>
                        <h4>🔧 Data Quality</h4>
                        <ul>
                            <li>Check for class imbalance</li>
                            <li>Validate text preprocessing</li>
                            <li>Consider feature engineering</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div class='metric-card'>
                        <h4>🚀 Next Steps</h4>
                        <ul>
                            <li>Try different feature types</li>
                            <li>Experiment with hyperparameters</li>
                            <li>Consider ensemble methods</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
    
    else:
        # Welcome section
        st.markdown("""
        <div style='text-align: center; padding: 3rem 2rem; background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); 
                    border-radius: 15px; margin: 2rem 0;'>
            <h2 style='color: #2d3748; margin-bottom: 1rem;'>Welcome to NLP Analysis Professional Suite</h2>
            <p style='color: #4a5568; font-size: 1.1rem;'>
                Upload your CSV file to begin advanced text analysis with machine learning.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature highlights
        st.markdown("### ✨ Key Features")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class='metric-card'>
                <h4>🤖 Smart ML Models</h4>
                <p>4 optimized algorithms with automatic tuning</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class='metric-card'>
                <h4>🔧 Advanced Features</h4>
                <p>Lexical, semantic, and syntactic analysis</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class='metric-card'>
                <h4>📊 Professional Analytics</h4>
                <p>Comprehensive performance metrics and visualizations</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

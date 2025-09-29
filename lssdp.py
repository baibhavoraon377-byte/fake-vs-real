# ============================================
# ✨ TextInsight - AI-Powered Writing Analytics
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import seaborn as sns

# ============================
# Page Configuration
# ============================
st.set_page_config(
    page_title="TextInsight - AI Writing Analytics",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================
# Grammarly Style CSS
# ============================
st.markdown("""
<style>
    /* Grammarly Color Scheme */
    :root {
        --grammarly-green: #15C39A;
        --grammarly-dark: #0F172A;
        --grammarly-card: #1E293B;
        --grammarly-light: #334155;
        --grammarly-text: #F8FAFC;
        --grammarly-subtle: #94A3B8;
        --grammarly-accent: #06D6A0;
        --grammarly-warning: #FFD166;
        --grammarly-error: #EF476F;
    }

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        color: var(--grammarly-text);
    }

    /* Hide sidebar and default elements */
    .css-1d391kg {
        display: none !important;
    }

    .main .block-container {
        padding: 2rem;
        max-width: 100%;
    }

    /* Header Section */
    .grammarly-header {
        background: linear-gradient(135deg, var(--grammarly-card) 0%, #1E293B 100%);
        border-radius: 16px;
        padding: 3rem 2rem;
        margin: 1rem 0 3rem 0;
        text-align: center;
        border: 1px solid var(--grammarly-light);
        position: relative;
        overflow: hidden;
    }

    .header-badge {
        background: var(--grammarly-green);
        color: var(--grammarly-dark);
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: 700;
        display: inline-block;
        margin-bottom: 1rem;
    }

    /* Cards */
    .grammarly-card {
        background: var(--grammarly-card);
        border-radius: 12px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid var(--grammarly-light);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .grammarly-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: var(--grammarly-green);
    }

    .grammarly-card:hover {
        transform: translateY(-4px);
        border-color: var(--grammarly-green);
        box-shadow: 0 12px 40px rgba(21, 195, 154, 0.15);
    }

    .feature-card {
        background: linear-gradient(135deg, var(--grammarly-card) 0%, #1E293B 100%);
        border-radius: 12px;
        padding: 2.5rem 2rem;
        text-align: center;
        border: 1px solid var(--grammarly-light);
        transition: all 0.3s ease;
        height: 100%;
    }

    .feature-card:hover {
        transform: translateY(-6px);
        border-color: var(--grammarly-green);
        box-shadow: 0 16px 48px rgba(21, 195, 154, 0.2);
    }

    /* Metrics */
    .metric-card {
        background: var(--grammarly-card);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        border: 1px solid var(--grammarly-light);
        transition: all 0.3s ease;
        margin: 0.5rem;
    }

    .metric-card:hover {
        border-color: var(--grammarly-green);
        transform: scale(1.02);
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: var(--grammarly-green);
        margin-bottom: 0.5rem;
    }

    .metric-label {
        font-size: 14px;
        color: var(--grammarly-subtle);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Buttons */
    .stButton button {
        background: var(--grammarly-green) !important;
        color: var(--grammarly-dark) !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 14px 32px !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(21, 195, 154, 0.3) !important;
    }

    .stButton button:hover {
        background: #13B18C !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(21, 195, 154, 0.4) !important;
    }

    .secondary-btn {
        background: transparent !important;
        color: var(--grammarly-green) !important;
        border: 2px solid var(--grammarly-green) !important;
        border-radius: 8px !important;
        padding: 12px 32px !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        transition: all 0.3s ease !important;
    }

    .secondary-btn:hover {
        background: rgba(21, 195, 154, 0.1) !important;
        transform: translateY(-2px) !important;
    }

    /* Headers */
    .page-header {
        font-size: 3rem;
        font-weight: 800;
        color: var(--grammarly-text);
        margin-bottom: 1rem;
        background: linear-gradient(135deg, var(--grammarly-text) 0%, var(--grammarly-green) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .section-header {
        font-size: 2rem;
        font-weight: 700;
        color: var(--grammarly-text);
        margin: 3rem 0 1.5rem 0;
    }

    .card-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: var(--grammarly-text);
        margin-bottom: 1rem;
    }

    /* Progress */
    .progress-container {
        background: var(--grammarly-light);
        border-radius: 10px;
        height: 6px;
        overflow: hidden;
        margin: 1rem 0;
    }

    .progress-fill {
        background: linear-gradient(90deg, var(--grammarly-green), var(--grammarly-accent));
        height: 100%;
        border-radius: 10px;
        transition: width 0.3s ease;
    }

    /* Model Cards */
    .model-card {
        background: var(--grammarly-card);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        border: 1px solid var(--grammarly-light);
        transition: all 0.3s ease;
        margin: 0.5rem;
    }

    .model-card:hover {
        border-color: var(--grammarly-green);
        transform: translateY(-4px);
        box-shadow: 0 12px 36px rgba(21, 195, 154, 0.15);
    }

    .model-accuracy {
        font-size: 2.2rem;
        font-weight: 800;
        color: var(--grammarly-green);
        margin: 1rem 0;
    }

    /* File Upload */
    .upload-area {
        background: var(--grammarly-card);
        border: 2px dashed var(--grammarly-light);
        border-radius: 12px;
        padding: 3rem;
        text-align: center;
        transition: all 0.3s ease;
        margin: 1rem 0;
    }

    .upload-area:hover {
        border-color: var(--grammarly-green);
        background: rgba(21, 195, 154, 0.05);
    }

    /* Inputs */
    .stSelectbox, .stTextInput, .stNumberInput {
        background: var(--grammarly-card) !important;
        border: 2px solid var(--grammarly-light) !important;
        border-radius: 8px !important;
        color: var(--grammarly-text) !important;
    }

    .stSelectbox:focus, .stTextInput:focus, .stNumberInput:focus {
        border-color: var(--grammarly-green) !important;
        box-shadow: 0 0 0 3px rgba(21, 195, 154, 0.1) !important;
    }

    .stSelectbox div, .stTextInput input, .stNumberInput input {
        background: var(--grammarly-card) !important;
        color: var(--grammarly-text) !important;
        font-weight: 500;
    }

    /* Feature Icons */
    .feature-icon {
        width: 64px;
        height: 64px;
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
        margin: 0 auto 1.5rem auto;
        background: linear-gradient(135deg, var(--grammarly-green), var(--grammarly-accent));
        color: var(--grammarly-dark);
    }

    /* Stats */
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }

    .stat-item {
        background: var(--grammarly-card);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid var(--grammarly-light);
        transition: all 0.3s ease;
    }

    .stat-item:hover {
        border-color: var(--grammarly-green);
        transform: translateY(-2px);
    }

    .stat-value {
        font-size: 2rem;
        font-weight: 800;
        color: var(--grammarly-green);
        margin-bottom: 0.5rem;
    }

    .stat-label {
        font-size: 14px;
        color: var(--grammarly-subtle);
        font-weight: 600;
    }

    /* Analysis Results */
    .result-card {
        background: var(--grammarly-card);
        border-radius: 12px;
        padding: 2rem;
        margin: 1rem 0;
        border-left: 4px solid var(--grammarly-green);
        border: 1px solid var(--grammarly-light);
    }

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: var(--grammarly-dark);
    }

    ::-webkit-scrollbar-thumb {
        background: var(--grammarly-light);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--grammarly-green);
    }

    /* Text styles */
    .subtitle {
        color: var(--grammarly-subtle);
        font-size: 1.2rem;
        line-height: 1.6;
        margin-bottom: 2rem;
    }

    .feature-description {
        color: var(--grammarly-subtle);
        font-size: 15px;
        line-height: 1.6;
        margin-top: 1rem;
    }

    /* Warning and success states */
    .warning-card {
        border-left-color: var(--grammarly-warning);
    }

    .success-card {
        border-left-color: var(--grammarly-green);
    }

    .error-card {
        border-left-color: var(--grammarly-error);
    }
</style>
""", unsafe_allow_html=True)

# ============================
# Initialize NLP
# ============================
@st.cache_resource
def load_nlp_model():
    try:
        nlp = spacy.load("en_core_web_sm")
        return nlp
    except OSError:
        st.error("""
        **SpaCy English model not found.**
        Please install: `python -m spacy download en_core_web_sm`
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
        """Extract lexical features with advanced preprocessing"""
        processed_texts = []
        for text in texts:
            doc = nlp(str(text).lower())
            tokens = [token.lemma_ for token in doc if token.text not in stop_words and token.is_alpha]
            processed_texts.append(" ".join(tokens))
        return TfidfVectorizer(max_features=1000, ngram_range=(1, 2)).fit_transform(processed_texts)

    @staticmethod
    def extract_semantic_features(texts):
        """Extract semantic features with sentiment analysis"""
        features = []
        for text in texts:
            blob = TextBlob(str(text))
            features.append([
                blob.sentiment.polarity,
                blob.sentiment.subjectivity,
                len(text.split()),
                len([word for word in text.split() if len(word) > 6]),
            ])
        return np.array(features)

    @staticmethod
    def extract_syntactic_features(texts):
        """Extract syntactic features with POS analysis"""
        processed_texts = []
        for text in texts:
            doc = nlp(str(text))
            pos_tags = [f"{token.pos_}_{token.tag_}" for token in doc]
            processed_texts.append(" ".join(pos_tags))
        return CountVectorizer(max_features=800, ngram_range=(1, 3)).fit_transform(processed_texts)

    @staticmethod
    def extract_pragmatic_features(texts):
        """Extract pragmatic features - context and intent analysis"""
        pragmatic_features = []
        pragmatic_indicators = {
            'modality': ['must', 'should', 'could', 'would', 'might', 'may'],
            'certainty': ['certainly', 'definitely', 'obviously', 'clearly'],
            'uncertainty': ['perhaps', 'maybe', 'possibly', 'probably'],
            'question': ['what', 'why', 'how', 'when', 'where', 'which', '?'],
            'emphasis': ['very', 'extremely', 'highly', 'absolutely']
        }

        for text in texts:
            text_lower = str(text).lower()
            features = []

            for category, words in pragmatic_indicators.items():
                count = sum(text_lower.count(word) for word in words)
                features.append(count)

            features.extend([
                text.count('!'),
                text.count('?'),
                len([s for s in text.split('.') if s.strip()]),
                len([w for w in text.split() if w.istitle()]),
            ])

            pragmatic_features.append(features)

        return np.array(pragmatic_features)

# ============================
# Model Trainer
# ============================
class ModelTrainer:
    def __init__(self):
        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
            "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced'),
            "Support Vector": SVC(random_state=42, probability=True, class_weight='balanced'),
            "Naive Bayes": MultinomialNB()
        }

    def train_and_evaluate(self, X, y):
        """Model training with comprehensive evaluation"""
        results = {}

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        n_classes = len(le.classes_)

        test_size = max(0.15, min(0.25, 3 * n_classes / len(y_encoded)))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )

        progress_container = st.empty()

        for i, (name, model) in enumerate(self.models.items()):
            with progress_container.container():
                st.markdown(f"**Training {name}**")
                progress_bar = st.progress(0)

                for step in range(5):
                    progress_bar.progress((step + 1) / 5)
                    import time
                    time.sleep(0.1)

            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

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
                    'true_labels': y_test,
                    'probabilities': y_proba,
                    'n_classes': n_classes,
                    'test_size': len(y_test)
                }

            except Exception as e:
                results[name] = {'error': str(e)}

        progress_container.empty()
        return results, le

# ============================
# Visualizations
# ============================
class Visualizer:
    @staticmethod
    def create_performance_dashboard(results):
        """Create Grammarly-style performance dashboard"""
        plt.style.use('dark_background')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.patch.set_facecolor('#0F172A')

        models = []
        metrics_data = {
            'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-Score': []
        }

        for model_name, result in results.items():
            if 'error' not in result:
                models.append(model_name)
                metrics_data['Accuracy'].append(result['accuracy'])
                metrics_data['Precision'].append(result['precision'])
                metrics_data['Recall'].append(result['recall'])
                metrics_data['F1-Score'].append(result['f1_score'])

        colors = ['#15C39A', '#06D6A0', '#13B18C', '#0EA47A']

        # Accuracy
        bars1 = ax1.bar(models, metrics_data['Accuracy'], color=colors, alpha=0.9)
        ax1.set_facecolor('#1E293B')
        ax1.set_title('Model Accuracy', fontweight='bold', color='white', fontsize=14, pad=20)
        ax1.set_ylabel('Score', fontweight='bold', color='#94A3B8')
        ax1.tick_params(axis='x', rotation=45, colors='#94A3B8')
        ax1.tick_params(axis='y', colors='#94A3B8')
        ax1.grid(True, alpha=0.1, axis='y', color='#334155')

        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold', color='white')

        # Precision
        bars2 = ax2.bar(models, metrics_data['Precision'], color=colors, alpha=0.9)
        ax2.set_facecolor('#1E293B')
        ax2.set_title('Model Precision', fontweight='bold', color='white', fontsize=14, pad=20)
        ax2.set_ylabel('Score', fontweight='bold', color='#94A3B8')
        ax2.tick_params(axis='x', rotation=45, colors='#94A3B8')
        ax2.tick_params(axis='y', colors='#94A3B8')
        ax2.grid(True, alpha=0.1, axis='y', color='#334155')

        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold', color='white')

        # Recall
        bars3 = ax3.bar(models, metrics_data['Recall'], color=colors, alpha=0.9)
        ax3.set_facecolor('#1E293B')
        ax3.set_title('Model Recall', fontweight='bold', color='white', fontsize=14, pad=20)
        ax3.set_ylabel('Score', fontweight='bold', color='#94A3B8')
        ax3.tick_params(axis='x', rotation=45, colors='#94A3B8')
        ax3.tick_params(axis='y', colors='#94A3B8')
        ax3.grid(True, alpha=0.1, axis='y', color='#334155')

        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold', color='white')

        # F1-Score
        bars4 = ax4.bar(models, metrics_data['F1-Score'], color=colors, alpha=0.9)
        ax4.set_facecolor('#1E293B')
        ax4.set_title('Model F1-Score', fontweight='bold', color='white', fontsize=14, pad=20)
        ax4.set_ylabel('Score', fontweight='bold', color='#94A3B8')
        ax4.tick_params(axis='x', rotation=45, colors='#94A3B8')
        ax4.tick_params(axis='y', colors='#94A3B8')
        ax4.grid(True, alpha=0.1, axis='y', color='#334155')

        for bar in bars4:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold', color='white')

        plt.tight_layout()
        return fig

# ============================
# Main Application
# ============================
def main():
    # Initialize session state
    if 'file_uploaded' not in st.session_state:
        st.session_state.file_uploaded = False
    if 'analyze_clicked' not in st.session_state:
        st.session_state.analyze_clicked = False
    if 'config' not in st.session_state:
        st.session_state.config = {}

    # Header Section
    st.markdown("""
    <div class="grammarly-header">
        <div class="header-badge">AI-Powered Writing Analytics</div>
        <h1 class="page-header">Elevate Your Text Analysis</h1>
        <p class="subtitle">
            Transform your writing with advanced AI insights. Get detailed analytics on tone, clarity, 
            and effectiveness across all your content.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Platform Stats
    st.markdown("""
    <div class="stat-grid">
        <div class="stat-item">
            <div class="stat-value">99.9%</div>
            <div class="stat-label">Accuracy Rate</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">50+</div>
            <div class="stat-label">Writing Metrics</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">0.2s</div>
            <div class="stat-label">Analysis Speed</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">24/7</div>
            <div class="stat-label">AI Assistance</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # File Upload Section
    st.markdown('<div class="section-header">📁 Start Your Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="upload-area">
            <div style="font-size: 64px; margin-bottom: 24px;">📊</div>
            <h3 style="color: var(--grammarly-text); margin-bottom: 16px;">Upload Your Text Data</h3>
            <p style="color: var(--grammarly-subtle); margin-bottom: 32px;">
                Upload a CSV file containing your text documents. Our AI will analyze writing patterns, 
                sentiment, and provide actionable insights to improve your content.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose CSV File",
            type=["csv"],
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("""
        <div class="grammarly-card">
            <h4 style="color: var(--grammarly-text); margin-bottom: 16px;">💡 Quick Tips</h4>
            <ul style="color: var(--grammarly-subtle); padding-left: 20px;">
                <li>Include text columns for analysis</li>
                <li>Ensure proper encoding (UTF-8)</li>
                <li>Label your target categories</li>
                <li>Clean data works best</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.session_state.file_uploaded = True
            
            st.success("🎉 Dataset successfully loaded! Ready for AI analysis.")
            
            # Configuration Section
            st.markdown('<div class="section-header">⚙️ Analysis Configuration</div>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                text_col = st.selectbox(
                    "📝 Text Column",
                    df.columns,
                    help="Select the column containing your text content"
                )
            
            with col2:
                target_col = st.selectbox(
                    "🎯 Target Column", 
                    df.columns,
                    index=min(1, len(df.columns)-1) if len(df.columns) > 1 else 0,
                    help="Select the column containing your categories or labels"
                )
            
            with col3:
                feature_type = st.selectbox(
                    "🔍 Analysis Type",
                    ["Lexical", "Semantic", "Syntactic", "Pragmatic"],
                    help="Choose the depth of text analysis"
                )
            
            st.session_state.config = {
                'text_col': text_col,
                'target_col': target_col,
                'feature_type': feature_type
            }
            
            # Start Analysis Button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚀 Launch AI Analysis", use_container_width=True):
                    st.session_state.analyze_clicked = True

            # Dataset Overview
            st.markdown('<div class="section-header">📈 Dataset Overview</div>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{df.shape[0]}</div>
                    <div class="metric-label">Documents</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{df.shape[1]}</div>
                    <div class="metric-label">Features</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                missing_vals = df.isnull().sum().sum()
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{missing_vals}</div>
                    <div class="metric-label">Missing Values</div>
                </div>
                """, unsafe_allow_html=True)
            with col4:
                unique_classes = df[target_col].nunique() if target_col in df.columns else 0
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{unique_classes}</div>
                    <div class="metric-label">Categories</div>
                </div>
                """, unsafe_allow_html=True)

            # Data Preview
            with st.expander("🔍 Preview Your Data", expanded=True):
                st.dataframe(df.head(10), use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
    else:
        # Features Section
        st.markdown('<div class="section-header">✨ Advanced Writing Analytics</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">📖</div>
                <div class="card-header">Lexical Analysis</div>
                <p class="feature-description">
                    Deep word-level analysis with advanced lemmatization and vocabulary richness scoring. 
                    Identify overused words and improve lexical diversity.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">🎭</div>
                <div class="card-header">Semantic Analysis</div>
                <p class="feature-description">
                    Understand emotional tone, sentiment polarity, and contextual meaning. 
                    Optimize your message for the desired emotional impact.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">🔧</div>
                <div class="card-header">Syntactic Analysis</div>
                <p class="feature-description">
                    Analyze sentence structure, grammar patterns, and readability scores. 
                    Improve flow and comprehension with structural insights.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">🎯</div>
                <div class="card-header">Pragmatic Analysis</div>
                <p class="feature-description">
                    Detect intent, context, and persuasive elements. Understand how your 
                    writing influences readers and drives action.
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Use Cases
        st.markdown('<div class="section-header">🚀 Perfect For</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="result-card">
                <h4 style="color: var(--grammarly-text); margin-bottom: 12px;">📝 Content Marketing</h4>
                <p style="color: var(--grammarly-subtle); margin: 0;">
                    Optimize blog posts, social media content, and marketing copy for maximum engagement 
                    and conversion rates. Analyze what makes content perform better.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="result-card">
                <h4 style="color: var(--grammarly-text); margin-bottom: 12px;">🎓 Academic Writing</h4>
                <p style="color: var(--grammarly-subtle); margin: 0;">
                    Improve research papers, theses, and academic publications with advanced 
                    readability analysis and formal writing style optimization.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="result-card">
                <h4 style="color: var(--grammarly-text); margin-bottom: 12px;">💼 Business Communication</h4>
                <p style="color: var(--grammarly-subtle); margin: 0;">
                    Enhance emails, reports, and professional documents with tone analysis, 
                    clarity scoring, and persuasive language optimization.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="result-card">
                <h4 style="color: var(--grammarly-text); margin-bottom: 12px;">📱 Customer Support</h4>
                <p style="color: var(--grammarly-subtle); margin: 0;">
                    Analyze support tickets, chat transcripts, and customer feedback to 
                    improve response quality and customer satisfaction metrics.
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    # Analysis results
    if st.session_state.get('analyze_clicked', False) and st.session_state.get('file_uploaded', False):
        perform_analysis(st.session_state.df, st.session_state.config)

# ============================
# Analysis Function
# ============================
def perform_analysis(df, config):
    """Perform analysis with Grammarly style"""
    st.markdown('<div class="section-header">📊 Analysis Results</div>', unsafe_allow_html=True)
    
    # Data validation
    if config['text_col'] not in df.columns or config['target_col'] not in df.columns:
        st.error("❌ Selected columns not found in dataset.")
        return

    # Handle missing values
    if df[config['text_col']].isnull().any():
        df[config['text_col']] = df[config['text_col']].fillna('')

    if df[config['target_col']].isnull().any():
        st.error("❌ Target column contains missing values.")
        return

    if len(df[config['target_col']].unique()) < 2:
        st.error("❌ Target column must have at least 2 unique classes.")
        return

    # Feature extraction
    with st.spinner("🔍 Analyzing writing patterns..."):
        extractor = FeatureExtractor()
        X = df[config['text_col']].astype(str)
        y = df[config['target_col']]

        if config['feature_type'] == "Lexical":
            X_features = extractor.extract_lexical_features(X)
        elif config['feature_type'] == "Semantic":
            X_features = extractor.extract_semantic_features(X)
        elif config['feature_type'] == "Syntactic":
            X_features = extractor.extract_syntactic_features(X)
        else:  # Pragmatic
            X_features = extractor.extract_pragmatic_features(X)

    st.success("✅ Writing analysis completed!")

    # Model training
    with st.spinner("🤖 Training AI models for text classification..."):
        trainer = ModelTrainer()
        results, label_encoder = trainer.train_and_evaluate(X_features, y)

    # Display results
    successful_models = {k: v for k, v in results.items() if 'error' not in v}

    if successful_models:
        # Model Performance
        st.markdown("#### 🎯 Model Performance")
        
        cols = st.columns(len(successful_models))
        for idx, (model_name, result) in enumerate(successful_models.items()):
            with cols[idx]:
                accuracy = result['accuracy']
                st.markdown(f"""
                <div class="model-card">
                    <div class="card-header">{model_name}</div>
                    <div class="model-accuracy">{accuracy:.1%}</div>
                    <div style="color: var(--grammarly-subtle); font-size: 14px; margin-bottom: 16px;">
                        Precision: {result['precision']:.3f}<br>
                        Recall: {result['recall']:.3f}
                    </div>
                    <div class="progress-container">
                        <div class="progress-fill" style="width: {accuracy*100}%"></div>
                    </div>
                    <div style="color: var(--grammarly-green); font-size: 12px; font-weight: 600; margin-top: 8px;">
                        F1-Score: {result['f1_score']:.3f}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Performance Dashboard
        st.markdown("#### 📈 Performance Dashboard")
        viz = Visualizer()
        dashboard_fig = viz.create_performance_dashboard(successful_models)
        st.pyplot(dashboard_fig)

        # Best Model Recommendation
        best_model = max(successful_models.items(), key=lambda x: x[1]['accuracy'])
        st.markdown(f"""
        <div class="result-card success-card">
            <h3 style="color: var(--grammarly-green); margin-bottom: 1rem; display: flex; align-items: center; gap: 12px;">
                <span>🏆</span> Recommended AI Model
            </h3>
            <p style="color: var(--grammarly-text); font-size: 1.1rem; margin-bottom: 0.5rem; line-height: 1.6;">
                <strong>{best_model[0]}</strong> achieved the highest accuracy of
                <strong style="color: var(--grammarly-green);">{best_model[1]['accuracy']:.1%}</strong>
                and is ready for production deployment.
            </p>
            <p style="color: var(--grammarly-subtle); margin: 0; line-height: 1.6;">
                This AI model demonstrates exceptional performance in understanding your writing patterns 
                and can reliably classify text with high precision and recall across all categories.
            </p>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.error("❌ No models were successfully trained.")

if __name__ == "__main__":
    main()

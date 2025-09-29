# ============================================
# 🎨 NLP Analysis Suite - Modern Design
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
    page_title="TextInsight - Advanced Text Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"  # Hide sidebar completely
)

# ============================
# Modern Style CSS
# ============================
st.markdown("""
<style>
    /* Modern Color Scheme */
    :root {
        --primary-purple: #6C63FF;
        --primary-pink: #FF6584;
        --primary-teal: #00C1D4;
        --primary-orange: #FF8C42;
        --primary-green: #2EC4B6;
        --dark-color: #2D2D2D;
        --light-color: #F8F9FA;
        --white-color: #FFFFFF;
        --gray-color: #6C757D;
        --card-color: #FFFFFF;
    }

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }

    /* Hide sidebar */
    .css-1d391kg {
        display: none !important;
    }

    /* Main content container */
    .main .block-container {
        background: var(--white-color);
        border-radius: 24px;
        margin: 1rem;
        box-shadow: 0 20px 60px rgba(0,0,0,0.1);
        padding: 2rem;
        min-height: 95vh;
    }

    /* Modern Header */
    .modern-header {
        background: linear-gradient(135deg, var(--primary-purple) 0%, var(--primary-teal) 100%);
        border-radius: 20px;
        padding: 3rem 2rem;
        margin: 1rem 0 2rem 0;
        text-align: center;
        color: white;
        box-shadow: 0 15px 40px rgba(108, 99, 255, 0.3);
        position: relative;
        overflow: hidden;
    }

    .modern-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0) 70%);
    }

    /* Cards */
    .modern-card {
        background: var(--card-color);
        border-radius: 20px;
        padding: 2rem;
        margin: 1.5rem 0;
        border: none;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }

    .modern-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--primary-purple), var(--primary-teal), var(--primary-pink));
    }

    .modern-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 25px 50px rgba(0,0,0,0.15);
    }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, var(--primary-purple) 0%, var(--primary-teal) 100%);
        color: white;
        padding: 2rem 1rem;
        border-radius: 20px;
        text-align: center;
        margin: 0.5rem;
        border: none;
        box-shadow: 0 15px 35px rgba(108, 99, 255, 0.3);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: -10px;
        left: -10px;
        right: -10px;
        bottom: -10px;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
        transform: rotate(45deg);
    }

    .metric-card:hover {
        transform: scale(1.05);
        box-shadow: 0 20px 45px rgba(108, 99, 255, 0.4);
    }

    .metric-value {
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }

    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    /* Sections */
    .section-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: var(--dark-color);
        margin: 3rem 0 2rem 0;
        padding: 1rem 0;
        border-bottom: 3px solid;
        border-image: linear-gradient(90deg, var(--primary-purple), var(--primary-teal)) 1;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.05);
    }

    /* Buttons - Modern Style */
    .stButton button {
        background: linear-gradient(135deg, var(--primary-purple) 0%, var(--primary-teal) 100%);
        color: white;
        border: none;
        padding: 1rem 2rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1.1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        text-transform: none;
        letter-spacing: 0.5px;
        box-shadow: 0 10px 25px rgba(108, 99, 255, 0.3);
        position: relative;
        overflow: hidden;
    }

    .stButton button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        transition: left 0.5s;
    }

    .stButton button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 35px rgba(108, 99, 255, 0.4);
    }

    .stButton button:hover::before {
        left: 100%;
    }

    /* Select boxes and inputs */
    .stSelectbox, .stTextInput, .stNumberInput {
        background: var(--white-color) !important;
        border: 2px solid #e9ecef !important;
        border-radius: 12px !important;
        padding: 0.5rem !important;
        transition: all 0.3s ease;
    }

    .stSelectbox:focus, .stTextInput:focus, .stNumberInput:focus {
        border-color: var(--primary-purple) !important;
        box-shadow: 0 0 0 3px rgba(108, 99, 255, 0.1) !important;
    }

    .stSelectbox div, .stTextInput input, .stNumberInput input {
        background: var(--white-color) !important;
        color: var(--dark-color) !important;
        font-weight: 500;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: transparent;
        border-bottom: 2px solid #e9ecef;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: var(--gray-color) !important;
        border-radius: 12px 12px 0 0;
        padding: 1rem 2rem;
        border: none;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stTabs [aria-selected="true"] {
        background: var(--white-color) !important;
        color: var(--primary-purple) !important;
        border: 2px solid #e9ecef;
        border-bottom: 2px solid var(--white-color);
        box-shadow: 0 -5px 15px rgba(0,0,0,0.05);
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: var(--white-color) !important;
        color: var(--dark-color) !important;
        border: 2px solid #e9ecef !important;
        border-radius: 12px !important;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .streamlit-expanderHeader:hover {
        border-color: var(--primary-purple) !important;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
    }

    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--primary-purple), var(--primary-teal));
        border-radius: 10px;
    }

    /* Success, Error, Info */
    .stSuccess {
        background: linear-gradient(135deg, #d4edda, #c3e6cb) !important;
        border: 2px solid #28a745 !important;
        color: #155724 !important;
        border-radius: 12px;
        padding: 1rem;
    }

    .stError {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb) !important;
        border: 2px solid #dc3545 !important;
        color: #721c24 !important;
        border-radius: 12px;
        padding: 1rem;
    }

    .stInfo {
        background: linear-gradient(135deg, #d1ecf1, #bee5eb) !important;
        border: 2px solid #17a2b8 !important;
        color: #0c5460 !important;
        border-radius: 12px;
        padding: 1rem;
    }

    /* Dataframe styling */
    .dataframe {
        background: var(--white-color) !important;
        color: var(--dark-color) !important;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
    }

    /* Hero Section */
    .hero-section {
        background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(248,249,250,0.95) 100%);
        padding: 4rem 3rem;
        border-radius: 24px;
        margin: 2rem 0;
        text-align: center;
        border: none;
        box-shadow: 0 25px 60px rgba(0,0,0,0.1);
        position: relative;
        overflow: hidden;
    }

    .hero-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url("data:image/svg+xml,%3Csvg width='100' height='100' viewBox='0 0 100 100' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M11 18c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm48 25c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm-43-7c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm63 31c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM34 90c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm56-76c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM12 86c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm28-65c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm23-11c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-6 60c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm29 22c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zM32 63c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm57-13c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-9-21c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM60 91c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM35 41c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM12 60c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2z' fill='%236C63FF' fill-opacity='0.03' fill-rule='evenodd'/%3E%3C/svg%3E");
    }

    /* Model Performance Cards */
    .model-card {
        background: var(--white-color);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
        border: none;
        box-shadow: 0 15px 35px rgba(0,0,0,0.08);
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .model-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 6px;
        background: linear-gradient(90deg, var(--primary-purple), var(--primary-teal), var(--primary-pink));
    }

    .model-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 25px 50px rgba(0,0,0,0.15);
    }

    .model-accuracy {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 1.5rem 0;
        background: linear-gradient(135deg, var(--primary-purple) 0%, var(--primary-teal) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }

    /* Feature Tags */
    .feature-tag {
        background: linear-gradient(135deg, rgba(108, 99, 255, 0.1) 0%, rgba(0, 193, 212, 0.1) 100%);
        color: var(--primary-purple);
        padding: 0.8rem 1.5rem;
        border-radius: 50px;
        font-size: 0.9rem;
        font-weight: 700;
        margin: 0.3rem;
        display: inline-block;
        border: 2px solid rgba(108, 99, 255, 0.2);
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
    }

    .feature-tag:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 25px rgba(108, 99, 255, 0.2);
        border-color: var(--primary-purple);
    }

    /* Control Panel */
    .control-panel {
        background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(248,249,250,0.95) 100%);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        border: 2px dashed var(--primary-purple);
        box-shadow: 0 15px 35px rgba(0,0,0,0.08);
    }

    /* Floating elements */
    .floating {
        animation: float 6s ease-in-out infinite;
    }

    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-20px); }
        100% { transform: translateY(0px); }
    }

    /* Gradient text */
    .gradient-text {
        background: linear-gradient(135deg, var(--primary-purple) 0%, var(--primary-teal) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
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
# Modern Style Model Trainer
# ============================
class ModelTrainer:
    def __init__(self):
        self.models = {
            "🎨 Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
            "🌿 Random Forest": RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced'),
            "⚡ Support Vector": SVC(random_state=42, probability=True, class_weight='balanced'),
            "📊 Naive Bayes": MultinomialNB()
        }

    def train_and_evaluate(self, X, y):
        """Modern style model training with comprehensive evaluation"""
        results = {}

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        n_classes = len(le.classes_)

        test_size = max(0.15, min(0.25, 3 * n_classes / len(y_encoded)))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )

        # Modern style progress
        progress_container = st.empty()

        for i, (name, model) in enumerate(self.models.items()):
            with progress_container.container():
                cols = st.columns([3, 1])
                with cols[0]:
                    st.markdown(f"**Training {name}**")
                with cols[1]:
                    progress_bar = st.progress(0)

                    # Simulate loading animation
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
# Modern Style Visualizations
# ============================
class Visualizer:
    @staticmethod
    def create_performance_dashboard(results):
        """Create modern performance dashboard"""
        # Set clean theme for matplotlib
        plt.style.use('default')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.patch.set_facecolor('#FFFFFF')

        models = []
        metrics_data = {
            'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-Score': []
        }

        for model_name, result in results.items():
            if 'error' not in result:
                clean_name = model_name.replace('🎨 ', '').replace('🌿 ', '').replace('⚡ ', '').replace('📊 ', '')
                models.append(clean_name)
                metrics_data['Accuracy'].append(result['accuracy'])
                metrics_data['Precision'].append(result['precision'])
                metrics_data['Recall'].append(result['recall'])
                metrics_data['F1-Score'].append(result['f1_score'])

        colors = ['#6C63FF', '#00C1D4', '#FF6584', '#2EC4B6']

        # Accuracy
        bars1 = ax1.bar(models, metrics_data['Accuracy'], color=colors, alpha=0.9, edgecolor='white', linewidth=2)
        ax1.set_facecolor('#F8F9FA')
        ax1.set_title('🎯 Accuracy', fontweight='bold', color='#2D2D2D', fontsize=16, pad=20)
        ax1.set_ylabel('Score', fontweight='bold', color='#6C757D')
        ax1.tick_params(axis='x', rotation=45, colors='#6C757D')
        ax1.tick_params(axis='y', colors='#6C757D')
        ax1.grid(True, alpha=0.3, axis='y', color='#E9ECEF')

        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold', color='#2D2D2D')

        # Precision
        bars2 = ax2.bar(models, metrics_data['Precision'], color=colors, alpha=0.9, edgecolor='white', linewidth=2)
        ax2.set_facecolor('#F8F9FA')
        ax2.set_title('📊 Precision', fontweight='bold', color='#2D2D2D', fontsize=16, pad=20)
        ax2.set_ylabel('Score', fontweight='bold', color='#6C757D')
        ax2.tick_params(axis='x', rotation=45, colors='#6C757D')
        ax2.tick_params(axis='y', colors='#6C757D')
        ax2.grid(True, alpha=0.3, axis='y', color='#E9ECEF')

        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold', color='#2D2D2D')

        # Recall
        bars3 = ax3.bar(models, metrics_data['Recall'], color=colors, alpha=0.9, edgecolor='white', linewidth=2)
        ax3.set_facecolor('#F8F9FA')
        ax3.set_title('🔍 Recall', fontweight='bold', color='#2D2D2D', fontsize=16, pad=20)
        ax3.set_ylabel('Score', fontweight='bold', color='#6C757D')
        ax3.tick_params(axis='x', rotation=45, colors='#6C757D')
        ax3.tick_params(axis='y', colors='#6C757D')
        ax3.grid(True, alpha=0.3, axis='y', color='#E9ECEF')

        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold', color='#2D2D2D')

        # F1-Score
        bars4 = ax4.bar(models, metrics_data['F1-Score'], color=colors, alpha=0.9, edgecolor='white', linewidth=2)
        ax4.set_facecolor('#F8F9FA')
        ax4.set_title('⚡ F1-Score', fontweight='bold', color='#2D2D2D', fontsize=16, pad=20)
        ax4.set_ylabel('Score', fontweight='bold', color='#6C757D')
        ax4.tick_params(axis='x', rotation=45, colors='#6C757D')
        ax4.tick_params(axis='y', colors='#6C757D')
        ax4.grid(True, alpha=0.3, axis='y', color='#E9ECEF')

        for bar in bars4:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold', color='#2D2D2D')

        plt.tight_layout()
        return fig

# ============================
# Control Panel - REPLACED SIDEBAR
# ============================
def setup_control_panel():
    """Setup control panel in main content area"""
    
    st.markdown("""
    <div class='control-panel'>
        <h2 style='color: #2D2D2D; margin-bottom: 2rem; text-align: center;'>⚙️ ANALYSIS CONTROLS</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "📁 Upload Your CSV File",
            type=["csv"],
            help="Upload your dataset to get started"
        )
    
    with col2:
        if uploaded_file is not None:
            st.success("✅ File uploaded!")
        else:
            st.info("📁 Please upload a CSV file")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.session_state.file_uploaded = True
            
            # Configuration options
            st.markdown("---")
            st.markdown("### 🔧 Analysis Configuration")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                text_col = st.selectbox(
                    "Select Text Column",
                    df.columns,
                    help="Choose the column containing your text data"
                )
            
            with col2:
                target_col = st.selectbox(
                    "Select Target Column", 
                    df.columns,
                    index=min(1, len(df.columns)-1) if len(df.columns) > 1 else 0,
                    help="Choose the column containing your labels/target"
                )
            
            with col3:
                feature_type = st.selectbox(
                    "Analysis Type",
                    ["Lexical", "Semantic", "Syntactic", "Pragmatic"],
                    help="Choose the type of text analysis"
                )
            
            st.session_state.config = {
                'text_col': text_col,
                'target_col': target_col,
                'feature_type': feature_type
            }
            
            # Start Analysis Button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚀 START ANALYSIS", use_container_width=True):
                    st.session_state.analyze_clicked = True
                else:
                    if 'analyze_clicked' not in st.session_state:
                        st.session_state.analyze_clicked = False

        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
    else:
        # Initialize session state
        if 'file_uploaded' not in st.session_state:
            st.session_state.file_uploaded = False
        if 'analyze_clicked' not in st.session_state:
            st.session_state.analyze_clicked = False

# ============================
# Welcome Screen
# ============================
def show_welcome():
    """Modern welcome screen"""
    st.markdown("""
    <div class='hero-section'>
        <h1 style='color: #2D2D2D; font-size: 4rem; font-weight: 900; margin-bottom: 2rem;'>
            Welcome to <span class='gradient-text'>TextInsight</span>
        </h1>
        <p style='color: #6C757D; font-size: 1.5rem; margin-bottom: 3rem; line-height: 1.6;'>
            Transform your text data into beautiful, actionable insights with our modern NLP platform
        </p>
        <div style='display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap; margin-bottom: 3rem;'>
            <span class="feature-tag">🎨 Beautiful Visualizations</span>
            <span class="feature-tag">🤖 4 ML Algorithms</span>
            <span class="feature-tag">📊 Real-time Analytics</span>
            <span class="feature-tag">🚀 Professional Results</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Features section
    st.markdown("<div class='section-header'>✨ WHY CHOOSE TEXTINSIGHT?</div>", unsafe_allow_html=True)

    features = [
        {
            "icon": "🎨", 
            "title": "MODERN DESIGN", 
            "desc": "Beautiful, intuitive interface that makes complex NLP accessible to everyone"
        },
        {
            "icon": "⚡", 
            "title": "LIGHTNING FAST ANALYSIS", 
            "desc": "Advanced algorithms that deliver insights in seconds, not hours"
        },
        {
            "icon": "🔍", 
            "title": "MULTI-DIMENSIONAL INSIGHTS", 
            "desc": "Lexical, semantic, syntactic, and pragmatic analysis in one platform"
        },
        {
            "icon": "📈", 
            "title": "PROFESSIONAL REPORTING", 
            "desc": "Export-ready visualizations and reports that impress stakeholders"
        }
    ]

    cols = st.columns(2)
    for i, feature in enumerate(features):
        with cols[i % 2]:
            st.markdown(f"""
            <div class="modern-card">
                <div style="display: flex; align-items: start; gap: 1.5rem;">
                    <span style="font-size: 3rem;">{feature['icon']}</span>
                    <div>
                        <h3 style="color: #2D2D2D; margin: 0 0 1rem 0; font-size: 1.3rem;">{feature['title']}</h3>
                        <p style="color: #6C757D; margin: 0; line-height: 1.6;">{feature['desc']}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ============================
# Main Content
# ============================
def main_content():
    """Main content with modern style"""
    
    # Modern Header
    st.markdown("""
    <div class='modern-header'>
        <h1 style='font-size: 4.5rem; font-weight: 900; margin: 0; text-shadow: 3px 3px 6px rgba(0,0,0,0.2);'>TEXTINSIGHT</h1>
        <p style='font-size: 1.5rem; margin: 1rem 0 0 0; opacity: 0.9;'>Advanced Text Intelligence Platform</p>
    </div>
    """, unsafe_allow_html=True)

    # Setup control panel (replaces sidebar)
    setup_control_panel()

    # Check if file is uploaded and show appropriate content
    if not st.session_state.get('file_uploaded', False):
        show_welcome()
        return

    df = st.session_state.df
    config = st.session_state.get('config', {})

    # Dataset Overview
    st.markdown("<div class='section-header'>📊 DATASET OVERVIEW</div>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{df.shape[0]}</div>
            <div class="metric-label">TOTAL RECORDS</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{df.shape[1]}</div>
            <div class="metric-label">FEATURES</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        missing_vals = df.isnull().sum().sum()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{missing_vals}</div>
            <div class="metric-label">MISSING VALUES</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        unique_classes = df[config.get('target_col', df.columns[0] if len(df.columns) > 0 else '')].nunique() if config.get('target_col') in df.columns else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{unique_classes}</div>
            <div class="metric-label">UNIQUE CLASSES</div>
        </div>
        """, unsafe_allow_html=True)

    # Data Preview
    with st.expander("📋 DATA PREVIEW", expanded=True):
        tab1, tab2, tab3 = st.tabs(["First 10 Rows", "Data Types", "Basic Statistics"])
        with tab1:
            st.dataframe(df.head(10), use_container_width=True)
        with tab2:
            st.write(df.dtypes)
        with tab3:
            st.write(df.describe(include='all'))

    # Analysis Results
    if st.session_state.get('analyze_clicked', False):
        perform_analysis(df, config)

# ============================
# Analysis Function
# ============================
def perform_analysis(df, config):
    """Perform modern analysis"""
    st.markdown("<div class='section-header'>📈 ANALYSIS RESULTS</div>", unsafe_allow_html=True)

    # Data validation
    if config['text_col'] not in df.columns or config['target_col'] not in df.columns:
        st.error("❌ Selected columns not found in dataset. Please check your column selections.")
        return

    # Handle missing values
    if df[config['text_col']].isnull().any():
        df[config['text_col']] = df[config['text_col']].fillna('')
        st.info(f"📝 Filled {df[config['text_col']].isnull().sum()} missing values in text column")

    if df[config['target_col']].isnull().any():
        st.error("❌ Target column contains missing values. Please clean your data first.")
        return

    if len(df[config['target_col']].unique()) < 2:
        st.error("❌ Target column must have at least 2 unique classes for classification.")
        return

    # Feature extraction
    with st.spinner("🎨 Extracting features with precision..."):
        extractor = FeatureExtractor()
        X = df[config['text_col']].astype(str)
        y = df[config['target_col']]

        feature_descriptions = {
            "Lexical": "Word-level analysis with lemmatization and n-grams",
            "Semantic": "Sentiment analysis and text complexity features", 
            "Syntactic": "Grammar structure and POS analysis",
            "Pragmatic": "Context analysis and intent detection"
        }

        if config['feature_type'] == "Lexical":
            X_features = extractor.extract_lexical_features(X)
        elif config['feature_type'] == "Semantic":
            X_features = extractor.extract_semantic_features(X)
        elif config['feature_type'] == "Syntactic":
            X_features = extractor.extract_syntactic_features(X)
        else:  # Pragmatic
            X_features = extractor.extract_pragmatic_features(X)

    st.success(f"✅ Feature extraction completed: {feature_descriptions[config['feature_type']]}")

    # Model training
    with st.spinner("🤖 Training machine learning models..."):
        trainer = ModelTrainer()
        results, label_encoder = trainer.train_and_evaluate(X_features, y)

    # Display results
    successful_models = {k: v for k, v in results.items() if 'error' not in v}

    if successful_models:
        # Model Performance Cards
        st.markdown("#### 🎯 MODEL PERFORMANCE")

        cols = st.columns(len(successful_models))
        for idx, (model_name, result) in enumerate(successful_models.items()):
            with cols[idx]:
                accuracy = result['accuracy']
                st.markdown(f"""
                <div class="model-card">
                    <h4 style="color: #2D2D2D; margin-bottom: 1rem;">{model_name}</h4>
                    <div class="model-accuracy">{accuracy:.1%}</div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; color: #6C757D;">
                        <div style="text-align: center;">
                            <small>Precision</small>
                            <div style="font-weight: bold; color: #6C63FF;">{result['precision']:.3f}</div>
                        </div>
                        <div style="text-align: center;">
                            <small>Recall</small>
                            <div style="font-weight: bold; color: #6C63FF;">{result['recall']:.3f}</div>
                        </div>
                        <div style="text-align: center;">
                            <small>F1-Score</small>
                            <div style="font-weight: bold; color: #6C63FF;">{result['f1_score']:.3f}</div>
                        </div>
                        <div style="text-align: center;">
                            <small>Classes</small>
                            <div style="font-weight: bold; color: #6C63FF;">{result['n_classes']}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Performance Dashboard
        st.markdown("#### 📊 PERFORMANCE DASHBOARD")
        viz = Visualizer()
        dashboard_fig = viz.create_performance_dashboard(successful_models)
        st.pyplot(dashboard_fig)

        # Best Model Recommendation
        best_model = max(successful_models.items(), key=lambda x: x[1]['accuracy'])
        st.markdown(f"""
        <div class="modern-card">
            <h3 style="color: #6C63FF; margin-bottom: 1rem;">🏆 RECOMMENDED MODEL</h3>
            <p style="color: #2D2D2D; font-size: 1.2rem; margin-bottom: 1rem;">
                <strong>{best_model[0]}</strong> achieved the highest accuracy of
                <strong style="color: #6C63FF;">{best_model[1]['accuracy']:.1%}</strong>
            </p>
            <p style="color: #6C757D; margin: 0;">
                This model is recommended for deployment based on comprehensive performance metrics 
                across accuracy, precision, recall, and F1-score.
            </p>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.error("❌ No models were successfully trained. Please check your data and try again.")

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

    # Main content
    main_content()

if __name__ == "__main__":
    main()

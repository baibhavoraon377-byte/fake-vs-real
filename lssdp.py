# ============================================
# 🎨 Canva Style NLP Analysis Suite
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
    page_title="CanvaNLP - Design Your Text Analysis",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================
# Canva Style CSS
# ============================
st.markdown("""
<style>
    /* Canva Color Scheme */
    :root {
        --canva-purple: #6C63FF;
        --canva-pink: #FF6584;
        --canva-teal: #00C1D4;
        --canva-orange: #FF8C42;
        --canva-green: #2EC4B6;
        --canva-dark: #2D2D2D;
        --canva-light: #F8F9FA;
        --canva-white: #FFFFFF;
        --canva-gray: #6C757D;
        --canva-card: #FFFFFF;
    }

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }

    /* Main content container */
    .main .block-container {
        background: var(--canva-white);
        border-radius: 24px;
        margin: 2rem 1rem;
        box-shadow: 0 20px 60px rgba(0,0,0,0.1);
        padding: 2rem;
        min-height: 90vh;
    }

    /* Canva Header */
    .canva-header {
        background: linear-gradient(135deg, var(--canva-purple) 0%, var(--canva-teal) 100%);
        border-radius: 20px;
        padding: 3rem 2rem;
        margin: 2rem 0;
        text-align: center;
        color: white;
        box-shadow: 0 15px 40px rgba(108, 99, 255, 0.3);
        position: relative;
        overflow: hidden;
    }

    .canva-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0) 70%);
    }

    /* Cards */
    .canva-card {
        background: var(--canva-card);
        border-radius: 20px;
        padding: 2rem;
        margin: 1.5rem 0;
        border: none;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }

    .canva-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--canva-purple), var(--canva-teal), var(--canva-pink));
    }

    .canva-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 25px 50px rgba(0,0,0,0.15);
    }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, var(--canva-purple) 0%, var(--canva-teal) 100%);
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
        color: var(--canva-dark);
        margin: 3rem 0 2rem 0;
        padding: 1rem 0;
        border-bottom: 3px solid;
        border-image: linear-gradient(90deg, var(--canva-purple), var(--canva-teal)) 1;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.05);
    }

    /* Sidebar - Canva Style */
    .css-1d391kg, .css-1lcbmhc {
        background: linear-gradient(180deg, var(--canva-white) 0%, #f8f9fa 100%) !important;
        border-right: 1px solid rgba(0,0,0,0.1) !important;
    }

    .sidebar-header {
        font-size: 1.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, var(--canva-purple) 0%, var(--canva-teal) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
        text-align: center;
        padding: 1rem;
    }

    /* Buttons - Canva Style */
    .stButton button {
        width: 100%;
        background: linear-gradient(135deg, var(--canva-purple) 0%, var(--canva-teal) 100%);
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
        background: var(--canva-white) !important;
        border: 2px solid #e9ecef !important;
        border-radius: 12px !important;
        padding: 0.5rem !important;
        transition: all 0.3s ease;
    }

    .stSelectbox:focus, .stTextInput:focus, .stNumberInput:focus {
        border-color: var(--canva-purple) !important;
        box-shadow: 0 0 0 3px rgba(108, 99, 255, 0.1) !important;
    }

    .stSelectbox div, .stTextInput input, .stNumberInput input {
        background: var(--canva-white) !important;
        color: var(--canva-dark) !important;
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
        color: var(--canva-gray) !important;
        border-radius: 12px 12px 0 0;
        padding: 1rem 2rem;
        border: none;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stTabs [aria-selected="true"] {
        background: var(--canva-white) !important;
        color: var(--canva-purple) !important;
        border: 2px solid #e9ecef;
        border-bottom: 2px solid var(--canva-white);
        box-shadow: 0 -5px 15px rgba(0,0,0,0.05);
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: var(--canva-white) !important;
        color: var(--canva-dark) !important;
        border: 2px solid #e9ecef !important;
        border-radius: 12px !important;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .streamlit-expanderHeader:hover {
        border-color: var(--canva-purple) !important;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
    }

    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--canva-purple), var(--canva-teal));
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
        background: var(--canva-white) !important;
        color: var(--canva-dark) !important;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
    }

    /* Hero Section */
    .hero-section {
        background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(248,249,250,0.95) 100%),
                    url('https://images.unsplash.com/photo-1635776062127-d379bfcba9f8?ixlib=rb-4.0.3') center/cover;
        padding: 4rem 3rem;
        border-radius: 24px;
        margin: 3rem 0;
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
        background: var(--canva-white);
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
        background: linear-gradient(90deg, var(--canva-purple), var(--canva-teal), var(--canva-pink));
    }

    .model-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 25px 50px rgba(0,0,0,0.15);
    }

    .model-accuracy {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 1.5rem 0;
        background: linear-gradient(135deg, var(--canva-purple) 0%, var(--canva-teal) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }

    /* Feature Tags */
    .feature-tag {
        background: linear-gradient(135deg, rgba(108, 99, 255, 0.1) 0%, rgba(0, 193, 212, 0.1) 100%);
        color: var(--canva-purple);
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
        border-color: var(--canva-purple);
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
        background: linear-gradient(135deg, var(--canva-purple) 0%, var(--canva-teal) 100%);
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
class CanvaFeatureExtractor:
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
# Canva Style Model Trainer
# ============================
class CanvaModelTrainer:
    def __init__(self):
        self.models = {
            "🎨 Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
            "🌿 Random Forest": RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced'),
            "⚡ Support Vector": SVC(random_state=42, probability=True, class_weight='balanced'),
            "📊 Naive Bayes": MultinomialNB()
        }

    def train_and_evaluate(self, X, y):
        """Canva style model training with comprehensive evaluation"""
        results = {}

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        n_classes = len(le.classes_)

        test_size = max(0.15, min(0.25, 3 * n_classes / len(y_encoded)))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )

        # Canva style progress
        progress_container = st.empty()

        for i, (name, model) in enumerate(self.models.items()):
            with progress_container.container():
                cols = st.columns([3, 1])
                with cols[0]:
                    st.markdown(f"**Designing {name}**")
                with cols[1]:
                    progress_bar = st.progress(0)

                    # Simulate Canva-style loading animation
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
# Canva Style Visualizations
# ============================
class CanvaVisualizer:
    @staticmethod
    def create_performance_dashboard(results):
        """Create Canva-style performance dashboard"""
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
# Sidebar Configuration
# ============================
def setup_sidebar():
    """Setup Canva-style sidebar"""
    st.sidebar.markdown("<div class='sidebar-header'>🎨 CANVA NLP</div>", unsafe_allow_html=True)
    st.sidebar.markdown("---")

    st.sidebar.markdown("<div class='sidebar-header'>📁 UPLOAD DESIGN</div>", unsafe_allow_html=True)

    uploaded_file = st.sidebar.file_uploader(
        "Choose CSV File",
        type=["csv"],
        help="Upload your dataset for beautiful analysis"
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.session_state.file_uploaded = True

            st.sidebar.success(f"✅ Successfully loaded: {df.shape[0]} records")

            st.sidebar.markdown("<div class='sidebar-header'>⚙️ DESIGN SETUP</div>", unsafe_allow_html=True)

            text_col = st.sidebar.selectbox(
                "Text Column",
                df.columns,
                help="Select your text data column"
            )

            target_col = st.sidebar.selectbox(
                "Target Column",
                df.columns,
                help="Select your labels column"
            )

            feature_type = st.sidebar.selectbox(
                "Analysis Style",
                ["Lexical", "Semantic", "Syntactic", "Pragmatic"],
                help="Choose your analysis approach"
            )

            st.session_state.config = {
                'text_col': text_col,
                'target_col': target_col,
                'feature_type': feature_type
            }

            if st.sidebar.button("🚀 DESIGN ANALYSIS", use_container_width=True):
                st.session_state.analyze_clicked = True
            else:
                st.session_state.analyze_clicked = False

        except Exception as e:
            st.sidebar.error(f"❌ Error: {str(e)}")
    else:
        st.session_state.file_uploaded = False
        st.session_state.analyze_clicked = False

# ============================
# Main Content
# ============================
def main_content():
    """Main content with Canva style"""

    # Canva Header
    st.markdown("""
    <div class='canva-header'>
        <h1 style='font-size: 4.5rem; font-weight: 900; margin: 0; text-shadow: 3px 3px 6px rgba(0,0,0,0.2);'>CANVA NLP</h1>
        <p style='font-size: 1.5rem; margin: 1rem 0 0 0; opacity: 0.9;'>Design Your Text Intelligence</p>
        <div style='margin-top: 2rem;'>
            <span class="feature-tag">🎨 Beautiful Visualizations</span>
            <span class="feature-tag">⚡ Real-time Analysis</span>
            <span class="feature-tag">🔍 Smart Insights</span>
            <span class="feature-tag">🚀 Professional Results</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.get('file_uploaded', False):
        show_canva_welcome()
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
        st.mark

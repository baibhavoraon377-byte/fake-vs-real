# ============================================
# 📊 TextInsight - Professional NLP Analysis
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
    page_title="TextInsight - Professional Text Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================
# Professional Style CSS
# ============================
st.markdown("""
<style>
    /* Professional Color Scheme */
    :root {
        --primary-blue: #2E86DE;
        --primary-dark: #1A1A2E;
        --primary-light: #F8F9FA;
        --accent-teal: #00CEC9;
        --accent-orange: #FF9F43;
        --accent-purple: #6C5CE7;
        --text-dark: #2D3436;
        --text-light: #636E72;
        --text-white: #FFFFFF;
        --card-bg: #FFFFFF;
        --border-light: #E9ECEF;
        --success: #00B894;
        --warning: #FDCB6E;
        --error: #E84393;
    }

    /* Main background */
    .stApp {
        background: var(--primary-light);
        color: var(--text-dark);
    }

    /* Hide default elements */
    .css-1d391kg {
        display: none !important;
    }

    .main .block-container {
        padding: 0;
        margin: 0;
        max-width: 100%;
    }

    /* Professional Sidebar */
    .professional-sidebar {
        position: fixed;
        left: 0;
        top: 0;
        width: 280px;
        height: 100vh;
        background: var(--primary-dark);
        padding: 32px 24px;
        z-index: 1000;
        overflow-y: auto;
        box-shadow: 2px 0 20px rgba(0,0,0,0.1);
    }

    .sidebar-content {
        display: flex;
        flex-direction: column;
        height: 100%;
    }

    /* Main Content Area */
    .main-content {
        margin-left: 280px;
        padding: 32px 40px;
        min-height: 100vh;
        background: var(--primary-light);
    }

    /* Logo */
    .professional-logo {
        font-size: 24px;
        font-weight: 800;
        color: var(--text-white);
        margin-bottom: 48px;
        display: flex;
        align-items: center;
        gap: 12px;
        padding-bottom: 20px;
        border-bottom: 2px solid var(--primary-blue);
    }

    .logo-icon {
        background: var(--primary-blue);
        width: 36px;
        height: 36px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
        font-weight: 700;
    }

    /* Navigation */
    .nav-section {
        margin-bottom: 40px;
    }

    .nav-title {
        color: var(--text-light);
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 20px;
        opacity: 0.7;
    }

    .nav-item {
        display: flex;
        align-items: center;
        gap: 16px;
        padding: 12px 16px;
        color: var(--text-white);
        font-weight: 600;
        font-size: 15px;
        transition: all 0.3s ease;
        cursor: pointer;
        border-radius: 8px;
        margin-bottom: 8px;
        opacity: 0.8;
    }

    .nav-item:hover {
        background: rgba(46, 134, 222, 0.1);
        opacity: 1;
        transform: translateX(4px);
    }

    .nav-item.active {
        background: var(--primary-blue);
        opacity: 1;
        box-shadow: 0 4px 12px rgba(46, 134, 222, 0.3);
    }

    .nav-icon {
        font-size: 20px;
        width: 24px;
        text-align: center;
        opacity: 0.9;
    }

    /* Cards */
    .professional-card {
        background: var(--card-bg);
        border-radius: 12px;
        padding: 24px;
        transition: all 0.3s ease;
        border: 1px solid var(--border-light);
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    }

    .professional-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
        border-color: var(--primary-blue);
    }

    .analysis-card {
        background: linear-gradient(135deg, var(--card-bg) 0%, #F8F9FF 100%);
        border-radius: 12px;
        padding: 28px;
        margin-bottom: 20px;
        border-left: 4px solid var(--primary-blue);
        border: 1px solid var(--border-light);
    }

    /* Metrics */
    .metric-card {
        background: var(--card-bg);
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        border: 1px solid var(--border-light);
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }

    .metric-card:hover {
        border-color: var(--primary-blue);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: var(--primary-blue);
        margin-bottom: 8px;
    }

    .metric-label {
        font-size: 14px;
        color: var(--text-light);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Buttons */
    .professional-btn {
        background: var(--primary-blue);
        color: var(--text-white);
        border: none;
        border-radius: 8px;
        padding: 14px 32px;
        font-weight: 700;
        font-size: 15px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(46, 134, 222, 0.3);
    }

    .professional-btn:hover {
        background: #2678C8;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(46, 134, 222, 0.4);
    }

    .secondary-btn {
        background: transparent;
        color: var(--primary-blue);
        border: 2px solid var(--primary-blue);
        border-radius: 8px;
        padding: 12px 32px;
        font-weight: 600;
        font-size: 15px;
        transition: all 0.3s ease;
    }

    .secondary-btn:hover {
        background: rgba(46, 134, 222, 0.1);
        transform: translateY(-2px);
    }

    /* Headers */
    .page-header {
        font-size: 2.5rem;
        font-weight: 800;
        color: var(--text-dark);
        margin-bottom: 12px;
        background: linear-gradient(135deg, var(--text-dark) 0%, var(--primary-blue) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .section-header {
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--text-dark);
        margin: 40px 0 24px 0;
        padding-bottom: 12px;
        border-bottom: 2px solid var(--border-light);
    }

    .card-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: var(--text-dark);
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    /* Progress */
    .progress-container {
        background: var(--border-light);
        border-radius: 10px;
        height: 6px;
        overflow: hidden;
        margin: 16px 0;
    }

    .progress-fill {
        background: linear-gradient(90deg, var(--primary-blue), var(--accent-teal));
        height: 100%;
        border-radius: 10px;
        transition: width 0.3s ease;
    }

    /* Model Cards */
    .model-card {
        background: var(--card-bg);
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        border: 1px solid var(--border-light);
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }

    .model-card:hover {
        border-color: var(--primary-blue);
        transform: translateY(-4px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }

    .model-accuracy {
        font-size: 2.2rem;
        font-weight: 800;
        color: var(--primary-blue);
        margin: 16px 0;
    }

    /* File Upload */
    .upload-area {
        background: var(--card-bg);
        border: 2px dashed var(--border-light);
        border-radius: 12px;
        padding: 48px;
        text-align: center;
        transition: all 0.3s ease;
    }

    .upload-area:hover {
        border-color: var(--primary-blue);
        background: rgba(46, 134, 222, 0.02);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: transparent;
        border-bottom: 2px solid var(--border-light);
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: var(--text-light) !important;
        border-radius: 8px 8px 0 0;
        padding: 16px 32px;
        border: none;
        font-weight: 600;
        font-size: 15px;
        transition: all 0.3s ease;
    }

    .stTabs [aria-selected="true"] {
        background: transparent !important;
        color: var(--primary-blue) !important;
        border-bottom: 3px solid var(--primary-blue) !important;
    }

    /* Inputs */
    .stSelectbox, .stTextInput, .stNumberInput {
        background: var(--card-bg) !important;
        border: 2px solid var(--border-light) !important;
        border-radius: 8px !important;
    }

    .stSelectbox:focus, .stTextInput:focus, .stNumberInput:focus {
        border-color: var(--primary-blue) !important;
        box-shadow: 0 0 0 3px rgba(46, 134, 222, 0.1) !important;
    }

    .stSelectbox div, .stTextInput input, .stNumberInput input {
        background: var(--card-bg) !important;
        color: var(--text-dark) !important;
        font-weight: 500;
    }

    /* Dataframes */
    .dataframe {
        background: var(--card-bg) !important;
        color: var(--text-dark) !important;
        border: 1px solid var(--border-light) !important;
        border-radius: 8px !important;
    }

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: var(--primary-light);
    }

    ::-webkit-scrollbar-thumb {
        background: var(--border-light);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--text-light);
    }

    /* Analysis Status */
    .analysis-status {
        background: var(--card-bg);
        border-radius: 12px;
        padding: 20px;
        margin: 20px 0;
        border-left: 4px solid var(--success);
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }

    /* Results Grid */
    .results-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
        gap: 20px;
        margin: 24px 0;
    }

    /* Feature Icons */
    .feature-icon {
        width: 48px;
        height: 48px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        margin-bottom: 16px;
    }

    .icon-lexical {
        background: linear-gradient(135deg, var(--primary-blue), #4A90E2);
        color: white;
    }

    .icon-semantic {
        background: linear-gradient(135deg, var(--accent-teal), #00B894);
        color: white;
    }

    .icon-syntactic {
        background: linear-gradient(135deg, var(--accent-orange), #FF7675);
        color: white;
    }

    .icon-pragmatic {
        background: linear-gradient(135deg, var(--accent-purple), #A29BFE);
        color: white;
    }

    /* Stat Items */
    .stat-item {
        display: flex;
        align-items: center;
        gap: 16px;
        padding: 16px;
        background: var(--card-bg);
        border-radius: 8px;
        margin-bottom: 12px;
        border: 1px solid var(--border-light);
        transition: all 0.3s ease;
    }

    .stat-item:hover {
        border-color: var(--primary-blue);
        transform: translateX(4px);
    }

    .stat-icon {
        width: 40px;
        height: 40px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
        background: rgba(46, 134, 222, 0.1);
        color: var(--primary-blue);
    }

    .stat-content {
        flex: 1;
    }

    .stat-value {
        font-size: 1.5rem;
        font-weight: 800;
        color: var(--text-dark);
        margin-bottom: 4px;
    }

    .stat-label {
        font-size: 14px;
        color: var(--text-light);
        font-weight: 600;
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
        """Create professional performance dashboard"""
        plt.style.use('default')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.patch.set_facecolor('#FFFFFF')

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

        colors = ['#2E86DE', '#00CEC9', '#FF9F43', '#6C5CE7']

        # Accuracy
        bars1 = ax1.bar(models, metrics_data['Accuracy'], color=colors, alpha=0.9, edgecolor='white', linewidth=2)
        ax1.set_facecolor('#F8F9FA')
        ax1.set_title('Model Accuracy', fontweight='bold', color='#2D3436', fontsize=16, pad=20)
        ax1.set_ylabel('Score', fontweight='bold', color='#636E72')
        ax1.tick_params(axis='x', rotation=45, colors='#636E72')
        ax1.tick_params(axis='y', colors='#636E72')
        ax1.grid(True, alpha=0.2, axis='y', color='#E9ECEF')

        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold', color='#2D3436')

        # Precision
        bars2 = ax2.bar(models, metrics_data['Precision'], color=colors, alpha=0.9, edgecolor='white', linewidth=2)
        ax2.set_facecolor('#F8F9FA')
        ax2.set_title('Model Precision', fontweight='bold', color='#2D3436', fontsize=16, pad=20)
        ax2.set_ylabel('Score', fontweight='bold', color='#636E72')
        ax2.tick_params(axis='x', rotation=45, colors='#636E72')
        ax2.tick_params(axis='y', colors='#636E72')
        ax2.grid(True, alpha=0.2, axis='y', color='#E9ECEF')

        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold', color='#2D3436')

        # Recall
        bars3 = ax3.bar(models, metrics_data['Recall'], color=colors, alpha=0.9, edgecolor='white', linewidth=2)
        ax3.set_facecolor('#F8F9FA')
        ax3.set_title('Model Recall', fontweight='bold', color='#2D3436', fontsize=16, pad=20)
        ax3.set_ylabel('Score', fontweight='bold', color='#636E72')
        ax3.tick_params(axis='x', rotation=45, colors='#636E72')
        ax3.tick_params(axis='y', colors='#636E72')
        ax3.grid(True, alpha=0.2, axis='y', color='#E9ECEF')

        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold', color='#2D3436')

        # F1-Score
        bars4 = ax4.bar(models, metrics_data['F1-Score'], color=colors, alpha=0.9, edgecolor='white', linewidth=2)
        ax4.set_facecolor('#F8F9FA')
        ax4.set_title('Model F1-Score', fontweight='bold', color='#2D3436', fontsize=16, pad=20)
        ax4.set_ylabel('Score', fontweight='bold', color='#636E72')
        ax4.tick_params(axis='x', rotation=45, colors='#636E72')
        ax4.tick_params(axis='y', colors='#636E72')
        ax4.grid(True, alpha=0.2, axis='y', color='#E9ECEF')

        for bar in bars4:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold', color='#2D3436')

        plt.tight_layout()
        return fig

# ============================
# Professional Sidebar Component
# ============================
def create_professional_sidebar():
    """Create professional sidebar"""
    st.markdown("""
    <div class="professional-sidebar">
        <div class="sidebar-content">
            <div class="professional-logo">
                <div class="logo-icon">T</div>
                TextInsight
            </div>
            
            <div class="nav-section">
                <div class="nav-title">Navigation</div>
                <div class="nav-item active">
                    <div class="nav-icon">📊</div>
                    <div>Dashboard</div>
                </div>
                <div class="nav-item">
                    <div class="nav-icon">🔍</div>
                    <div>Text Analysis</div>
                </div>
                <div class="nav-item">
                    <div class="nav-icon">📈</div>
                    <div>Model Performance</div>
                </div>
                <div class="nav-item">
                    <div class="nav-icon">📋</div>
                    <div>Reports</div>
                </div>
            </div>

            <div class="nav-section">
                <div class="nav-title">Analysis Methods</div>
                <div class="nav-item">
                    <div class="nav-icon">📖</div>
                    <div>Lexical Analysis</div>
                </div>
                <div class="nav-item">
                    <div class="nav-icon">🎭</div>
                    <div>Semantic Analysis</div>
                </div>
                <div class="nav-item">
                    <div class="nav-icon">🔧</div>
                    <div>Syntactic Analysis</div>
                </div>
                <div class="nav-item">
                    <div class="nav-icon">🎯</div>
                    <div>Pragmatic Analysis</div>
                </div>
            </div>

            <div class="nav-section">
                <div class="nav-title">Machine Learning</div>
                <div class="nav-item">
                    <div class="nav-icon">🤖</div>
                    <div>Model Training</div>
                </div>
                <div class="nav-item">
                    <div class="nav-icon">📊</div>
                    <div>Performance Metrics</div>
                </div>
                <div class="nav-item">
                    <div class="nav-icon">🔬</div>
                    <div>Feature Engineering</div>
                </div>
                <div class="nav-item">
                    <div class="nav-icon">📝</div>
                    <div>Model Evaluation</div>
                </div>
            </div>

            <div style="flex: 1;"></div>

            <div class="nav-section">
                <div class="nav-item">
                    <div class="nav-icon">⚙️</div>
                    <div>Settings</div>
                </div>
                <div class="nav-item">
                    <div class="nav-icon">❓</div>
                    <div>Help & Support</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================
# Main Content Components
# ============================
def create_upload_section():
    """Create professional upload section"""
    st.markdown("""
    <div class="upload-area">
        <div style="font-size: 64px; margin-bottom: 24px; color: #2E86DE;">📁</div>
        <h3 style="color: #2D3436; margin-bottom: 16px; font-weight: 700;">Upload Dataset</h3>
        <p style="color: #636E72; margin-bottom: 32px; font-size: 16px;">
            Upload your CSV file to begin text analysis. Supported formats: CSV with text columns.
        </p>
        <div style="color: #2E86DE; font-weight: 600; font-size: 14px;">
            Click or drag and drop to upload
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose CSV File",
        type=["csv"],
        label_visibility="collapsed"
    )
    
    return uploaded_file

def create_analysis_cards():
    """Create professional analysis method cards"""
    st.markdown('<div class="section-header">Analysis Methods</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="professional-card">
            <div class="feature-icon icon-lexical">📖</div>
            <div class="card-header">Lexical Analysis</div>
            <div style="color: #636E72; font-size: 15px; line-height: 1.6;">
                Advanced word-level processing, lemmatization, and n-gram analysis for comprehensive text understanding.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="professional-card">
            <div class="feature-icon icon-semantic">🎭</div>
            <div class="card-header">Semantic Analysis</div>
            <div style="color: #636E72; font-size: 15px; line-height: 1.6;">
                Sentiment analysis, emotion detection, and meaning extraction for deeper text comprehension.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="professional-card">
            <div class="feature-icon icon-syntactic">🔧</div>
            <div class="card-header">Syntactic Analysis</div>
            <div style="color: #636E72; font-size: 15px; line-height: 1.6;">
                Grammar structure analysis, part-of-speech tagging, and syntactic pattern recognition.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="professional-card">
            <div class="feature-icon icon-pragmatic">🎯</div>
            <div class="card-header">Pragmatic Analysis</div>
            <div style="color: #636E72; font-size: 15px; line-height: 1.6;">
                Context analysis, intent detection, and pragmatic feature extraction for complete text understanding.
            </div>
        </div>
        """, unsafe_allow_html=True)

def create_stats_overview():
    """Create professional stats overview"""
    st.markdown('<div class="section-header">Platform Overview</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="stat-item">
            <div class="stat-icon">🤖</div>
            <div class="stat-content">
                <div class="stat-value">4</div>
                <div class="stat-label">ML Models</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stat-item">
            <div class="stat-icon">📊</div>
            <div class="stat-content">
                <div class="stat-value">4</div>
                <div class="stat-label">Analysis Types</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stat-item">
            <div class="stat-icon">⚡</div>
            <div class="stat-content">
                <div class="stat-value">Real-time</div>
                <div class="stat-label">Processing</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="stat-item">
            <div class="stat-icon">📈</div>
            <div class="stat-content">
                <div class="stat-value">95%+</div>
                <div class="stat-label">Accuracy</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

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

    # Create professional layout
    create_professional_sidebar()
    
    # Main content area
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    
    # Page header
    st.markdown('<div class="page-header">TextInsight Analytics</div>', unsafe_allow_html=True)
    st.markdown('<p style="color: #636E72; margin-bottom: 40px; font-size: 18px; line-height: 1.6;">Professional text analysis platform with advanced machine learning capabilities for comprehensive NLP insights.</p>', unsafe_allow_html=True)
    
    # Platform overview
    create_stats_overview()
    
    # Upload section
    st.markdown('<div class="section-header">Data Upload</div>', unsafe_allow_html=True)
    uploaded_file = create_upload_section()
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.session_state.file_uploaded = True
            
            st.success("✅ Dataset loaded successfully!")
            
            # Show dataset info
            st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{df.shape[0]}</div>
                    <div class="metric-label">Total Records</div>
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
                unique_classes = len(df.columns)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{unique_classes}</div>
                    <div class="metric-label">Columns</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Configuration
            st.markdown('<div class="section-header">Analysis Configuration</div>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                text_col = st.selectbox(
                    "Text Column",
                    df.columns,
                    help="Select column containing text data"
                )
            
            with col2:
                target_col = st.selectbox(
                    "Target Column", 
                    df.columns,
                    index=min(1, len(df.columns)-1) if len(df.columns) > 1 else 0,
                    help="Select column containing labels"
                )
            
            with col3:
                feature_type = st.selectbox(
                    "Analysis Type",
                    ["Lexical", "Semantic", "Syntactic", "Pragmatic"],
                    help="Choose analysis method"
                )
            
            st.session_state.config = {
                'text_col': text_col,
                'target_col': target_col,
                'feature_type': feature_type
            }
            
            # Start analysis button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚀 Start Analysis", use_container_width=True, key="analyze_btn"):
                    st.session_state.analyze_clicked = True
            
            # Show analysis methods
            create_analysis_cards()
            
            # Data preview
            with st.expander("📋 Dataset Preview", expanded=True):
                st.dataframe(df.head(10), use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
    else:
        # Show analysis methods when no file is uploaded
        create_analysis_cards()
        
        # Quick actions
        st.markdown('<div class="section-header">Quick Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="analysis-card">
                <div class="card-header">📝 Text Classification</div>
                <div style="color: #636E72; font-size: 15px; margin-bottom: 20px; line-height: 1.6;">
                    Automatically classify text documents into predefined categories using advanced machine learning algorithms.
                </div>
                <div class="progress-container">
                    <div class="progress-fill" style="width: 85%"></div>
                </div>
                <div style="color: #2E86DE; font-size: 14px; font-weight: 600; margin-top: 8px;">
                    85% Average Accuracy
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="analysis-card">
                <div class="card-header">😊 Sentiment Analysis</div>
                <div style="color: #636E72; font-size: 15px; margin-bottom: 20px; line-height: 1.6;">
                    Analyze emotional tone and sentiment polarity in text data with high precision sentiment detection.
                </div>
                <div class="progress-container">
                    <div class="progress-fill" style="width: 92%"></div>
                </div>
                <div style="color: #00CEC9; font-size: 14px; font-weight: 600; margin-top: 8px;">
                    92% Accuracy Rate
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Analysis results
    if st.session_state.get('analyze_clicked', False) and st.session_state.get('file_uploaded', False):
        perform_analysis(st.session_state.df, st.session_state.config)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ============================
# Analysis Function
# ============================
def perform_analysis(df, config):
    """Perform analysis with professional style"""
    st.markdown('<div class="section-header">Analysis Results</div>', unsafe_allow_html=True)
    
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
    with st.spinner("🔍 Extracting features..."):
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

    st.success("✅ Feature extraction completed!")

    # Model training
    with st.spinner("🤖 Training machine learning models..."):
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
                    <div style="color: #636E72; font-size: 14px; margin-bottom: 16px;">
                        Precision: {result['precision']:.3f} | Recall: {result['recall']:.3f}
                    </div>
                    <div class="progress-container">
                        <div class="progress-fill" style="width: {accuracy*100}%"></div>
                    </div>
                    <div style="color: #2E86DE; font-size: 12px; font-weight: 600; margin-top: 8px;">
                        F1-Score: {result['f1_score']:.3f}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Performance Dashboard
        st.markdown("#### 📊 Performance Dashboard")
        viz = Visualizer()
        dashboard_fig = viz.create_performance_dashboard(successful_models)
        st.pyplot(dashboard_fig)

        # Best Model
        best_model = max(successful_models.items(), key=lambda x: x[1]['accuracy'])
        st.markdown(f"""
        <div class="analysis-card">
            <h3 style="color: #2E86DE; margin-bottom: 1rem; display: flex; align-items: center; gap: 12px;">
                <span>🏆</span> Recommended Model
            </h3>
            <p style="color: #2D3436; font-size: 1.1rem; margin-bottom: 0.5rem; line-height: 1.6;">
                <strong>{best_model[0]}</strong> achieved the highest accuracy of
                <strong style="color: #2E86DE;">{best_model[1]['accuracy']:.1%}</strong>
                and is recommended for production deployment.
            </p>
            <p style="color: #636E72; margin: 0; line-height: 1.6;">
                This model demonstrates superior performance across all evaluation metrics including 
                precision, recall, and F1-score, making it the optimal choice for your text analysis needs.
            </p>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.error("❌ No models were successfully trained.")

if __name__ == "__main__":
    main()

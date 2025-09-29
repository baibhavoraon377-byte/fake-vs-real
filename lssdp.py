# ============================================
# 🚀 NLP Analyzer Pro
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from textblob import TextBlob
import time

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# ============================
# Page Configuration
# ============================
st.set_page_config(
    page_title="NLP Analyzer Pro",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================
# Perplexity Style CSS
# ============================
st.markdown("""
<style>
    /* Perplexity Color Scheme */
    :root {
        --ppx-blue: #2D7FF9;
        --ppx-dark: #0A0F1C;
        --ppx-darker: #050A14;
        --ppx-card: #131720;
        --ppx-border: #2A2F3C;
        --ppx-text: #FFFFFF;
        --ppx-text-light: #A0A4B8;
        --ppx-accent: #00C2FF;
        --ppx-success: #00D4AA;
        --ppx-warning: #FFB800;
        --ppx-error: #FF4757;
    }
    
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, var(--ppx-darker) 0%, var(--ppx-dark) 100%);
        color: var(--ppx-text);
    }
    
    /* Headers */
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, var(--ppx-blue) 0%, var(--ppx-accent) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        letter-spacing: -0.02em;
    }
    
    .section-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--ppx-text);
        margin: 3rem 0 1.5rem 0;
        padding-bottom: 0.8rem;
        border-bottom: 2px solid var(--ppx-border);
        position: relative;
    }
    
    .section-header::after {
        content: '';
        position: absolute;
        bottom: -2px;
        left: 0;
        width: 80px;
        height: 2px;
        background: linear-gradient(90deg, var(--ppx-blue), var(--ppx-accent));
    }
    
    /* Cards */
    .ppx-card {
        background: var(--ppx-card);
        border: 1px solid var(--ppx-border);
        border-radius: 16px;
        padding: 2rem;
        margin: 1.5rem 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .ppx-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--ppx-blue), transparent);
    }
    
    .ppx-card:hover {
        transform: translateY(-4px);
        border-color: var(--ppx-blue);
        box-shadow: 0 12px 40px rgba(45, 127, 249, 0.15);
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, var(--ppx-card) 0%, #1A1F2E 100%);
        border: 1px solid var(--ppx-border);
        border-radius: 12px;
        padding: 1.8rem;
        text-align: center;
        margin: 0.8rem;
        transition: all 0.3s ease;
        position: relative;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, var(--ppx-blue), var(--ppx-accent));
        border-radius: 12px 12px 0 0;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: var(--ppx-blue);
        box-shadow: 0 8px 25px rgba(45, 127, 249, 0.2);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, var(--ppx-text) 0%, var(--ppx-text-light) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: var(--ppx-text-light);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Buttons */
    .stButton button {
        width: 100%;
        background: linear-gradient(135deg, var(--ppx-blue) 0%, var(--ppx-accent) 100%);
        color: white;
        border: none;
        padding: 1rem 2rem;
        border-radius: 12px;
        font-weight: 700;
        font-size: 1rem;
        transition: all 0.3s ease;
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
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    .stButton button:hover::before {
        left: 100%;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(45, 127, 249, 0.4);
    }
    
    /* File Uploader */
    .stFileUploader {
        background: var(--ppx-card) !important;
        border: 2px dashed var(--ppx-border) !important;
        border-radius: 12px !important;
        padding: 2rem !important;
    }
    
    .stFileUploader:hover {
        border-color: var(--ppx-blue) !important;
    }
    
    /* Select boxes and inputs */
    .stSelectbox, .stTextInput, .stNumberInput {
        background: var(--ppx-card) !important;
        color: var(--ppx-text) !important;
        border: 1px solid var(--ppx-border) !important;
        border-radius: 8px !important;
    }
    
    .stSelectbox div, .stTextInput input, .stNumberInput input {
        background: var(--ppx-card) !important;
        color: var(--ppx-text) !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: var(--ppx-card);
        border-bottom: 1px solid var(--ppx-border);
        border-radius: 12px 12px 0 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: var(--ppx-card) !important;
        color: var(--ppx-text-light) !important;
        border-radius: 0;
        padding: 1.2rem 2rem;
        border-bottom: 2px solid transparent;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--ppx-card) !important;
        color: var(--ppx-blue) !important;
        border-bottom: 2px solid var(--ppx-blue) !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: var(--ppx-card) !important;
        color: var(--ppx-text) !important;
        border: 1px solid var(--ppx-border) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--ppx-blue), var(--ppx-accent));
    }
    
    /* Success, Error, Info */
    .stSuccess {
        background: rgba(0, 212, 170, 0.1) !important;
        border: 1px solid var(--ppx-success) !important;
        color: var(--ppx-success) !important;
        border-radius: 8px !important;
    }
    
    .stError {
        background: rgba(255, 71, 87, 0.1) !important;
        border: 1px solid var(--ppx-error) !important;
        color: var(--ppx-error) !important;
        border-radius: 8px !important;
    }
    
    .stInfo {
        background: rgba(0, 194, 255, 0.1) !important;
        border: 1px solid var(--ppx-accent) !important;
        color: var(--ppx-accent) !important;
        border-radius: 8px !important;
    }
    
    /* Dataframe styling */
    .dataframe {
        background: var(--ppx-card) !important;
        color: var(--ppx-text) !important;
        border: 1px solid var(--ppx-border) !important;
    }
    
    /* Hero Section */
    .hero-section {
        background: linear-gradient(135deg, var(--ppx-card) 0%, #1A1F2E 100%);
        border: 1px solid var(--ppx-border);
        border-radius: 20px;
        padding: 4rem 3rem;
        margin: 3rem 0;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--ppx-blue), transparent);
    }
    
    /* Model Performance Cards */
    .model-card {
        background: var(--ppx-card);
        border: 1px solid var(--ppx-border);
        border-radius: 12px;
        padding: 2rem;
        margin: 1rem;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
    }
    
    .model-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, var(--ppx-blue), var(--ppx-accent));
        border-radius: 12px 12px 0 0;
    }
    
    .model-card:hover {
        transform: translateY(-5px);
        border-color: var(--ppx-blue);
        box-shadow: 0 10px 30px rgba(45, 127, 249, 0.2);
    }
    
    .model-accuracy {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, var(--ppx-blue) 0%, var(--ppx-accent) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 1.5rem 0;
    }
    
    /* Feature Tags */
    .feature-tag {
        background: rgba(45, 127, 249, 0.15);
        color: var(--ppx-blue);
        padding: 0.6rem 1.2rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.3rem;
        display: inline-block;
        border: 1px solid rgba(45, 127, 249, 0.3);
        transition: all 0.3s ease;
    }
    
    .feature-tag:hover {
        background: rgba(45, 127, 249, 0.25);
        transform: translateY(-1px);
    }
    
    /* Live Analysis Cards */
    .live-analysis-card {
        background: linear-gradient(135deg, var(--ppx-card) 0%, #1A1F2E 100%);
        border: 1px solid var(--ppx-border);
        border-radius: 12px;
        padding: 1.8rem;
        margin: 1rem 0;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .live-analysis-card:hover {
        border-color: var(--ppx-blue);
        transform: translateY(-2px);
    }
    
    /* Upload Area */
    .upload-area {
        background: var(--ppx-card);
        border: 2px dashed var(--ppx-border);
        border-radius: 16px;
        padding: 3rem;
        text-align: center;
        margin: 2rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: var(--ppx-blue);
        background: rgba(45, 127, 249, 0.05);
    }
    
    /* Analysis Controls */
    .analysis-controls {
        background: var(--ppx-card);
        border: 1px solid var(--ppx-border);
        border-radius: 16px;
        padding: 2rem;
        margin: 2rem 0;
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
class PpxFeatureExtractor:
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
# Ppx Style Model Trainer
# ============================
class PpxModelTrainer:
    def __init__(self):
        self.models = {
            "🚀 Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
            "🌲 Random Forest": RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced'),
            "⚡ Support Vector": SVC(random_state=42, probability=True, class_weight='balanced'),
            "📊 Naive Bayes": MultinomialNB()
        }
    
    def train_and_evaluate(self, X, y):
        """Ppx style model training with comprehensive evaluation"""
        results = {}
        
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        n_classes = len(le.classes_)
        
        test_size = max(0.15, min(0.25, 3 * n_classes / len(y_encoded)))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        # Ppx style progress
        progress_container = st.empty()
        
        for i, (name, model) in enumerate(self.models.items()):
            with progress_container.container():
                cols = st.columns([3, 1])
                with cols[0]:
                    st.markdown(f"**Training {name}**")
                with cols[1]:
                    progress_bar = st.progress(0)
                    
                    # Simulate Ppx-style loading
                    for step in range(5):
                        progress_bar.progress((step + 1) / 5)
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
# Ppx Style Visualizations
# ============================
class PpxVisualizer:
    @staticmethod
    def create_performance_dashboard(results):
        """Create Ppx-style performance dashboard"""
        # Set dark theme for matplotlib
        plt.style.use('dark_background')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.patch.set_facecolor('#0A0F1C')
        
        models = []
        metrics_data = {
            'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-Score': []
        }
        
        for model_name, result in results.items():
            if 'error' not in result:
                clean_name = model_name.replace('🚀 ', '').replace('🌲 ', '').replace('⚡ ', '').replace('📊 ', '')
                models.append(clean_name)
                metrics_data['Accuracy'].append(result['accuracy'])
                metrics_data['Precision'].append(result['precision'])
                metrics_data['Recall'].append(result['recall'])
                metrics_data['F1-Score'].append(result['f1_score'])
        
        colors = ['#2D7FF9', '#00C2FF', '#00D4AA', '#FFB800']
        
        # Accuracy
        bars1 = ax1.bar(models, metrics_data['Accuracy'], color=colors, alpha=0.9, edgecolor='white', linewidth=2)
        ax1.set_facecolor('#131720')
        ax1.set_title('🎯 Accuracy', fontweight='bold', color='white', fontsize=14, pad=20)
        ax1.set_ylabel('Score', fontweight='bold', color='white')
        ax1.tick_params(axis='x', rotation=45, colors='white')
        ax1.tick_params(axis='y', colors='white')
        ax1.grid(True, alpha=0.2, axis='y', color='#2A2F3C')
        
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold', color='white')
        
        # Precision
        bars2 = ax2.bar(models, metrics_data['Precision'], color=colors, alpha=0.9, edgecolor='white', linewidth=2)
        ax2.set_facecolor('#131720')
        ax2.set_title('📊 Precision', fontweight='bold', color='white', fontsize=14, pad=20)
        ax2.set_ylabel('Score', fontweight='bold', color='white')
        ax2.tick_params(axis='x', rotation=45, colors='white')
        ax2.tick_params(axis='y', colors='white')
        ax2.grid(True, alpha=0.2, axis='y', color='#2A2F3C')
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold', color='white')
        
        # Recall
        bars3 = ax3.bar(models, metrics_data['Recall'], color=colors, alpha=0.9, edgecolor='white', linewidth=2)
        ax3.set_facecolor('#131720')
        ax3.set_title('🔍 Recall', fontweight='bold', color='white', fontsize=14, pad=20)
        ax3.set_ylabel('Score', fontweight='bold', color='white')
        ax3.tick_params(axis='x', rotation=45, colors='white')
        ax3.tick_params(axis='y', colors='white')
        ax3.grid(True, alpha=0.2, axis='y', color='#2A2F3C')
        
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold', color='white')
        
        # F1-Score
        bars4 = ax4.bar(models, metrics_data['F1-Score'], color=colors, alpha=0.9, edgecolor='white', linewidth=2)
        ax4.set_facecolor('#131720')
        ax4.set_title('⚡ F1-Score', fontweight='bold', color='white', fontsize=14, pad=20)
        ax4.set_ylabel('Score', fontweight='bold', color='white')
        ax4.tick_params(axis='x', rotation=45, colors='white')
        ax4.tick_params(axis='y', colors='white')
        ax4.grid(True, alpha=0.2, axis='y', color='#2A2F3C')
        
        for bar in bars4:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold', color='white')
        
        plt.tight_layout()
        return fig

# ============================
# Enhanced Cool Features
# ============================
class CoolFeatures:
    @staticmethod
    def text_analysis_preview():
        """Real-time text analysis preview"""
        st.markdown("<div class='section-header'>🔍 Live Text Analysis</div>", unsafe_allow_html=True)
        
        sample_text = st.text_area(
            "Try it yourself - Enter text to analyze:",
            "This is an amazing product! I absolutely love using it every day. The features are incredible and the user experience is outstanding.",
            height=120,
            key="live_analysis"
        )
        
        if sample_text:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                # Sentiment analysis
                blob = TextBlob(sample_text)
                sentiment_score = blob.sentiment.polarity
                if sentiment_score > 0.1:
                    sentiment = "😊 Positive"
                    color = "#00D4AA"
                elif sentiment_score < -0.1:
                    sentiment = "😞 Negative" 
                    color = "#FF4757"
                else:
                    sentiment = "😐 Neutral"
                    color = "#FFB800"
                
                st.markdown(f"""
                <div class="live-analysis-card">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">😊</div>
                    <div style="font-size: 1.2rem; font-weight: bold; color: {color};">{sentiment}</div>
                    <div style="font-size: 0.9rem; color: #A0A4B8;">Sentiment Score: {sentiment_score:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Text statistics
                word_count = len(sample_text.split())
                char_count = len(sample_text)
                st.markdown(f"""
                <div class="live-analysis-card">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">📊</div>
                    <div style="font-size: 1.2rem; font-weight: bold; color: #2D7FF9;">{word_count}</div>
                    <div style="font-size: 0.9rem; color: #A0A4B8;">Words / {char_count} chars</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                # Readability score
                sentences = len([s for s in sample_text.split('.') if s.strip()])
                avg_sentence_length = word_count / max(sentences, 1)
                st.markdown(f"""
                <div class="live-analysis-card">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">📖</div>
                    <div style="font-size: 1.2rem; font-weight: bold; color: #00C2FF;">{avg_sentence_length:.1f}</div>
                    <div style="font-size: 0.9rem; color: #A0A4B8;">Avg Words/Sentence</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                # Complexity score
                complex_words = len([word for word in sample_text.split() if len(word) > 6])
                complexity_ratio = complex_words / max(word_count, 1)
                st.markdown(f"""
                <div class="live-analysis-card">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">⚡</div>
                    <div style="font-size: 1.2rem; font-weight: bold; color: #FFB800;">{complexity_ratio:.1%}</div>
                    <div style="font-size: 0.9rem; color: #A0A4B8;">Complexity Ratio</div>
                </div>
                """, unsafe_allow_html=True)

    @staticmethod
    def generate_wordcloud(df, text_col):
        """Generate word cloud visualization"""
        st.markdown("<div class='section-header'>☁️ Word Cloud</div>", unsafe_allow_html=True)
        
        try:
            # Combine all text
            all_text = ' '.join(df[text_col].astype(str))
            
            if len(all_text.strip()) > 0:
                # Create word cloud
                wordcloud = WordCloud(
                    width=800, 
                    height=400, 
                    background_color='#0A0F1C',
                    colormap='Blues',
                    max_words=100,
                    relative_scaling=0.5
                ).generate(all_text)
                
                # Display
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                ax.set_title('Most Frequent Words in Your Dataset', color='white', fontsize=16, pad=20)
                st.pyplot(fig)
            else:
                st.info("No text data available for word cloud generation.")
        except Exception as e:
            st.error(f"Could not generate word cloud: {str(e)}")

    @staticmethod
    def interactive_model_comparison(results):
        """Interactive model comparison with selectable metrics"""
        st.markdown("<div class='section-header'>📈 Interactive Model Comparison</div>", unsafe_allow_html=True)
        
        # Create interactive metrics selector
        metrics = st.multiselect(
            "Select metrics to compare:",
            ["Accuracy", "Precision", "Recall", "F1-Score"],
            default=["Accuracy", "F1-Score"],
            key="model_metrics"
        )
        
        if metrics:
            # Prepare data for plotting
            models = []
            metric_data = {metric: [] for metric in metrics}
            
            for model_name, result in results.items():
                if 'error' not in result:
                    models.append(model_name)
                    for metric in metrics:
                        metric_data[metric].append(result[metric.lower()])
            
            # Create interactive plot
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(12, 6))
            fig.patch.set_facecolor('#0A0F1C')
            ax.set_facecolor('#131720')
            
            x = np.arange(len(models))
            width = 0.8 / len(metrics)
            
            colors = ['#2D7FF9', '#00C2FF', '#00D4AA', '#FFB800']
            
            for i, metric in enumerate(metrics):
                offset = width * i
                bars = ax.bar(x + offset, metric_data[metric], width, 
                            label=metric, alpha=0.8, color=colors[i % len(colors)],
                            edgecolor='white', linewidth=1)
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.3f}', ha='center', va='bottom', 
                           fontweight='bold', color='white', fontsize=10)
            
            ax.set_xlabel('Models', color='white', fontweight='bold')
            ax.set_ylabel('Scores', color='white', fontweight='bold')
            ax.set_title('Interactive Model Comparison', color='white', fontsize=16, pad=20)
            ax.set_xticks(x + width * (len(metrics)-1)/2)
            ax.set_xticklabels([name.replace('🚀 ', '').replace('🌲 ', '').replace('⚡ ', '').replace('📊 ', '') 
                              for name in models], rotation=45, color='white')
            ax.tick_params(colors='white')
            ax.legend(facecolor='#131720', edgecolor='none', labelcolor='white')
            ax.grid(True, alpha=0.2, color='#2A2F3C')
            
            st.pyplot(fig)

    @staticmethod
    def text_similarity_analysis(df, text_col):
        """Text similarity analysis with heatmap"""
        st.markdown("<div class='section-header'>🔗 Text Similarity Analysis</div>", unsafe_allow_html=True)
        
        try:
            # Sample some texts for comparison
            sample_size = min(6, len(df))
            sample_texts = df[text_col].sample(sample_size).tolist()
            
            if len(sample_texts) >= 2:
                # Create TF-IDF vectors
                vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(sample_texts)
                
                # Calculate similarity matrix
                similarity_matrix = cosine_similarity(tfidf_matrix)
                
                # Display similarity heatmap
                plt.style.use('dark_background')
                fig, ax = plt.subplots(figsize=(10, 8))
                fig.patch.set_facecolor('#0A0F1C')
                
                sns.heatmap(similarity_matrix, 
                           annot=True, 
                           cmap='Blues', 
                           ax=ax,
                           xticklabels=[f"Text {i+1}" for i in range(len(sample_texts))],
                           yticklabels=[f"Text {i+1}" for i in range(len(sample_texts))],
                           cbar_kws={'label': 'Similarity Score'})
                
                ax.set_title('Text Similarity Matrix', color='white', fontsize=16, pad=20)
                st.pyplot(fig)
                
                # Show most similar pair
                max_similarity = 0
                most_similar_pair = (0, 0)
                
                for i in range(len(similarity_matrix)):
                    for j in range(i+1, len(similarity_matrix)):
                        if similarity_matrix[i][j] > max_similarity:
                            max_similarity = similarity_matrix[i][j]
                            most_similar_pair = (i, j)
                
                if max_similarity > 0:
                    st.info(f"**🎯 Most Similar Texts** (Similarity Score: `{max_similarity:.3f}`)")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.text_area("Text 1", sample_texts[most_similar_pair[0]], height=100, key="text1")
                    with col2:
                        st.text_area("Text 2", sample_texts[most_similar_pair[1]], height=100, key="text2")
            else:
                st.info("Need at least 2 text samples for similarity analysis.")
        except Exception as e:
            st.error(f"Could not perform similarity analysis: {str(e)}")

    @staticmethod
    def cool_training_animation():
        """Show cool training animations"""
        st.markdown("<div class='section-header'>🚀 Training Progress</div>", unsafe_allow_html=True)
        
        placeholder = st.empty()
        
        training_steps = [
            "🔄 Preprocessing text data...",
            "🔤 Extracting features...",
            "🤖 Training Logistic Regression...",
            "🌲 Growing Random Forest...",
            "⚡ Optimizing Support Vector Machine...",
            "📊 Fitting Naive Bayes...",
            "🎯 Evaluating models...",
            "✨ Finalizing results..."
        ]
        
        for i, step in enumerate(training_steps):
            with placeholder.container():
                # Progress bar with custom styling
                progress = (i + 1) / len(training_steps)
                st.markdown(f"""
                <div class='ppx-card'>
                    <div style='color: #2D7FF9; font-weight: bold; font-size: 1.1rem; margin-bottom: 1rem;'>{step}</div>
                    <div style='background: #131720; border-radius: 8px; height: 20px; position: relative; border: 1px solid #2A2F3C;'>
                        <div style='background: linear-gradient(90deg, #2D7FF9, #00C2FF); width: {progress*100}%; height: 100%; border-radius: 8px; transition: width 0.5s;'></div>
                        <div style='position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: white; font-weight: bold; font-size: 0.8rem;'>
                            {int(progress*100)}%
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            time.sleep(0.8)  # Simulate processing time
        
        placeholder.empty()

    @staticmethod
    def generate_analysis_report(results, df, config):
        """Generate downloadable analysis report"""
        st.markdown("<div class='section-header'>📄 Analysis Report</div>", unsafe_allow_html=True)
        
        # Create comprehensive report content
        report_content = f"""# 📊 NLP Analysis Report
## Dataset Overview
- **Total Records**: {len(df):,}
- **Features**: {len(df.columns)}
- **Text Column**: {config.get('text_col', 'N/A')}
- **Target Column**: {config.get('target_col', 'N/A')}
- **Analysis Type**: {config.get('feature_type', 'N/A')}
- **Timestamp**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Model Performance Summary
"""
        
        best_accuracy = 0
        best_model = ""
        
        for model_name, result in results.items():
            if 'error' not in result:
                report_content += f"""
### {model_name}
- **Accuracy**: {result['accuracy']:.3f} ({result['accuracy']:.1%})
- **Precision**: {result['precision']:.3f}
- **Recall**: {result['recall']:.3f}
- **F1-Score**: {result['f1_score']:.3f}
- **Number of Classes**: {result['n_classes']}
- **Test Set Size**: {result['test_size']} samples

"""
                if result['accuracy'] > best_accuracy:
                    best_accuracy = result['accuracy']
                    best_model = model_name
        
        report_content += f"""
## 🎯 Recommendation
**Best Performing Model**: {best_model}
- **Accuracy**: {best_accuracy:.1%}

## 📈 Key Insights
- Model performance varies based on feature type and algorithm
- Consider trying different feature engineering approaches
- Evaluate model performance on business-specific metrics
- Monitor for potential overfitting with complex models

---
*Generated by NLP Analyzer Pro*
*Perplexity-style Advanced Text Analytics Platform*
"""
        
        # Create download button
        st.download_button(
            label="📥 Download Comprehensive Analysis Report",
            data=report_content,
            file_name=f"nlp_analysis_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            use_container_width=True
        )

# ============================
# Main Content
# ============================
def main():
    # Initialize session state
    if 'file_uploaded' not in st.session_state:
        st.session_state.file_uploaded = False
    if 'analyze_clicked' not in st.session_state:
        st.session_state.analyze_clicked = False
    if 'results' not in st.session_state:
        st.session_state.results = {}
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'config' not in st.session_state:
        st.session_state.config = {}
    
    # Main Header
    st.markdown("<div class='main-header'>NLP Analyzer Pro</div>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #A0A4B8; font-size: 1.2rem; margin-bottom: 3rem;'>Advanced Text Intelligence Platform</p>", unsafe_allow_html=True)
    
    # Always show live text analysis preview
    CoolFeatures.text_analysis_preview()
    
    # File Upload Section
    st.markdown("<div class='section-header'>📁 Upload Your Dataset</div>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file to analyze",
        type=["csv"],
        help="Upload your dataset containing text data and labels"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.session_state.file_uploaded = True
            
            st.success(f"✅ Successfully loaded dataset with {df.shape[0]:,} rows and {df.shape[1]} features")
            
            # Configuration Section
            st.markdown("<div class='section-header'>⚙️ Analysis Configuration</div>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                text_col = st.selectbox(
                    "Text Column",
                    df.columns,
                    help="Select the column containing text data"
                )
            
            with col2:
                target_col = st.selectbox(
                    "Target Column",
                    df.columns,
                    help="Select the column containing labels"
                )
            
            with col3:
                feature_type = st.selectbox(
                    "Feature Type",
                    ["Lexical", "Semantic", "Syntactic", "Pragmatic"],
                    help="Choose the type of analysis to perform"
                )
            
            # Advanced Options
            with st.expander("🎛️ Advanced Options"):
                col1, col2 = st.columns(2)
                with col1:
                    test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
                with col2:
                    enable_cool_features = st.checkbox("Enable Enhanced Visualizations", value=True)
            
            st.session_state.config = {
                'text_col': text_col,
                'target_col': target_col,
                'feature_type': feature_type,
                'test_size': test_size,
                'enable_cool_features': enable_cool_features
            }
            
            # Analysis Button
            if st.button("🚀 Start Advanced Analysis", use_container_width=True):
                st.session_state.analyze_clicked = True
            else:
                st.session_state.analyze_clicked = False
                
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    else:
        st.session_state.file_uploaded = False
        st.session_state.analyze_clicked = False
        show_welcome_screen()
    
    # Display analysis results if applicable
    if st.session_state.get('file_uploaded', False) and st.session_state.get('analyze_clicked', False):
        perform_analysis()

def show_welcome_screen():
    """Show welcome screen when no file is uploaded"""
    st.markdown("""
    <div class='hero-section'>
        <h1 style='color: white; font-size: 3rem; font-weight: 800; margin-bottom: 1.5rem;'>
            Ready to Analyze Your Text Data?
        </h1>
        <p style='color: #A0A4B8; font-size: 1.3rem; margin-bottom: 2.5rem; line-height: 1.6;'>
            Upload your CSV file to unlock powerful text analysis capabilities with state-of-the-art machine learning algorithms and beautiful visualizations.
        </p>
        <div style='display: inline-flex; gap: 1rem; flex-wrap: wrap; justify-content: center;'>
            <span class="feature-tag">🚀 4 ML Algorithms</span>
            <span class="feature-tag">🔍 Live Analysis</span>
            <span class="feature-tag">☁️ Word Clouds</span>
            <span class="feature-tag">📊 Interactive Charts</span>
            <span class="feature-tag">🔗 Similarity Analysis</span>
            <span class="feature-tag">📄 Download Reports</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature Highlights
    st.markdown("<div class='section-header'>✨ How It Works</div>", unsafe_allow_html=True)
    
    steps = [
        {"icon": "1️⃣", "title": "Upload Data", "desc": "Upload your CSV file with text data and labels"},
        {"icon": "2️⃣", "title": "Configure", "desc": "Select columns and analysis parameters"},
        {"icon": "3️⃣", "title": "Analyze", "desc": "Watch as our algorithms process your data"},
        {"icon": "4️⃣", "title": "Insights", "desc": "Get comprehensive insights and visualizations"}
    ]
    
    cols = st.columns(4)
    for idx, step in enumerate(steps):
        with cols[idx]:
            st.markdown(f"""
            <div class="ppx-card">
                <div style="font-size: 2rem; margin-bottom: 1rem; text-align: center;">{step['icon']}</div>
                <h3 style="color: #2D7FF9; margin-bottom: 1rem; text-align: center;">{step['title']}</h3>
                <p style="color: #A0A4B8; line-height: 1.5; text-align: center;">{step['desc']}</p>
            </div>
            """, unsafe_allow_html=True)

def perform_analysis():
    """Perform the main analysis"""
    df = st.session_state.df
    config = st.session_state.config
    
    if df is None:
        st.error("No dataset loaded.")
        return
    
    # Data validation
    if config['text_col'] not in df.columns or config['target_col'] not in df.columns:
        st.error("Selected columns not found in dataset.")
        return
    
    if df[config['text_col']].isnull().any():
        df[config['text_col']] = df[config['text_col']].fillna('')
    
    if df[config['target_col']].isnull().any():
        st.error("Target column contains missing values.")
        return
    
    if len(df[config['target_col']].unique()) < 2:
        st.error("Target column must have at least 2 unique classes.")
        return
    
    # Show dataset overview
    st.markdown("<div class='section-header'>📊 Dataset Overview</div>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{df.shape[0]:,}</div>
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
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{df.isnull().sum().sum()}</div>
            <div class="metric-label">Missing Values</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        unique_classes = df[config.get('target_col', '')].nunique() if config.get('target_col') in df.columns else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{unique_classes}</div>
            <div class="metric-label">Unique Classes</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Data Preview
    with st.expander("📋 Data Preview", expanded=True):
        tab1, tab2 = st.tabs(["First 10 Rows", "Dataset Statistics"])
        with tab1:
            st.dataframe(df.head(10), use_container_width=True)
        with tab2:
            st.write(df.describe(include='all'))
    
    # Show cool training animation
    CoolFeatures.cool_training_animation()
    
    # Feature extraction
    with st.spinner("🔧 Extracting features..."):
        extractor = PpxFeatureExtractor()
        X = df[config['text_col']].astype(str)
        y = df[config['target_col']]
        
        if config['feature_type'] == "Lexical":
            X_features = extractor.extract_lexical_features(X)
            feature_desc = "Word-level analysis with lemmatization"
        elif config['feature_type'] == "Semantic":
            X_features = extractor.extract_semantic_features(X)
            feature_desc = "Sentiment analysis and text complexity"
        elif config['feature_type'] == "Syntactic":
            X_features = extractor.extract_syntactic_features(X)
            feature_desc = "Grammar structure and POS analysis"
        else:  # Pragmatic
            X_features = extractor.extract_pragmatic_features(X)
            feature_desc = "Context analysis and intent detection"
    
    st.success(f"✅ Feature extraction completed: {feature_desc}")
    
    # Model training
    with st.spinner("🤖 Training machine learning models..."):
        trainer = PpxModelTrainer()
        results, label_encoder = trainer.train_and_evaluate(X_features, y)
        st.session_state.results = results
    
    # Display results
    successful_models = {k: v for k, v in results.items() if 'error' not in v}
    
    if successful_models:
        st.markdown("<div class='section-header'>📈 Analysis Results</div>", unsafe_allow_html=True)
        
        # Model Performance Cards
        st.markdown("#### 🎯 Model Performance")
        
        cols = st.columns(len(successful_models))
        for idx, (model_name, result) in enumerate(successful_models.items()):
            with cols[idx]:
                accuracy = result['accuracy']
                st.markdown(f"""
                <div class="model-card">
                    <h4 style="color: white; margin-bottom: 1rem;">{model_name}</h4>
                    <div class="model-accuracy">{accuracy:.1%}</div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; color: #A0A4B8;">
                        <div style="text-align: center;">
                            <small>Precision</small>
                            <div style="font-weight: bold; color: #2D7FF9;">{result['precision']:.3f}</div>
                        </div>
                        <div style="text-align: center;">
                            <small>Recall</small>
                            <div style="font-weight: bold; color: #2D7FF9;">{result['recall']:.3f}</div>
                        </div>
                        <div style="text-align: center;">
                            <small>F1-Score</small>
                            <div style="font-weight: bold; color: #2D7FF9;">{result['f1_score']:.3f}</div>
                        </div>
                        <div style="text-align: center;">
                            <small>Classes</small>
                            <div style="font-weight: bold; color: #2D7FF9;">{result['n_classes']}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Performance Dashboard
        st.markdown("#### 📊 Performance Dashboard")
        viz = PpxVisualizer()
        dashboard_fig = viz.create_performance_dashboard(successful_models)
        st.pyplot(dashboard_fig)
        
        # Enhanced Cool Features
        if config.get('enable_cool_features', True):
            # Interactive Model Comparison
            CoolFeatures.interactive_model_comparison(successful_models)
            
            # Word Cloud
            CoolFeatures.generate_wordcloud(df, config['text_col'])
            
            # Text Similarity Analysis
            CoolFeatures.text_similarity_analysis(df, config['text_col'])
            
            # Analysis Report
            CoolFeatures.generate_analysis_report(successful_models, df, config)
        
        # Best Model Recommendation
        best_model = max(successful_models.items(), key=lambda x: x[1]['accuracy'])
        st.markdown(f"""
        <div class="ppx-card">
            <h3 style="color: #2D7FF9; margin-bottom: 1rem;">🎯 Recommended Model</h3>
            <p style="color: white; font-size: 1.2rem;">
                <strong>{best_model[0]}</strong> achieved the highest accuracy of 
                <strong style="color: #2D7FF9;">{best_model[1]['accuracy']:.1%}</strong>
            </p>
            <p style="color: #A0A4B8;">This model is recommended for deployment based on comprehensive performance metrics.</p>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        st.error("❌ No models were successfully trained. Please check your data and configuration.")

if __name__ == "__main__":
    main()

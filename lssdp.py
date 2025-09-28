# ============================================
# 🎯 NLP Analysis Suite 
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

import matplotlib.pyplot as plt
import seaborn as sns

# ============================
# Configuration
# ============================
st.set_page_config(
    page_title="NLP Pro | Amazon Prime Style",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================
# Enhanced Amazon Prime Style CSS
# ============================
st.markdown("""
<style>
    /* Enhanced Amazon Prime Color Scheme */
    :root {
        --prime-blue: #146EB4;
        --prime-dark: #0F4E8A;
        --prime-navy: #1A3E6B;
        --prime-royal: #2D5A8A;
        --prime-white: #FFFFFF;
        --prime-light: #F5F8FA;
        --prime-orange: #FF9900;
        --prime-text: #232F3E;
        --prime-text-light: #4A5568;
    }
    
    .main-header {
        font-size: 3.5rem;
        color: var(--prime-navy);
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .prime-card {
        background: var(--prime-white);
        border: 1px solid #E1E8ED;
        border-radius: 12px;
        padding: 1.8rem;
        margin: 0.8rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
        border-left: 4px solid var(--prime-blue);
    }
    
    .prime-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(20, 110, 180, 0.15);
        border-left-color: var(--prime-orange);
    }
    
    .metric-highlight {
        background: linear-gradient(135deg, var(--prime-blue) 0%, var(--prime-dark) 100%);
        color: var(--prime-white);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 0.8rem 0;
        box-shadow: 0 4px 12px rgba(20, 110, 180, 0.2);
    }
    
    .prime-button {
        background: linear-gradient(135deg, var(--prime-orange) 0%, #E67E22 100%) !important;
        color: var(--prime-white) !important;
        border: none !important;
        padding: 1rem 2.5rem !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(255, 153, 0, 0.3) !important;
    }
    
    .prime-button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(255, 153, 0, 0.4) !important;
    }
    
    .section-title {
        font-size: 2rem;
        color: var(--prime-navy);
        margin: 2.5rem 0 1.5rem 0;
        font-weight: 700;
        border-bottom: 3px solid var(--prime-blue);
        padding-bottom: 0.8rem;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
    }
    
    .feature-tag {
        background: linear-gradient(135deg, var(--prime-light) 0%, #E1E8ED 100%);
        color: var(--prime-text);
        padding: 0.6rem 1.2rem;
        border-radius: 25px;
        font-size: 0.9rem;
        font-weight: 600;
        margin: 0.4rem;
        display: inline-block;
        border: 1px solid #D1D9E0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 0.8rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .status-active { background: #27AE60; }
    .status-warning { background: #F39C12; }
    .status-error { background: #E74C3C; }
    
    .nav-bar {
        background: linear-gradient(135deg, var(--prime-navy) 0%, var(--prime-dark) 100%);
        padding: 1.5rem 3rem;
        border-radius: 12px;
        margin-bottom: 2.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    .nav-title {
        color: var(--prime-white);
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .insight-box {
        background: linear-gradient(135deg, #F8FBFF 0%, #E8F2FF 100%);
        border-left: 4px solid var(--prime-blue);
        padding: 2rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 12px rgba(20, 110, 180, 0.1);
        border: 1px solid #E1E8ED;
    }
    
    .performance-badge {
        background: linear-gradient(135deg, var(--prime-orange) 0%, #E67E22 100%);
        color: var(--prime-white);
        padding: 0.8rem 1.5rem;
        border-radius: 25px;
        font-size: 1rem;
        font-weight: 700;
        display: inline-block;
        box-shadow: 0 4px 12px rgba(255, 153, 0, 0.3);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: var(--prime-blue);
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .metric-label {
        font-size: 1rem;
        color: var(--prime-text-light);
        font-weight: 600;
        margin-top: 0.5rem;
    }
    
    /* Text color enhancements */
    .stMarkdown, .stText, .stLabel {
        color: var(--prime-text) !important;
    }
    
    .stSelectbox label, .stRadio label, .stSlider label {
        color: var(--prime-text) !important;
        font-weight: 600 !important;
    }
    
    .stExpander {
        border: 1px solid #E1E8ED !important;
        border-radius: 12px !important;
        margin: 1rem 0 !important;
    }
    
    .stExpander summary {
        color: var(--prime-navy) !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
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
class PrimeFeatureExtractor:
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
# Prime Model Trainer
# ============================
class PrimeModelTrainer:
    def __init__(self):
        self.models = {
            "🎯 Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
            "🌲 Random Forest": RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced'),
            "⚡ Support Vector Machine": SVC(random_state=42, probability=True, class_weight='balanced'),
            "📊 Naive Bayes": MultinomialNB()
        }
    
    def train_and_evaluate(self, X, y):
        """Prime-style model training with comprehensive evaluation"""
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
                cols = st.columns([1, 4])
                with cols[0]:
                    st.markdown(f"**{name}**")
                with cols[1]:
                    progress_bar = st.progress(0)
                    for step in range(5):
                        progress_bar.progress((step + 1) / 5)
                        import time
                        time.sleep(0.1)
            
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
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
                    'n_classes': n_classes,
                    'test_size': len(y_test)
                }
                
            except Exception as e:
                results[name] = {'error': str(e)}
        
        progress_container.empty()
        return results, le

# ============================
# Enhanced Visualization Engine
# ============================
class PrimeVisualizer:
    @staticmethod
    def create_performance_dashboard(results):
        """Create enhanced performance dashboard"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Performance Dashboard', fontsize=18, fontweight='bold', color='#1A3E6B')
        
        # Set overall style
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.facecolor'] = '#F5F8FA'
        
        models = []
        metrics = {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-Score': []}
        
        for model_name, result in results.items():
            if 'error' not in result:
                clean_name = model_name.replace('🎯 ', '').replace('🌲 ', '').replace('⚡ ', '').replace('📊 ', '')
                models.append(clean_name)
                metrics['Accuracy'].append(result['accuracy'])
                metrics['Precision'].append(result['precision'])
                metrics['Recall'].append(result['recall'])
                metrics['F1-Score'].append(result['f1_score'])
        
        # Color scheme
        colors = ['#146EB4', '#FF9900', '#27AE60', '#E74C3C']
        
        # Accuracy plot
        bars1 = ax1.bar(models, metrics['Accuracy'], color=colors, alpha=0.9)
        ax1.set_facecolor('#F8FBFF')
        ax1.set_title('Accuracy Comparison', fontweight='bold', fontsize=14, color='#1A3E6B')
        ax1.set_ylabel('Accuracy', fontweight='bold', color='#4A5568')
        ax1.tick_params(axis='x', rotation=45, colors='#4A5568')
        ax1.tick_params(axis='y', colors='#4A5568')
        ax1.grid(True, alpha=0.3)
        
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold', color='#1A3E6B')
        
        # F1-Score plot
        bars2 = ax2.bar(models, metrics['F1-Score'], color=colors, alpha=0.9)
        ax2.set_facecolor('#F8FBFF')
        ax2.set_title('F1-Score Comparison', fontweight='bold', fontsize=14, color='#1A3E6B')
        ax2.set_ylabel('F1-Score', fontweight='bold', color='#4A5568')
        ax2.tick_params(axis='x', rotation=45, colors='#4A5568')
        ax2.tick_params(axis='y', colors='#4A5568')
        ax2.grid(True, alpha=0.3)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold', color='#1A3E6B')
        
        # Precision-Recall comparison
        x_index = np.arange(len(models))
        width = 0.35
        bars3 = ax3.bar(x_index - width/2, metrics['Precision'], width, label='Precision', 
                       color='#146EB4', alpha=0.9)
        bars4 = ax3.bar(x_index + width/2, metrics['Recall'], width, label='Recall', 
                       color='#FF9900', alpha=0.9)
        ax3.set_facecolor('#F8FBFF')
        ax3.set_title('Precision vs Recall', fontweight='bold', fontsize=14, color='#1A3E6B')
        ax3.set_ylabel('Scores', fontweight='bold', color='#4A5568')
        ax3.set_xticks(x_index)
        ax3.set_xticklabels(models, rotation=45, color='#4A5568')
        ax3.tick_params(axis='y', colors='#4A5568')
        ax3.legend(facecolor='#F8FBFF')
        ax3.grid(True, alpha=0.3)
        
        # Model comparison radar
        metrics_array = np.array([metrics['Accuracy'], metrics['Precision'], 
                                metrics['Recall'], metrics['F1-Score']])
        metrics_normalized = metrics_array / metrics_array.max(axis=1, keepdims=True)
        
        angles = np.linspace(0, 2*np.pi, len(metrics_normalized), endpoint=False).tolist()
        angles += angles[:1]
        
        for i, model in enumerate(models):
            values = metrics_normalized[:, i].tolist()
            values += values[:1]
            ax4.plot(angles, values, 'o-', linewidth=3, label=model, color=colors[i], markersize=8)
            ax4.fill(angles, values, alpha=0.1, color=colors[i])
        
        ax4.set_facecolor('#F8FBFF')
        ax4.set_yticklabels([])
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1-Score'], 
                           color='#4A5568', fontweight='bold')
        ax4.set_title('Performance Radar', fontweight='bold', fontsize=14, color='#1A3E6B')
        ax4.legend(facecolor='#F8FBFF', bbox_to_anchor=(1.2, 1.0))
        
        plt.tight_layout()
        return fig

# ============================
# Main Application
# ============================
def main():
    # Enhanced Amazon Prime Style Header
    st.markdown("""
    <div class="nav-bar">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 class="nav-title">🎯 NLP Pro Analysis Suite</h1>
                <p style="color: #A0BCC8; margin: 0; font-size: 1.1rem;">Enterprise-grade Text Analytics Platform</p>
            </div>
            <div style="display: flex; gap: 1.5rem; align-items: center;">
                <span class="status-indicator status-active"></span>
                <span style="color: white; font-weight: 600;">System Online</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='main-header'>Advanced Text Intelligence Platform</div>", unsafe_allow_html=True)
    
    # Enhanced Feature Highlights
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-highlight">
            <h3 style="color: white; margin: 0;">📖 Lexical</h3>
            <p style="color: #E1F0FF; margin: 0.5rem 0 0 0;">Word-level Analysis</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-highlight">
            <h3 style="color: white; margin: 0;">🎭 Semantic</h3>
            <p style="color: #E1F0FF; margin: 0.5rem 0 0 0;">Meaning & Sentiment</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-highlight">
            <h3 style="color: white; margin: 0;">🔧 Syntactic</h3>
            <p style="color: #E1F0FF; margin: 0.5rem 0 0 0;">Grammar & Structure</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-highlight">
            <h3 style="color: white; margin: 0;">🎯 Pragmatic</h3>
            <p style="color: #E1F0FF; margin: 0.5rem 0 0 0;">Context & Intent</p>
        </div>
        """, unsafe_allow_html=True)
    
    # File Upload Section
    st.markdown('<div class="section-title">📁 Data Integration</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("", type=["csv"], 
                                   help="Upload your dataset in CSV format")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Enhanced Success Message
            st.markdown(f"""
            <div class="insight-box">
                <h4 style="color: #146EB4; margin-bottom: 1rem;">✅ Dataset Loaded Successfully</h4>
                <p style="color: #4A5568; margin: 0.5rem 0;"><strong>Dimensions:</strong> {df.shape[0]} records × {df.shape[1]} features</p>
                <p style="color: #4A5568; margin: 0.5rem 0;"><strong>Memory Usage:</strong> {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Enhanced Data Explorer
            with st.expander("🔍 Data Explorer", expanded=True):
                tab1, tab2 = st.tabs(["📊 Preview", "📈 Statistics"])
                
                with tab1:
                    st.dataframe(df.head(8), use_container_width=True)
                
                with tab2:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.markdown(f'<div class="metric-value">{df.shape[0]}</div>', unsafe_allow_html=True)
                        st.markdown('<div class="metric-label">Total Records</div>', unsafe_allow_html=True)
                    with col2:
                        st.markdown(f'<div class="metric-value">{df.shape[1]}</div>', unsafe_allow_html=True)
                        st.markdown('<div class="metric-label">Features</div>', unsafe_allow_html=True)
                    with col3:
                        st.markdown(f'<div class="metric-value">{df.isnull().sum().sum()}</div>', unsafe_allow_html=True)
                        st.markdown('<div class="metric-label">Missing Values</div>', unsafe_allow_html=True)
                    with col4:
                        st.markdown(f'<div class="metric-value">{len(df.dtypes.unique())}</div>', unsafe_allow_html=True)
                        st.markdown('<div class="metric-label">Data Types</div>', unsafe_allow_html=True)
            
            # Analysis Configuration
            st.markdown('<div class="section-title">⚙️ Analysis Configuration</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                text_col = st.selectbox("Text Column", df.columns, 
                                      help="Select the column containing text data")
            with col2:
                target_col = st.selectbox("Target Column", df.columns,
                                        help="Select the column containing labels")
            
            # Feature Type Selection
            st.markdown('<div class="section-title">🎯 Feature Engineering</div>', unsafe_allow_html=True)
            
            feature_type = st.radio("Select Analysis Type:", 
                                  ["📖 Lexical Features", "🎭 Semantic Features", 
                                   "🔧 Syntactic Features", "🎯 Pragmatic Features"],
                                  horizontal=True)
            
            # Enhanced Analysis Button
            if st.button("🚀 Launch Advanced Analysis", use_container_width=True):
                # Data Validation
                if df[text_col].isnull().any():
                    df[text_col] = df[text_col].fillna('')
                
                if df[target_col].isnull().any():
                    st.error("Target column contains missing values. Please clean your data.")
                    return
                
                if len(df[target_col].unique()) < 2:
                    st.error("Target column must have at least 2 unique classes.")
                    return
                
                # Enhanced Class Distribution
                st.markdown('<div class="section-title">📊 Class Distribution Analysis</div>', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(12, 5))
                value_counts = df[target_col].value_counts()
                bars = ax.bar(range(len(value_counts)), value_counts.values, 
                            color='#146EB4', alpha=0.8, edgecolor='#0F4E8A', linewidth=2)
                ax.set_facecolor('#F8FBFF')
                ax.set_xlabel('Classes', fontweight='bold', color='#4A5568')
                ax.set_ylabel('Count', fontweight='bold', color='#4A5568')
                ax.set_title('Class Distribution Analysis', fontweight='bold', fontsize=14, color='#1A3E6B')
                ax.grid(True, alpha=0.3)
                
                for bar, count in zip(bars, value_counts.values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                           str(count), ha='center', va='bottom', fontweight='bold', color='#1A3E6B')
                
                st.pyplot(fig)
                
                # Feature Extraction
                with st.spinner("🔄 Extracting advanced features..."):
                    extractor = PrimeFeatureExtractor()
                    X = df[text_col].astype(str)
                    y = df[target_col]
                    
                    if feature_type == "📖 Lexical Features":
                        X_features = extractor.extract_lexical_features(X)
                        feature_desc = "Word-level analysis with lemmatization and n-grams"
                    elif feature_type == "🎭 Semantic Features":
                        X_features = extractor.extract_semantic_features(X)
                        feature_desc = "Sentiment analysis and text complexity features"
                    elif feature_type == "🔧 Syntactic Features":
                        X_features = extractor.extract_syntactic_features(X)
                        feature_desc = "Grammar structure and part-of-speech analysis"
                    else:  # Pragmatic Features
                        X_features = extractor.extract_pragmatic_features(X)
                        feature_desc = "Context analysis, modality, and intent detection"
                
                # Feature Extraction Success
                st.markdown(f"""
                <div class="insight-box">
                    <h4 style="color: #146EB4; margin-bottom: 1rem;">✅ Feature Extraction Complete</h4>
                    <p style="color: #4A5568; margin: 0.5rem 0;"><strong>Feature Type:</strong> {feature_type}</p>
                    <p style="color: #4A5568; margin: 0.5rem 0;"><strong>Description:</strong> {feature_desc}</p>
                    <p style="color: #4A5568; margin: 0.5rem 0;"><strong>Feature Matrix:</strong> {X_features.shape}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Model Training
                with st.spinner("🤖 Training advanced models..."):
                    trainer = PrimeModelTrainer()
                    results, label_encoder = trainer.train_and_evaluate(X_features, y)
                
                # Enhanced Results Display
                st.markdown('<div class="section-title">📈 Performance Results</div>', unsafe_allow_html=True)
                
                successful_models = {k: v for k, v in results.items() if 'error' not in v}
                
                if successful_models:
                    # Performance Metrics Cards
                    st.markdown("#### 🏆 Model Performance Summary")
                    
                    cols = st.columns(len(successful_models))
                    for idx, (model_name, result) in enumerate(successful_models.items()):
                        with cols[idx]:
                            accuracy = result['accuracy']
                            badge_color = "#27AE60" if accuracy > 0.8 else "#F39C12" if accuracy > 0.6 else "#E74C3C"
                            
                            st.markdown(f"""
                            <div class="prime-card">
                                <h4 style="color: #1A3E6B; margin-bottom: 1rem;">{model_name}</h4>
                                <div class="performance-badge" style="background: {badge_color}">
                                    {accuracy:.1%}
                                </div>
                                <p style="color: #4A5568; margin: 0.5rem 0;"><small>F1: {result['f1_score']:.3f}</small></p>
                                <p style="color: #4A5568; margin: 0.5rem 0;"><small>Precision: {result['precision']:.3f}</small></p>
                                <p style="color: #4A5568; margin: 0.5rem 0;"><small>Classes: {result['n_classes']}</small></p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Enhanced Performance Dashboard
                    st.markdown("#### 📊 Performance Dashboard")
                    viz = PrimeVisualizer()
                    dashboard_fig = viz.create_performance_dashboard(successful_models)
                    st.pyplot(dashboard_fig)
                    
                    # Best Model Highlight
                    best_model_name = max(successful_models.items(), key=lambda x: x[1]['accuracy'])[0]
                    best_accuracy = successful_models[best_model_name]['accuracy']
                    
                    st.markdown(f"""
                    <div class="insight-box">
                        <h4 style="color: #146EB4; margin-bottom: 1rem;">🎯 Recommended Model</h4>
                        <p style="color: #4A5568; margin: 0.5rem 0;"><strong>{best_model_name}</strong> achieved the highest accuracy of <strong style="color: #27AE60;">{best_accuracy:.1%}</strong></p>
                        <p style="color: #4A5568; margin: 0.5rem 0;">This model is recommended for deployment based on comprehensive performance metrics.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                else:
                    st.error("No models were successfully trained. Please check your data and configuration.")
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    else:
        # Enhanced Welcome Section
        st.markdown("""
        <div style='text-align: center; padding: 4rem 2rem; background: linear-gradient(135deg, #F8FBFF 0%, #E8F2FF 100%); 
                    border-radius: 16px; margin: 3rem 0; border: 2px dashed #146EB4;'>
            <h2 style='color: #1A3E6B; margin-bottom: 1.5rem; font-size: 2.5rem;'>🚀 Get Started with NLP Pro</h2>
            <p style='color: #4A5568; font-size: 1.3rem; margin-bottom: 2.5rem; font-weight: 500;'>
                Upload your CSV file to unlock powerful text analysis capabilities
            </p>
            <div style="display: inline-flex; gap: 1.5rem; flex-wrap: wrap; justify-content: center;">
                <span class="feature-tag">🤖 4 Advanced Models</span>
                <span class="feature-tag">🎯 Pragmatic Analysis</span>
                <span class="feature-tag">📊 Real-time Analytics</span>
                <span class="feature-tag">⚡ Enterprise Grade</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced Feature Showcase
        st.markdown('<div class="section-title">✨ Advanced Features</div>', unsafe_allow_html=True)
        
        features = [
            {"icon": "📖", "title": "Lexical Analysis", "desc": "Advanced tokenization and lemmatization"},
            {"icon": "🎭", "title": "Semantic Intelligence", "desc": "Sentiment analysis and meaning extraction"},
            {"icon": "🔧", "title": "Syntactic Processing", "desc": "Grammar and structure analysis"},
            {"icon": "🎯", "title": "Pragmatic Context", "desc": "Intent detection and modality analysis"}
        ]
        
        cols = st.columns(4)
        for idx, feature in enumerate(features):
            with cols[idx]:
                st.markdown(f"""
                <div class="prime-card">
                    <h4 style="color: #1A3E6B; margin-bottom: 1rem;">{feature['icon']} {feature['title']}</h4>
                    <p style="color: #4A5568; font-size: 0.95rem; line-height: 1.5;">{feature['desc']}</p>
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

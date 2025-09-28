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
# Style CSS
# ============================
st.markdown("""
<style>
    /* Amazon Prime Color Scheme */
    :root {
        --prime-blue: #00A8E1;
        --prime-dark: #146EB4;
        --prime-black: #232F3E;
        --prime-white: #FFFFFF;
        --prime-gray: #EAEDED;
        --prime-orange: #FF9900;
    }
    
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(135deg, var(--prime-blue) 0%, var(--prime-dark) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .prime-card {
        background: var(--prime-white);
        border: 1px solid #DDD;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        border-left: 4px solid var(--prime-blue);
    }
    
    .prime-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 168, 225, 0.15);
        border-left-color: var(--prime-orange);
    }
    
    .metric-highlight {
        background: linear-gradient(135deg, var(--prime-blue) 0%, var(--prime-dark) 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .prime-button {
        background: linear-gradient(135deg, var(--prime-orange) 0%, #FF8C00 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        border-radius: 4px !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        transition: all 0.3s ease !important;
    }
    
    .prime-button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(255, 153, 0, 0.3) !important;
    }
    
    .section-title {
        font-size: 1.8rem;
        color: var(--prime-black);
        margin: 2rem 0 1rem 0;
        font-weight: 700;
        border-bottom: 3px solid var(--prime-blue);
        padding-bottom: 0.5rem;
    }
    
    .feature-tag {
        background: var(--prime-gray);
        color: var(--prime-black);
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        margin: 0.3rem;
        display: inline-block;
        border: 1px solid #DDD;
    }
    
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 0.5rem;
    }
    
    .status-active { background: #10B981; }
    .status-warning { background: #F59E0B; }
    .status-error { background: #EF4444; }
    
    .nav-bar {
        background: var(--prime-black);
        padding: 1rem 2rem;
        border-radius: 8px;
        margin-bottom: 2rem;
    }
    
    .nav-title {
        color: var(--prime-white);
        font-size: 1.5rem;
        font-weight: 600;
        margin: 0;
    }
    
    .insight-box {
        background: linear-gradient(135deg, #F0F9FF 0%, #E0F2FE 100%);
        border-left: 4px solid var(--prime-blue);
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .performance-badge {
        background: var(--prime-orange);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        display: inline-block;
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
            # Extended semantic features
            features.append([
                blob.sentiment.polarity,
                blob.sentiment.subjectivity,
                len(text.split()),  # Word count
                len([word for word in text.split() if len(word) > 6]),  # Complex words
            ])
        return np.array(features)
    
    @staticmethod
    def extract_syntactic_features(texts):
        """Extract syntactic features with POS analysis"""
        processed_texts = []
        for text in texts:
            doc = nlp(str(text))
            pos_tags = [f"{token.pos_}_{token.tag_}" for token in doc]  # More detailed POS
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
            
            # Modality analysis
            for category, words in pragmatic_indicators.items():
                count = sum(text_lower.count(word) for word in words)
                features.append(count)
            
            # Structural features
            features.extend([
                text.count('!'),  # Exclamation marks
                text.count('?'),  # Question marks
                len([s for s in text.split('.') if s.strip()]),  # Sentence count
                len([w for w in text.split() if w.istitle()]),  # Proper nouns count
            ])
            
            pragmatic_features.append(features)
        
        return np.array(pragmatic_features)

# ============================
# Model Trainer
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
        
        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        n_classes = len(le.classes_)
        
        # Smart data splitting
        test_size = max(0.15, min(0.25, 3 * n_classes / len(y_encoded)))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        # Training progress with Prime style
        progress_container = st.empty()
        
        for i, (name, model) in enumerate(self.models.items()):
            with progress_container.container():
                cols = st.columns([1, 4])
                with cols[0]:
                    st.markdown(f"**{name}**")
                with cols[1]:
                    progress_bar = st.progress(0)
                    
                    for step in range(5):  # Simulate steps for better UX
                        progress_bar.progress((step + 1) / 5)
                        # Simulate processing time
                        import time
                        time.sleep(0.1)
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred = model.predict(X_test)
                
                # Comprehensive metrics
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
# Visualization Engine
# ============================
class PrimeVisualizer:
    @staticmethod
    def create_performance_dashboard(results):
        """Create Amazon-style performance dashboard"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Performance Dashboard', fontsize=16, fontweight='bold', color='#232F3E')
        
        models = []
        metrics = {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-Score': []}
        
        for model_name, result in results.items():
            if 'error' not in result:
                models.append(model_name.replace('🎯 ', '').replace('🌲 ', '').replace('⚡ ', '').replace('📊 ', ''))
                metrics['Accuracy'].append(result['accuracy'])
                metrics['Precision'].append(result['precision'])
                metrics['Recall'].append(result['recall'])
                metrics['F1-Score'].append(result['f1_score'])
        
        # Accuracy plot
        bars1 = ax1.bar(models, metrics['Accuracy'], color='#00A8E1', alpha=0.8)
        ax1.set_title('Accuracy Comparison', fontweight='bold')
        ax1.set_ylabel('Accuracy')
        ax1.tick_params(axis='x', rotation=45)
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        
        # F1-Score plot
        bars2 = ax2.bar(models, metrics['F1-Score'], color='#FF9900', alpha=0.8)
        ax2.set_title('F1-Score Comparison', fontweight='bold')
        ax2.set_ylabel('F1-Score')
        ax2.tick_params(axis='x', rotation=45)
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        
        # Precision-Recell comparison
        x_index = np.arange(len(models))
        width = 0.35
        bars3 = ax3.bar(x_index - width/2, metrics['Precision'], width, label='Precision', color='#10B981', alpha=0.8)
        bars4 = ax3.bar(x_index + width/2, metrics['Recall'], width, label='Recall', color='#EF4444', alpha=0.8)
        ax3.set_title('Precision vs Recall', fontweight='bold')
        ax3.set_ylabel('Scores')
        ax3.set_xticks(x_index)
        ax3.set_xticklabels(models, rotation=45)
        ax3.legend()
        
        # Radar chart preparation
        metrics_array = np.array([metrics['Accuracy'], metrics['Precision'], metrics['Recall'], metrics['F1-Score']])
        metrics_normalized = metrics_array / metrics_array.max(axis=1, keepdims=True)
        
        angles = np.linspace(0, 2*np.pi, len(metrics_normalized), endpoint=False).tolist()
        angles += angles[:1]
        
        for i, model in enumerate(models):
            values = metrics_normalized[:, i].tolist()
            values += values[:1]
            ax4.plot(angles, values, 'o-', linewidth=2, label=model)
            ax4.fill(angles, values, alpha=0.1)
        
        ax4.set_yticklabels([])
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1-Score'])
        ax4.set_title('Performance Radar', fontweight='bold')
        ax4.legend(bbox_to_anchor=(1.1, 1.05))
        
        plt.tight_layout()
        return fig

# ============================
# Main Application
# ============================
def main():
    # Style Header
    st.markdown("""
    <div class="nav-bar">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 class="nav-title">🎯 NLP Pro Analysis Suite</h1>
                <p style="color: #AAA; margin: 0;">Enterprise-grade Text Analytics</p>
            </div>
            <div style="display: flex; gap: 1rem;">
                <span class="status-indicator status-active"></span>
                <span style="color: white;">System Online</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='main-header'>Advanced Text Intelligence Platform</div>", unsafe_allow_html=True)
    
    # Feature Highlights
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-highlight">
            <h3>📖 Lexical</h3>
            <p>Word-level Analysis</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-highlight">
            <h3>🎭 Semantic</h3>
            <p>Meaning & Sentiment</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-highlight">
            <h3>🔧 Syntactic</h3>
            <p>Grammar & Structure</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-highlight">
            <h3>🎯 Pragmatic</h3>
            <p>Context & Intent</p>
        </div>
        """, unsafe_allow_html=True)
    
    # File Upload Section
    st.markdown('<div class="section-title">📁 Data Integration</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("", type=["csv"], 
                                   help="Upload your dataset in CSV format")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Success Message
            st.markdown(f"""
            <div class="insight-box">
                <h4>✅ Dataset Loaded Successfully</h4>
                <p><strong>Dimensions:</strong> {df.shape[0]} records × {df.shape[1]} features</p>
                <p><strong>Memory Usage:</strong> {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Data Explorer
            with st.expander("🔍 Data Explorer", expanded=True):
                tab1, tab2 = st.tabs(["Preview", "Statistics"])
                
                with tab1:
                    st.dataframe(df.head(8), use_container_width=True)
                
                with tab2:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Records", df.shape[0])
                        st.metric("Number of Features", df.shape[1])
                    with col2:
                        st.metric("Missing Values", df.isnull().sum().sum())
                        st.metric("Data Types", len(df.dtypes.unique()))
            
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
            
            # Advanced Options
            with st.expander("⚡ Advanced Configuration"):
                col1, col2 = st.columns(2)
                with col1:
                    test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
                with col2:
                    max_features = st.slider("Feature Complexity", 100, 2000, 800, 100)
            
            # Analysis Button
            if st.button("🚀 Launch Analysis", use_container_width=True):
                # Data Validation
                if df[text_col].isnull().any():
                    df[text_col] = df[text_col].fillna('')
                
                if df[target_col].isnull().any():
                    st.error("Target column contains missing values. Please clean your data.")
                    return
                
                if len(df[target_col].unique()) < 2:
                    st.error("Target column must have at least 2 unique classes.")
                    return
                
                # Show Class Distribution
                st.markdown('<div class="section-title">📊 Class Distribution</div>', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(10, 4))
                value_counts = df[target_col].value_counts()
                bars = ax.bar(range(len(value_counts)), value_counts.values, color='#00A8E1', alpha=0.8)
                ax.set_xlabel('Classes')
                ax.set_ylabel('Count')
                ax.set_title('Class Distribution Analysis')
                
                for bar, count in zip(bars, value_counts.values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                           str(count), ha='center', va='bottom', fontweight='bold')
                
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
                    <h4>✅ Feature Extraction Complete</h4>
                    <p><strong>Feature Type:</strong> {feature_type}</p>
                    <p><strong>Description:</strong> {feature_desc}</p>
                    <p><strong>Feature Matrix:</strong> {X_features.shape}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Model Training
                with st.spinner("🤖 Training advanced models..."):
                    trainer = PrimeModelTrainer()
                    results, label_encoder = trainer.train_and_evaluate(X_features, y)
                
                # Results Display
                st.markdown('<div class="section-title">📈 Performance Results</div>', unsafe_allow_html=True)
                
                successful_models = {k: v for k, v in results.items() if 'error' not in v}
                
                if successful_models:
                    # Performance Metrics Cards
                    st.markdown("#### 🏆 Model Performance Summary")
                    
                    cols = st.columns(len(successful_models))
                    for idx, (model_name, result) in enumerate(successful_models.items()):
                        with cols[idx]:
                            accuracy = result['accuracy']
                            badge_color = "#10B981" if accuracy > 0.8 else "#F59E0B" if accuracy > 0.6 else "#EF4444"
                            
                            st.markdown(f"""
                            <div class="prime-card">
                                <h4>{model_name}</h4>
                                <div class="performance-badge" style="background: {badge_color}">
                                    {accuracy:.1%}
                                </div>
                                <p><small>F1: {result['f1_score']:.3f}</small></p>
                                <p><small>Precision: {result['precision']:.3f}</small></p>
                                <p><small>Classes: {result['n_classes']}</small></p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Performance Dashboard
                    st.markdown("#### 📊 Performance Dashboard")
                    viz = PrimeVisualizer()
                    dashboard_fig = viz.create_performance_dashboard(successful_models)
                    st.pyplot(dashboard_fig)
                    
                    # Best Model Highlight
                    best_model_name = max(successful_models.items(), key=lambda x: x[1]['accuracy'])[0]
                    best_accuracy = successful_models[best_model_name]['accuracy']
                    
                    st.markdown(f"""
                    <div class="insight-box">
                        <h4>🎯 Recommended Model</h4>
                        <p><strong>{best_model_name}</strong> achieved the highest accuracy of <strong>{best_accuracy:.1%}</strong></p>
                        <p>This model is recommended for deployment based on comprehensive performance metrics.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                else:
                    st.error("No models were successfully trained. Please check your data and configuration.")
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    else:
        # Welcome Section
        st.markdown("""
        <div style='text-align: center; padding: 3rem 2rem; background: linear-gradient(135deg, #F0F9FF 0%, #E0F2FE 100%); 
                    border-radius: 12px; margin: 2rem 0; border: 2px dashed #00A8E1;'>
            <h2 style='color: #232F3E; margin-bottom: 1rem;'>🚀 Get Started with NLP Pro</h2>
            <p style='color: #4A5568; font-size: 1.2rem; margin-bottom: 2rem;'>
                Upload your CSV file to unlock powerful text analysis capabilities
            </p>
            <div style="display: inline-flex; gap: 1rem; flex-wrap: wrap; justify-content: center;">
                <span class="feature-tag">🤖 4 Advanced Models</span>
                <span class="feature-tag">🎯 Pragmatic Analysis</span>
                <span class="feature-tag">📊 Real-time Analytics</span>
                <span class="feature-tag">⚡ Enterprise Grade</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature Showcase
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
                    <h4>{feature['icon']} {feature['title']}</h4>
                    <p style="color: #666; font-size: 0.9rem;">{feature['desc']}</p>
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

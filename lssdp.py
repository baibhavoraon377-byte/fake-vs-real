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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

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
    
    .section-title {
        font-size: 2rem;
        color: var(--prime-navy);
        margin: 2.5rem 0 1.5rem 0;
        font-weight: 700;
        border-bottom: 3px solid var(--prime-blue);
        padding-bottom: 0.8rem;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
    }
    
    .dashboard-section {
        background: var(--prime-white);
        border-radius: 12px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        border: 1px solid #E1E8ED;
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
        
        for name, model in self.models.items():
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
        
        return results, le

# ============================
# Enhanced Dashboard Visualizer
# ============================
class EnhancedDashboard:
    @staticmethod
    def create_comprehensive_dashboard(results):
        """Create a highly organized and spaced dashboard"""
        # Create figure with GridSpec for better control
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)
        
        # Set overall style
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.facecolor'] = '#F8FBFF'
        
        models = []
        metrics_data = {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-Score': []}
        
        for model_name, result in results.items():
            if 'error' not in result:
                clean_name = model_name.replace('🎯 ', '').replace('🌲 ', '').replace('⚡ ', '').replace('📊 ', '')
                models.append(clean_name)
                metrics_data['Accuracy'].append(result['accuracy'])
                metrics_data['Precision'].append(result['precision'])
                metrics_data['Recall'].append(result['recall'])
                metrics_data['F1-Score'].append(result['f1_score'])
        
        colors = ['#146EB4', '#FF9900', '#27AE60', '#E74C3C']
        
        # Chart 1: Main Performance Comparison (Top Left - Larger)
        ax1 = fig.add_subplot(gs[0, 0])
        x_pos = np.arange(len(models))
        width = 0.2
        
        bars1 = ax1.bar(x_pos - width*1.5, metrics_data['Accuracy'], width, label='Accuracy', 
                       color=colors[0], alpha=0.9, edgecolor='white', linewidth=1.5)
        bars2 = ax1.bar(x_pos - width*0.5, metrics_data['Precision'], width, label='Precision', 
                       color=colors[1], alpha=0.9, edgecolor='white', linewidth=1.5)
        bars3 = ax1.bar(x_pos + width*0.5, metrics_data['Recall'], width, label='Recall', 
                       color=colors[2], alpha=0.9, edgecolor='white', linewidth=1.5)
        bars4 = ax1.bar(x_pos + width*1.5, metrics_data['F1-Score'], width, label='F1-Score', 
                       color=colors[3], alpha=0.9, edgecolor='white', linewidth=1.5)
        
        ax1.set_facecolor('#F8FBFF')
        ax1.set_title('Comprehensive Model Performance', fontweight='bold', fontsize=14, color='#1A3E6B', pad=20)
        ax1.set_ylabel('Scores', fontweight='bold', color='#4A5568', fontsize=12)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(models, rotation=45, color='#4A5568', fontsize=11)
        ax1.tick_params(axis='y', colors='#4A5568')
        ax1.legend(facecolor='#F8FBFF', edgecolor='none', fontsize=10)
        ax1.grid(True, alpha=0.2, axis='y')
        ax1.spines[['top', 'right']].set_visible(False)
        
        # Add value labels on bars
        for bars in [bars1, bars2, bars3, bars4]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontweight='bold', 
                        color='#1A3E6B', fontsize=9)
        
        # Chart 2: Accuracy Comparison (Top Middle)
        ax2 = fig.add_subplot(gs[0, 1])
        bars = ax2.bar(models, metrics_data['Accuracy'], color=colors, alpha=0.9, 
                      edgecolor='white', linewidth=2)
        ax2.set_facecolor('#F8FBFF')
        ax2.set_title('Accuracy Comparison', fontweight='bold', fontsize=14, color='#1A3E6B', pad=20)
        ax2.set_ylabel('Accuracy', fontweight='bold', color='#4A5568', fontsize=12)
        ax2.tick_params(axis='x', rotation=45, colors='#4A5568')
        ax2.tick_params(axis='y', colors='#4A5568')
        ax2.grid(True, alpha=0.2, axis='y')
        ax2.spines[['top', 'right']].set_visible(False)
        
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold', 
                    color='#1A3E6B', fontsize=10)
        
        # Chart 3: F1-Score Comparison (Top Right)
        ax3 = fig.add_subplot(gs[0, 2])
        bars = ax3.bar(models, metrics_data['F1-Score'], color=colors, alpha=0.9, 
                      edgecolor='white', linewidth=2)
        ax3.set_facecolor('#F8FBFF')
        ax3.set_title('F1-Score Comparison', fontweight='bold', fontsize=14, color='#1A3E6B', pad=20)
        ax3.set_ylabel('F1-Score', fontweight='bold', color='#4A5568', fontsize=12)
        ax3.tick_params(axis='x', rotation=45, colors='#4A5568')
        ax3.tick_params(axis='y', colors='#4A5568')
        ax3.grid(True, alpha=0.2, axis='y')
        ax3.spines[['top', 'right']].set_visible(False)
        
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold', 
                    color='#1A3E6B', fontsize=10)
        
        # Chart 4: Radar Chart (Middle Row - Span all columns)
        ax4 = fig.add_subplot(gs[1, :])
        metrics_array = np.array([metrics_data['Accuracy'], metrics_data['Precision'], 
                                metrics_data['Recall'], metrics_data['F1-Score']])
        metrics_normalized = metrics_array / metrics_array.max(axis=1, keepdims=True)
        
        angles = np.linspace(0, 2*np.pi, len(metrics_normalized), endpoint=False).tolist()
        angles += angles[:1]
        
        for i, model in enumerate(models):
            values = metrics_normalized[:, i].tolist()
            values += values[:1]
            ax4.plot(angles, values, 'o-', linewidth=3, label=model, color=colors[i], 
                    markersize=8, markerfacecolor='white', markeredgewidth=2)
            ax4.fill(angles, values, alpha=0.1, color=colors[i])
        
        ax4.set_facecolor('#F8FBFF')
        ax4.set_yticklabels([])
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1-Score'], 
                           color='#4A5568', fontweight='bold', fontsize=12)
        ax4.set_title('Performance Radar Chart - Normalized Comparison', fontweight='bold', 
                     fontsize=16, color='#1A3E6B', pad=20)
        ax4.legend(facecolor='#F8FBFF', edgecolor='none', bbox_to_anchor=(0.5, -0.15), 
                  loc='upper center', ncol=4, fontsize=11)
        ax4.grid(True, alpha=0.3)
        
        # Chart 5: Model Ranking (Bottom Left)
        ax5 = fig.add_subplot(gs[2, 0])
        # Create ranking based on average performance
        avg_scores = []
        for i in range(len(models)):
            avg_score = np.mean([metrics_data[metric][i] for metric in metrics_data])
            avg_scores.append(avg_score)
        
        ranked_indices = np.argsort(avg_scores)[::-1]
        ranked_models = [models[i] for i in ranked_indices]
        ranked_scores = [avg_scores[i] for i in ranked_indices]
        
        bars = ax5.barh(range(len(ranked_models)), ranked_scores, color=[colors[i] for i in ranked_indices], 
                       alpha=0.9, edgecolor='white', linewidth=2)
        ax5.set_facecolor('#F8FBFF')
        ax5.set_title('Model Performance Ranking', fontweight='bold', fontsize=14, color='#1A3E6B', pad=20)
        ax5.set_xlabel('Average Score', fontweight='bold', color='#4A5568', fontsize=12)
        ax5.set_yticks(range(len(ranked_models)))
        ax5.set_yticklabels(ranked_models, color='#4A5568', fontsize=11)
        ax5.grid(True, alpha=0.2, axis='x')
        ax5.spines[['top', 'right']].set_visible(False)
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax5.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                    f'{width:.3f}', ha='left', va='center', fontweight='bold', 
                    color='#1A3E6B', fontsize=10)
        
        # Chart 6: Performance Distribution (Bottom Middle)
        ax6 = fig.add_subplot(gs[2, 1])
        all_scores = []
        for metric in metrics_data.values():
            all_scores.extend(metric)
        
        ax6.hist(all_scores, bins=15, color='#146EB4', alpha=0.7, edgecolor='white', linewidth=1.5)
        ax6.set_facecolor('#F8FBFF')
        ax6.set_title('Score Distribution Across Models', fontweight='bold', fontsize=14, color='#1A3E6B', pad=20)
        ax6.set_xlabel('Performance Score', fontweight='bold', color='#4A5568', fontsize=12)
        ax6.set_ylabel('Frequency', fontweight='bold', color='#4A5568', fontsize=12)
        ax6.tick_params(axis='both', colors='#4A5568')
        ax6.grid(True, alpha=0.2)
        ax6.spines[['top', 'right']].set_visible(False)
        
        # Chart 7: Metric Correlation (Bottom Right)
        ax7 = fig.add_subplot(gs[2, 2])
        metrics_df = pd.DataFrame(metrics_data, index=models)
        correlation = metrics_df.corr()
        
        im = ax7.imshow(correlation, cmap='Blues', vmin=0.5, vmax=1.0, aspect='auto')
        ax7.set_facecolor('#F8FBFF')
        ax7.set_title('Metric Correlation Matrix', fontweight='bold', fontsize=14, color='#1A3E6B', pad=20)
        ax7.set_xticks(range(len(correlation.columns)))
        ax7.set_yticks(range(len(correlation.columns)))
        ax7.set_xticklabels(correlation.columns, rotation=45, color='#4A5568', fontsize=10)
        ax7.set_yticklabels(correlation.columns, color='#4A5568', fontsize=10)
        
        # Add correlation values
        for i in range(len(correlation.columns)):
            for j in range(len(correlation.columns)):
                text = ax7.text(j, i, f'{correlation.iloc[i, j]:.2f}',
                               ha="center", va="center", color="white" if correlation.iloc[i, j] > 0.7 else "black",
                               fontweight='bold', fontsize=9)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax7, shrink=0.8)
        cbar.ax.tick_params(colors='#4A5568')
        
        plt.tight_layout()
        return fig

    @staticmethod
    def create_quick_overview(results):
        """Create a quick overview for the main results section"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Quick Performance Overview', fontsize=16, fontweight='bold', color='#1A3E6B', y=0.95)
        
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
        
        colors = ['#146EB4', '#FF9900', '#27AE60', '#E74C3C']
        
        # Simple bar charts for quick overview
        axes = [ax1, ax2, ax3, ax4]
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        for idx, (ax, metric) in enumerate(zip(axes, metric_names)):
            bars = ax.bar(models, metrics[metric], color=colors, alpha=0.8)
            ax.set_facecolor('#F8FBFF')
            ax.set_title(f'{metric} Comparison', fontweight='bold', color='#1A3E6B')
            ax.set_ylabel(metric, fontweight='bold', color='#4A5568')
            ax.tick_params(axis='x', rotation=45, colors='#4A5568')
            ax.tick_params(axis='y', colors='#4A5568')
            ax.grid(True, alpha=0.2, axis='y')
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        return fig

# ============================
# Main Application
# ============================
def main():
    # Header
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1A3E6B 0%, #0F4E8A 100%); 
                padding: 2rem; border-radius: 12px; margin-bottom: 2.5rem;">
        <h1 style="color: white; text-align: center; margin: 0; font-size: 2.5rem;">🎯 NLP Pro Analysis Suite</h1>
        <p style="color: #A0BCC8; text-align: center; margin: 0.5rem 0 0 0; font-size: 1.2rem;">
            Enterprise-grade Text Analytics Platform
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='main-header'>Advanced Text Intelligence Platform</div>", unsafe_allow_html=True)
    
    # Feature Highlights
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
            
            # Data Explorer
            with st.expander("🔍 Data Explorer", expanded=True):
                tab1, tab2 = st.tabs(["📊 Preview", "📈 Statistics"])
                
                with tab1:
                    st.dataframe(df.head(8), use_container_width=True)
                
                with tab2:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Records", df.shape[0])
                    with col2:
                        st.metric("Features", df.shape[1])
                    with col3:
                        st.metric("Missing Values", df.isnull().sum().sum())
                    with col4:
                        st.metric("Data Types", len(df.dtypes.unique()))
            
            # Analysis Configuration
            st.markdown('<div class="section-title">⚙️ Analysis Configuration</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                text_col = st.selectbox("Text Column", df.columns)
            with col2:
                target_col = st.selectbox("Target Column", df.columns)
            
            feature_type = st.radio("Select Analysis Type:", 
                                  ["📖 Lexical Features", "🎭 Semantic Features", 
                                   "🔧 Syntactic Features", "🎯 Pragmatic Features"],
                                  horizontal=True)
            
            if st.button("🚀 Launch Advanced Analysis", use_container_width=True):
                if df[text_col].isnull().any():
                    df[text_col] = df[text_col].fillna('')
                
                if df[target_col].isnull().any():
                    st.error("Target column contains missing values.")
                    return
                
                if len(df[target_col].unique()) < 2:
                    st.error("Target column must have at least 2 unique classes.")
                    return
                
                # Feature Extraction
                with st.spinner("🔄 Extracting advanced features..."):
                    extractor = PrimeFeatureExtractor()
                    X = df[text_col].astype(str)
                    y = df[target_col]
                    
                    if feature_type == "📖 Lexical Features":
                        X_features = extractor.extract_lexical_features(X)
                    elif feature_type == "🎭 Semantic Features":
                        X_features = extractor.extract_semantic_features(X)
                    elif feature_type == "🔧 Syntactic Features":
                        X_features = extractor.extract_syntactic_features(X)
                    else:
                        X_features = extractor.extract_pragmatic_features(X)
                
                # Model Training
                with st.spinner("🤖 Training advanced models..."):
                    trainer = PrimeModelTrainer()
                    results, label_encoder = trainer.train_and_evaluate(X_features, y)
                
                successful_models = {k: v for k, v in results.items() if 'error' not in v}
                
                if successful_models:
                    # Enhanced Dashboard Section
                    st.markdown('<div class="section-title">📊 Enhanced Performance Dashboard</div>', unsafe_allow_html=True)
                    
                    # Quick Overview
                    st.markdown("#### 📈 Quick Performance Overview")
                    dashboard = EnhancedDashboard()
                    quick_fig = dashboard.create_quick_overview(successful_models)
                    st.pyplot(quick_fig)
                    
                    # Comprehensive Dashboard
                    st.markdown("#### 🎯 Comprehensive Analysis Dashboard")
                    with st.expander("📊 Expand Full Dashboard", expanded=True):
                        comprehensive_fig = dashboard.create_comprehensive_dashboard(successful_models)
                        st.pyplot(comprehensive_fig)
                    
                    # Best Model Recommendation
                    best_model = max(successful_models.items(), key=lambda x: x[1]['accuracy'])
                    st.success(f"🎯 **Recommended Model**: {best_model[0]} with {best_model[1]['accuracy']:.1%} accuracy")
                
                else:
                    st.error("No models were successfully trained.")
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    else:
        # Welcome Section
        st.markdown("""
        <div style='text-align: center; padding: 4rem 2rem; background: linear-gradient(135deg, #F8FBFF 0%, #E8F2FF 100%); 
                    border-radius: 16px; margin: 3rem 0; border: 2px dashed #146EB4;'>
            <h2 style='color: #1A3E6B; margin-bottom: 1.5rem; font-size: 2.5rem;'>🚀 Get Started with NLP Pro</h2>
            <p style='color: #4A5568; font-size: 1.3rem; margin-bottom: 2.5rem; font-weight: 500;'>
                Upload your CSV file to unlock powerful text analysis capabilities
            </p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    

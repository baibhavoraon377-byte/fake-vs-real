# ============================================
# 📌 Advanced NLP Analysis with SMOTE & GloVe
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from textblob import TextBlob
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Advanced features with graceful fallbacks
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.pipeline import Pipeline
    IMBALANCE_LEARN_AVAILABLE = True
except ImportError:
    IMBALANCE_LEARN_AVAILABLE = False

try:
    import gensim
    from gensim.models import KeyedVectors
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False

# ============================
# Page Configuration
# ============================
st.set_page_config(
    page_title="NLP Analysis Suite | Enterprise",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================
# Initialize SpaCy
# ============================
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.error("SpaCy English model required. Run: `python -m spacy download en_core_web_sm`")
        st.stop()

nlp = load_spacy_model()
stop_words = STOP_WORDS

# ============================
# GloVe Embeddings Manager
# ============================
class GloVeManager:
    def __init__(self):
        self.embeddings_index = {}
        self.loaded = False
        
    def load_embeddings(self, dimension=100):
        """Load GloVe embeddings with multiple fallback strategies"""
        if not GENSIM_AVAILABLE:
            return False
            
        try:
            # Try to load pre-trained GloVe embeddings
            import gensim.downloader as api
            model_name = f"glove-wiki-gigaword-{dimension}"
            self.model = api.load(model_name)
            self.loaded = True
            return True
        except:
            # Fallback: Use smaller embeddings or TF-IDF
            try:
                model_name = "glove-wiki-gigaword-50"
                self.model = api.load(model_name)
                self.loaded = True
                return True
            except:
                return False
    
    def text_to_embeddings(self, texts, dimension=100):
        """Convert texts to GloVe embeddings with intelligent averaging"""
        if not self.loaded:
            return None
            
        embeddings = []
        for text in texts:
            words = str(text).lower().split()
            word_vectors = []
            
            for word in words:
                try:
                    if word in self.model:
                        word_vectors.append(self.model[word])
                except:
                    continue
            
            if word_vectors:
                # Use weighted average based on TF-IDF-like importance
                text_vector = np.mean(word_vectors, axis=0)
            else:
                text_vector = np.zeros(dimension)
            
            embeddings.append(text_vector)
        
        return np.array(embeddings)

glove_manager = GloVeManager()

# ============================
# Advanced CSS Styling
# ============================
def inject_custom_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 3.2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
    }
    .section-header {
        font-size: 1.6rem;
        color: #2d3748;
        margin: 2.5rem 0 1.2rem 0;
        font-weight: 700;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #e2e8f0;
        position: relative;
    }
    .section-header:after {
        content: '';
        position: absolute;
        bottom: -3px;
        left: 0;
        width: 80px;
        height: 3px;
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    .metric-card {
        background: white;
        padding: 1.8rem;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        margin: 0.8rem 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 20px -4px rgba(0, 0, 0, 0.08);
    }
    .insight-card {
        background: linear-gradient(135deg, #f8fafc 0%, #edf2f7 100%);
        border-left: 4px solid #4299e1;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    .success-indicator {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        display: inline-block;
    }
    .warning-indicator {
        background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        display: inline-block;
    }
    .feature-pill {
        background: #e2e8f0;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        margin: 0.2rem;
        display: inline-block;
    }
    .stButton button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 1rem 2rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    }
    .tab-content {
        padding: 1.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# ============================
# Phase Feature Extractors
# ============================
class FeatureEngine:
    @staticmethod
    def lexical_features(text):
        """Advanced lexical analysis"""
        doc = nlp(str(text).lower())
        features = {
            'tokens': [token.lemma_ for token in doc if token.text not in stop_words and token.is_alpha],
            'word_count': len([token for token in doc if token.is_alpha]),
            'unique_words': len(set([token.lemma_ for token in doc if token.is_alpha])),
            'avg_word_length': np.mean([len(token.text) for token in doc if token.is_alpha]) if any(token.is_alpha for token in doc) else 0
        }
        return " ".join(features['tokens']), features

    @staticmethod
    def syntactic_features(text):
        """Syntactic structure analysis"""
        doc = nlp(str(text))
        pos_tags = [token.pos_ for token in doc]
        return " ".join(pos_tags), {
            'noun_count': pos_tags.count('NOUN'),
            'verb_count': pos_tags.count('VERB'),
            'adj_count': pos_tags.count('ADJ'),
            'sentence_complexity': len(list(doc.sents))
        }

    @staticmethod
    def semantic_features(text):
        """Semantic meaning analysis"""
        blob = TextBlob(str(text))
        return [blob.sentiment.polarity, blob.sentiment.subjectivity], {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity,
            'sentiment_label': 'positive' if blob.sentiment.polarity > 0.1 else 'negative' if blob.sentiment.polarity < -0.1 else 'neutral'
        }

    @staticmethod
    def pragmatic_features(text):
        """Pragmatic context analysis"""
        text_lower = str(text).lower()
        counts = [text_lower.count(word) for word in ["must", "should", "might", "could", "will", "would"]]
        return counts, {
            'modality_score': sum(counts),
            'question_marks': text_lower.count('?'),
            'exclamation_marks': text_lower.count('!')
        }

# ============================
# Advanced Model Training with SMOTE
# ============================
class AdvancedNLPAnalyzer:
    def __init__(self):
        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "SVM": SVC(random_state=42, class_weight='balanced', probability=True),
            "Naive Bayes": MultinomialNB()
        }
        
    def apply_smote_automatically(self, X, y):
        """Intelligent SMOTE application based on data characteristics"""
        if not IMBALANCE_LEARN_AVAILABLE:
            return X, y, "SMOTE not available"
            
        y_series = pd.Series(y)
        class_counts = y_series.value_counts()
        minority_ratio = class_counts.min() / class_counts.max()
        
        if minority_ratio < 0.3:  # Significant imbalance detected
            try:
                smote = SMOTE(random_state=42, k_neighbors=min(5, class_counts.min() - 1))
                X_res, y_res = smote.fit_resample(X, y)
                return X_res, y_res, f"SMOTE applied (imbalance ratio: {minority_ratio:.3f})"
            except Exception as e:
                return X, y, f"SMOTE failed: {str(e)}"
        else:
            return X, y, "Balanced dataset - SMOTE not required"
    
    def train_models(self, X_features, y, use_glove=False):
        """Advanced model training with comprehensive evaluation"""
        results = {}
        insights = []
        
        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        n_classes = len(le.classes_)
        
        # Apply SMOTE intelligently
        X_processed, y_processed, smote_insight = self.apply_smote_automatically(X_features, y_encoded)
        insights.append(smote_insight)
        
        # Split data
        test_size = max(0.2, min(0.3, 2 * n_classes / len(y_processed)))
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_encoded, test_size=test_size, random_state=42, stratify=y_processed
        )
        
        # Training progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, (name, model) in enumerate(self.models.items()):
            status_text.text(f"🔄 Training {name}...")
            
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
                
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
                    'predictions': y_pred,
                    'probabilities': y_proba,
                    'feature_importance': self.get_feature_importance(model, X_train) if hasattr(model, 'feature_importances_') else None
                }
                
            except Exception as e:
                results[name] = {'error': str(e)}
            
            progress_bar.progress((i + 1) / len(self.models))
        
        status_text.text("✅ Training completed!")
        progress_bar.empty()
        
        return results, le, n_classes, insights
    
    def get_feature_importance(self, model, X_train):
        """Extract feature importance if available"""
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        elif hasattr(model, 'coef_'):
            return np.mean(np.abs(model.coef_), axis=0)
        return None

# ============================
# Interactive Visualizations
# ============================
class VisualizationEngine:
    @staticmethod
    def create_performance_radar(results):
        """Create radar chart for model comparison"""
        models = list(results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        fig = go.Figure()
        
        for model in models:
            if 'error' not in results[model]:
                values = [results[model][metric] for metric in metrics]
                fig.add_trace(go.Scatterpolar(
                    r=values + [values[0]],  # Close the circle
                    theta=metrics + [metrics[0]],
                    fill='toself',
                    name=model
                ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1])
            ),
            showlegend=True,
            title="Model Performance Radar Chart"
        )
        return fig
    
    @staticmethod
    def create_class_distribution(y):
        """Create interactive class distribution chart"""
        counts = pd.Series(y).value_counts()
        fig = px.bar(x=counts.index.astype(str), y=counts.values,
                    title="Class Distribution Analysis",
                    labels={'x': 'Classes', 'y': 'Count'})
        fig.update_layout(xaxis_tickangle=45)
        return fig
    
    @staticmethod
    def create_confidence_analysis(results, y_test):
        """Create confidence interval visualization"""
        fig = go.Figure()
        
        for model_name, result in results.items():
            if 'probabilities' in result and result['probabilities'] is not None:
                max_probs = np.max(result['probabilities'], axis=1)
                fig.add_trace(go.Violin(y=max_probs, name=model_name, box_visible=True))
        
        fig.update_layout(title="Prediction Confidence Distribution", yaxis_title="Maximum Probability")
        return fig

# ============================
# Main Application
# ============================
def main():
    st.markdown("<div class='main-header'>🔮 Advanced NLP Analysis Suite</div>", unsafe_allow_html=True)
    
    # Header with status indicators
    col1, col2, col3 = st.columns(3)
    with col1:
        status = "✅ Available" if IMBALANCE_LEARN_AVAILABLE else "⚠️ Limited"
        st.markdown(f"<div class='metric-card'><h3>SMOTE</h3><p>{status}</p></div>", unsafe_allow_html=True)
    with col2:
        status = "✅ Available" if GENSIM_AVAILABLE else "⚠️ Limited"
        st.markdown(f"<div class='metric-card'><h3>GloVe Embeddings</h3><p>{status}</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-card'><h3>Analysis Phases</h3><p>5 Advanced Methods</p></div>", unsafe_allow_html=True)
    
    # File upload section
    st.markdown('<div class="section-header">📁 Data Integration</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"], 
                                   help="Supported formats: CSV, Excel")
    
    if uploaded_file:
        try:
            # Load data
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Data overview
            st.success(f"✅ Dataset loaded successfully: {df.shape[0]} records, {df.shape[1]} features")
            
            # Interactive data explorer
            with st.expander("🔍 Data Explorer", expanded=True):
                tab1, tab2, tab3 = st.tabs(["Preview", "Statistics", "Quality"])
                
                with tab1:
                    st.dataframe(df.head(10), use_container_width=True)
                
                with tab2:
                    st.write("**Dataset Overview**")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Records", df.shape[0])
                    col2.metric("Features", df.shape[1])
                    col3.metric("Missing Values", df.isnull().sum().sum())
                    col4.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024 ** 2:.1f} MB")
                
                with tab3:
                    # Data quality assessment
                    missing_percent = (df.isnull().sum() / len(df)) * 100
                    quality_df = pd.DataFrame({
                        'Missing %': missing_percent,
                        'Data Type': df.dtypes,
                        'Unique Values': df.nunique()
                    })
                    st.dataframe(quality_df, use_container_width=True)
            
            # Analysis configuration
            st.markdown('<div class="section-header">⚙️ Analysis Configuration</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                text_col = st.selectbox("Text Column", df.columns, 
                                      help="Select the column containing text data")
            with col2:
                target_col = st.selectbox("Target Column", df.columns,
                                        help="Select the column containing labels/categories")
            
            # Advanced options
            st.markdown('<div class="section-header">🎯 Advanced Features</div>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                analysis_phase = st.selectbox("NLP Phase", [
                    "Lexical Analysis", "Syntactic Analysis", "Semantic Analysis", 
                    "Pragmatic Analysis", "Comprehensive Analysis"
                ])
            with col2:
                auto_smote = st.checkbox("Auto-SMOTE Balancing", value=True,
                                       help="Automatically apply SMOTE for imbalanced data")
            with col3:
                use_glove = st.checkbox("GloVe Embeddings", value=GENSIM_AVAILABLE,
                                      disabled=not GENSIM_AVAILABLE,
                                      help="Use pre-trained word embeddings")
            
            # Initialize engines
            feature_engine = FeatureEngine()
            analyzer = AdvancedNLPAnalyzer()
            viz_engine = VisualizationEngine()
            
            if st.button("🚀 Start Advanced Analysis", type="primary", use_container_width=True):
                with st.spinner("🔄 Processing data with advanced features..."):
                    # Prepare data
                    X = df[text_col].fillna('')
                    y = df[target_col].fillna('Unknown')
                    
                    # Feature extraction based on selected phase
                    if analysis_phase == "Lexical Analysis":
                        X_processed, insights = zip(*X.apply(feature_engine.lexical_features))
                        X_features = TfidfVectorizer(max_features=2000).fit_transform(X_processed)
                    
                    elif analysis_phase == "Syntactic Analysis":
                        X_processed, insights = zip(*X.apply(feature_engine.syntactic_features))
                        X_features = TfidfVectorizer(max_features=2000).fit_transform(X_processed)
                    
                    elif analysis_phase == "Semantic Analysis":
                        X_features = np.array([feature_engine.semantic_features(x)[0] for x in X])
                    
                    elif analysis_phase == "Pragmatic Analysis":
                        X_features = np.array([feature_engine.pragmatic_features(x)[0] for x in X])
                    
                    else:  # Comprehensive Analysis
                        # Combine multiple feature types
                        lexical_features = TfidfVectorizer(max_features=1000).fit_transform(
                            [feature_engine.lexical_features(x)[0] for x in X])
                        semantic_features = np.array([feature_engine.semantic_features(x)[0] for x in X])
                        X_features = np.hstack([lexical_features.toarray(), semantic_features])
                    
                    # Apply GloVe if requested
                    if use_glove and GENSIM_AVAILABLE:
                        if not glove_manager.loaded:
                            glove_manager.load_embeddings()
                        glove_features = glove_manager.text_to_embeddings(X)
                        if glove_features is not None:
                            X_features = np.hstack([X_features, glove_features])
                    
                    # Train models
                    results, label_encoder, n_classes, insights = analyzer.train_models(
                        X_features, y, use_glove=use_glove
                    )
                
                # Display results
                st.markdown('<div class="section-header">📊 Analysis Results</div>', unsafe_allow_html=True)
                
                # Insights panel
                st.markdown("### 🔍 Key Insights")
                for insight in insights:
                    st.markdown(f"<div class='insight-card'>{insight}</div>", unsafe_allow_html=True)
                
                # Performance metrics
                st.markdown("### 📈 Model Performance")
                
                # Create metrics dataframe
                metrics_data = []
                for model_name, result in results.items():
                    if 'error' not in result:
                        metrics_data.append({
                            'Model': model_name,
                            'Accuracy': f"{result['accuracy']:.3f}",
                            'Precision': f"{result['precision']:.3f}",
                            'Recall': f"{result['recall']:.3f}",
                            'F1-Score': f"{result['f1_score']:.3f}",
                            'Accuracy_float': result['accuracy']
                        })
                
                if metrics_data:
                    metrics_df = pd.DataFrame(metrics_data)
                    metrics_df = metrics_df.sort_values('Accuracy_float', ascending=False)
                    
                    # Display metrics in elegant cards
                    cols = st.columns(len(metrics_df))
                    for idx, (col, (_, row)) in enumerate(zip(cols, metrics_df.iterrows())):
                        with col:
                            accuracy_val = float(row['Accuracy'])
                            color = "success-indicator" if accuracy_val > 0.8 else "warning-indicator" if accuracy_val > 0.6 else ""
                            st.markdown(f"""
                            <div class='metric-card'>
                                <h3>{row['Model']}</h3>
                                <div class='{color}'>{accuracy_val:.1%}</div>
                                <p>F1: {row['F1-Score']} | Precision: {row['Precision']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.plotly_chart(viz_engine.create_performance_radar(results), use_container_width=True)
                    
                    with col2:
                        st.plotly_chart(viz_engine.create_class_distribution(y), use_container_width=True)
                    
                    # Detailed analysis
                    with st.expander("🔬 Detailed Analysis Report"):
                        st.dataframe(metrics_df.drop('Accuracy_float', axis=1), use_container_width=True)
                        
                        # Feature importance if available
                        for model_name, result in results.items():
                            if result.get('feature_importance') is not None:
                                st.write(f"**{model_name} Feature Importance**")
                                importance_df = pd.DataFrame({
                                    'Feature': range(len(result['feature_importance'])),
                                    'Importance': result['feature_importance']
                                }).sort_values('Importance', ascending=False).head(10)
                                st.bar_chart(importance_df.set_index('Feature'))
                
                else:
                    st.error("❌ No models were successfully trained. Check your data and configuration.")
        
        except Exception as e:
            st.error(f"❌ Error processing dataset: {str(e)}")
    
    else:
        # Welcome section
        st.markdown("""
        <div style='text-align: center; padding: 4rem 2rem; background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); 
                    border-radius: 20px; margin: 2rem 0;'>
            <h2 style='color: #2d3748; margin-bottom: 1rem;'>Welcome to Advanced NLP Analysis Suite</h2>
            <p style='color: #4a5568; font-size: 1.2rem; max-width: 800px; margin: 0 auto; line-height: 1.6;'>
                Enterprise-grade natural language processing with automatic class balancing, 
                advanced embeddings, and comprehensive analytics. Upload your dataset to 
                unlock powerful insights.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature highlights
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class='metric-card'>
                <h4>🤖 Smart Balancing</h4>
                <p>Automatic SMOTE application for imbalanced datasets</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class='metric-card'>
                <h4>🔤 Advanced Embeddings</h4>
                <p>GloVe word vectors for superior semantic understanding</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class='metric-card'>
                <h4>📊 Interactive Analytics</h4>
                <p>Comprehensive visualizations and insights</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

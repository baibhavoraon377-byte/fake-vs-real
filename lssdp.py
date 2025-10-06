# ============================================
# TextInsight - AI-Powered Fact Analytics
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from textblob import TextBlob
import re
from collections import Counter

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================
# Page Configuration
# ============================
st.set_page_config(
    page_title="TextInsight - AI Fact Analytics",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================
# Professional Style CSS
# ============================
st.markdown("""
<style>
    .professional-header {
        background: linear-gradient(135deg, #334155 0%, #1E293B 100%);
        border-radius: 16px;
        padding: 3rem 2rem;
        margin: 1rem 0 3rem 0;
        text-align: center;
        border: 1px solid #475569;
        position: relative;
        overflow: hidden;
    }
    .header-badge {
        background: #2563EB;
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: 700;
        display: inline-block;
        margin-bottom: 1rem;
    }
    .page-header {
        font-size: 3rem;
        font-weight: 800;
        color: #F8FAFC;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #F8FAFC 0%, #2563EB 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .professional-card {
        background: #334155;
        border-radius: 12px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid #475569;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .metric-card {
        background: #334155;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        border: 1px solid #475569;
        transition: all 0.3s ease;
        margin: 0.5rem;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #2563EB;
        margin-bottom: 0.5rem;
    }
    .metric-label {
        font-size: 14px;
        color: #94A3B8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .model-card {
        background: #334155;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        border: 1px solid #475569;
        transition: all 0.3s ease;
        margin: 0.5rem;
    }
    .model-accuracy {
        font-size: 2.2rem;
        font-weight: 800;
        color: #2563EB;
        margin: 1rem 0;
    }
    .progress-container {
        background: #475569;
        border-radius: 10px;
        height: 6px;
        overflow: hidden;
        margin: 1rem 0;
    }
    .progress-fill {
        background: linear-gradient(90deg, #2563EB, #3B82F6);
        height: 100%;
        border-radius: 10px;
        transition: width 0.3s ease;
    }
    .section-header {
        font-size: 2rem;
        font-weight: 700;
        color: #F8FAFC;
        margin: 3rem 0 1.5rem 0;
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
# Enhanced Feature Engineering Classes
# ============================
class AdvancedFeatureExtractor:
    @staticmethod
    def preprocess_text(texts):
        """Advanced text preprocessing with multiple cleaning steps"""
        processed_texts = []
        for text in texts:
            # Convert to string and lowercase
            text = str(text).lower()
            
            # Remove special characters and digits
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            
            # Advanced tokenization with spaCy
            doc = nlp(text)
            
            # Lemmatization with POS filtering
            tokens = []
            for token in doc:
                if (token.text not in stop_words and 
                    token.is_alpha and 
                    len(token.text) > 2 and
                    token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']):
                    tokens.append(token.lemma_)
            
            processed_texts.append(" ".join(tokens))
        return processed_texts

    @staticmethod
    def extract_lexical_features(texts):
        """Enhanced lexical features with multiple vectorization techniques"""
        processed_texts = AdvancedFeatureExtractor.preprocess_text(texts)
        
        # TF-IDF with optimal parameters
        tfidf_vectorizer = TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            strip_accents='unicode'
        )
        
        tfidf_features = tfidf_vectorizer.fit_transform(processed_texts)
        return tfidf_features

    @staticmethod
    def extract_semantic_features(texts):
        """Enhanced semantic features with comprehensive analysis"""
        features = []
        for text in texts:
            blob = TextBlob(str(text))
            
            # Sentiment features
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Readability features
            words = text.split()
            sentences = text.split('.')
            word_count = len(words)
            sentence_count = len([s for s in sentences if s.strip()])
            avg_sentence_length = word_count / max(sentence_count, 1)
            
            # Vocabulary richness
            unique_words = len(set(words))
            lexical_diversity = unique_words / max(word_count, 1)
            
            # Advanced features
            long_words = len([word for word in words if len(word) > 6])
            complex_word_ratio = long_words / max(word_count, 1)
            
            features.append([
                polarity,
                subjectivity,
                word_count,
                sentence_count,
                avg_sentence_length,
                lexical_diversity,
                complex_word_ratio,
                unique_words
            ])
        return np.array(features)

    @staticmethod
    def extract_syntactic_features(texts):
        """Enhanced syntactic features with POS patterns"""
        processed_texts = []
        for text in texts:
            doc = nlp(str(text))
            
            # POS pattern features
            pos_patterns = []
            
            for token in doc:
                if token.is_alpha and not token.is_stop:
                    pos_tag = f"{token.pos_}"
                    pos_patterns.append(pos_tag)
            
            # Convert to feature string
            pos_feature_string = " ".join(pos_patterns)
            processed_texts.append(pos_feature_string)
        
        # Vectorize POS patterns
        pos_vectorizer = CountVectorizer(
            max_features=1000,
            ngram_range=(1, 4),
            analyzer='word'
        )
        return pos_vectorizer.fit_transform(processed_texts)

    @staticmethod
    def extract_pragmatic_features(texts):
        """Enhanced pragmatic features for fact analysis"""
        pragmatic_features = []
        
        # Fact-oriented indicators
        fact_indicators = {
            'evidence': ['study', 'research', 'data', 'evidence', 'statistics', 'survey', 'report'],
            'certainty': ['proven', 'confirmed', 'verified', 'established', 'demonstrated'],
            'quantification': ['percent', 'percentage', 'majority', 'minority', 'average', 'median'],
            'temporal': ['recent', 'current', 'latest', 'annual', 'quarterly', 'monthly'],
            'source': ['according', 'source', 'reference', 'cited', 'journal', 'university']
        }

        for text in texts:
            text_lower = str(text).lower()
            features = []

            # Indicator counts
            for category, words in fact_indicators.items():
                count = sum(text_lower.count(word) for word in words)
                features.append(count)

            # Structural features
            features.extend([
                text.count('!'),  # Emphasis
                text.count('?'),  # Questions
                len([s for s in text.split('.') if s.strip()]),  # Sentences
                len([w for w in text.split() if w.istitle()]),  # Proper nouns
                text.count('%'),  # Percentages
                text.count('$'),  # Currency
                len(re.findall(r'\d+', text)),  # Numbers
            ])

            pragmatic_features.append(features)

        return np.array(pragmatic_features)

    @staticmethod
    def extract_hybrid_features(texts):
        """Combine all feature types for maximum performance"""
        lexical = AdvancedFeatureExtractor.extract_lexical_features(texts)
        semantic = AdvancedFeatureExtractor.extract_semantic_features(texts)
        syntactic = AdvancedFeatureExtractor.extract_syntactic_features(texts)
        pragmatic = AdvancedFeatureExtractor.extract_pragmatic_features(texts)
        
        # Convert all to dense arrays if needed and combine
        from scipy.sparse import hstack
        
        # Handle sparse matrices
        if hasattr(lexical, 'toarray'):
            lexical = lexical.toarray()
        if hasattr(syntactic, 'toarray'):
            syntactic = syntactic.toarray()
            
        # Combine features
        hybrid_features = np.hstack([lexical, semantic, syntactic, pragmatic])
        
        # Dimensionality reduction for better performance
        svd = TruncatedSVD(n_components=min(500, hybrid_features.shape[1]), random_state=42)
        hybrid_features_reduced = svd.fit_transform(hybrid_features)
        
        return hybrid_features_reduced

# ============================
# Enhanced Model Trainer with SMOTE
# ============================
class AdvancedModelTrainer:
    def __init__(self):
        self.models = {
            "Logistic Regression": LogisticRegression(
                max_iter=2000, 
                random_state=42, 
                class_weight='balanced',
                C=1.0,
                solver='liblinear'
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight='balanced',
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2
            ),
            "Support Vector Machine": SVC(
                random_state=42, 
                probability=True, 
                class_weight='balanced',
                C=1.0,
                kernel='rbf',
                gamma='scale'
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=150,
                random_state=42,
                max_depth=6,
                learning_rate=0.1,
                min_samples_split=5
            ),
            "Naive Bayes": MultinomialNB(alpha=0.1)
        }

    def enhanced_cross_validation(self, X, y, model, cv_folds=5):
        """Perform enhanced cross-validation"""
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
        return cv_scores.mean(), cv_scores.std()

    def apply_smote(self, X, y, sampling_strategy='auto', random_state=42):
        """Apply SMOTE for handling class imbalance"""
        try:
            # Check if SMOTE can be applied (need at least 2 samples per class)
            unique_classes, class_counts = np.unique(y, return_counts=True)
            min_samples = min(class_counts)
            
            if min_samples < 2:
                st.warning("SMOTE cannot be applied: Some classes have less than 2 samples")
                return X, y, False
                
            # Adjust sampling strategy based on class distribution
            if len(unique_classes) == 2:
                # For binary classification, balance the classes
                sampling_strategy = 'auto'
            else:
                # For multi-class, don't over-sample beyond the majority class
                sampling_strategy = 'not majority'
            
            smote = SMOTE(
                sampling_strategy=sampling_strategy,
                random_state=random_state,
                k_neighbors=min(5, min_samples - 1)
            )
            
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            # Log the resampling results
            original_dist = np.bincount(y)
            resampled_dist = np.bincount(y_resampled)
            
            st.success(f"SMOTE Applied: {len(X)} → {len(X_resampled)} samples")
            st.info(f"Class distribution before: {dict(zip(unique_classes, original_dist))}")
            st.info(f"Class distribution after: {dict(zip(unique_classes, resampled_dist))}")
            
            return X_resampled, y_resampled, True
            
        except Exception as e:
            st.warning(f"SMOTE failed: {str(e)}. Using original data.")
            return X, y, False

    def train_and_evaluate(self, X, y, use_smote=True):
        """Enhanced model training with SMOTE and comprehensive evaluation"""
        results = {}

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        n_classes = len(le.classes_)

        # Dynamic test size based on dataset characteristics
        if len(y_encoded) < 1000:
            test_size = 0.2
        elif len(y_encoded) < 5000:
            test_size = 0.15
        else:
            test_size = 0.1

        # Enhanced stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, 
            test_size=test_size, 
            random_state=42, 
            stratify=y_encoded
        )

        # Apply SMOTE to training data if requested and applicable
        smote_applied = False
        if use_smote and n_classes > 1:
            X_train_resampled, y_train_resampled, smote_applied = self.apply_smote(X_train, y_train)
            if smote_applied:
                X_train = X_train_resampled
                y_train = y_train_resampled

        progress_bar = st.progress(0)
        total_models = len(self.models)

        for i, (name, model) in enumerate(self.models.items()):
            try:
                # Enhanced model training
                if name in ["Logistic Regression", "Random Forest"] and len(y_encoded) > 100:
                    with st.spinner(f"Optimizing {name}..."):
                        if name == "Logistic Regression":
                            param_grid = {'C': [0.1, 1.0, 10.0]}
                        else:
                            param_grid = {'n_estimators': [100, 200], 'max_depth': [5, 10]}
                        
                        grid_search = GridSearchCV(
                            model, 
                            param_grid, 
                            cv=5, 
                            scoring='accuracy',
                            n_jobs=-1
                        )
                        grid_search.fit(X_train, y_train)
                        best_model = grid_search.best_estimator_
                else:
                    best_model = model
                    best_model.fit(X_train, y_train)

                # Cross-validation
                cv_mean, cv_std = self.enhanced_cross_validation(X_train, y_train, best_model)
                
                # Final evaluation
                y_pred = best_model.predict(X_test)
                y_proba = best_model.predict_proba(X_test) if hasattr(best_model, 'predict_proba') else None

                # Comprehensive metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

                # Store results
                results[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std,
                    'model': best_model,
                    'predictions': y_pred,
                    'true_labels': y_test,
                    'probabilities': y_proba,
                    'n_classes': n_classes,
                    'test_size': len(y_test),
                    'smote_applied': smote_applied,
                    'feature_importance': getattr(best_model, 'feature_importances_', None)
                }

                progress_bar.progress((i + 1) / total_models)

            except Exception as e:
                st.warning(f"Model {name} failed: {str(e)}")
                results[name] = {'error': str(e)}

        return results, le

# ============================
# Enhanced Visualizations
# ============================
class AdvancedVisualizer:
    @staticmethod
    def create_performance_dashboard(results):
        """Create professional performance dashboard"""
        plt.style.use('default')
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

        colors = ['#2563EB', '#3B82F6', '#1D4ED8', '#60A5FA', '#93C5FD']

        # Accuracy
        bars1 = ax1.bar(models, metrics_data['Accuracy'], color=colors, alpha=0.9)
        ax1.set_facecolor('#1E293B')
        ax1.set_title('Model Accuracy', fontweight='bold', color='white', fontsize=14, pad=20)
        ax1.set_ylabel('Score', fontweight='bold', color='white')
        ax1.tick_params(axis='x', rotation=45, colors='white')
        ax1.tick_params(axis='y', colors='white')
        ax1.grid(True, alpha=0.1, axis='y', color='white')
        ax1.set_ylim(0, 1.0)

        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold', color='white')

        # Precision
        bars2 = ax2.bar(models, metrics_data['Precision'], color=colors, alpha=0.9)
        ax2.set_facecolor('#1E293B')
        ax2.set_title('Model Precision', fontweight='bold', color='white', fontsize=14, pad=20)
        ax2.set_ylabel('Score', fontweight='bold', color='white')
        ax2.tick_params(axis='x', rotation=45, colors='white')
        ax2.tick_params(axis='y', colors='white')
        ax2.grid(True, alpha=0.1, axis='y', color='white')
        ax2.set_ylim(0, 1.0)

        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold', color='white')

        # Recall
        bars3 = ax3.bar(models, metrics_data['Recall'], color=colors, alpha=0.9)
        ax3.set_facecolor('#1E293B')
        ax3.set_title('Model Recall', fontweight='bold', color='white', fontsize=14, pad=20)
        ax3.set_ylabel('Score', fontweight='bold', color='white')
        ax3.tick_params(axis='x', rotation=45, colors='white')
        ax3.tick_params(axis='y', colors='white')
        ax3.grid(True, alpha=0.1, axis='y', color='white')
        ax3.set_ylim(0, 1.0)

        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold', color='white')

        # F1-Score
        bars4 = ax4.bar(models, metrics_data['F1-Score'], color=colors, alpha=0.9)
        ax4.set_facecolor('#1E293B')
        ax4.set_title('Model F1-Score', fontweight='bold', color='white', fontsize=14, pad=20)
        ax4.set_ylabel('Score', fontweight='bold', color='white')
        ax4.tick_params(axis='x', rotation=45, colors='white')
        ax4.tick_params(axis='y', colors='white')
        ax4.grid(True, alpha=0.1, axis='y', color='white')
        ax4.set_ylim(0, 1.0)

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

    # Header Section
    st.markdown("""
    <div class="professional-header">
        <div class="header-badge">AI-Powered Fact Analytics Platform</div>
        <h1 class="page-header">Advanced Fact Analysis & Verification</h1>
        <p style="color: #94A3B8; font-size: 1.2rem; line-height: 1.6; margin-bottom: 2rem;">
            Leverage cutting-edge AI to analyze, verify, and extract insights from textual data. 
            Identify patterns, validate claims, and make data-driven decisions with confidence.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # File Upload Section
    st.markdown('<div class="section-header">Start Your Fact Analysis</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload your CSV file for analysis",
        type=["csv"],
        help="Upload a CSV file with text data and target labels"
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.session_state.file_uploaded = True
            
            st.success("✅ Dataset successfully loaded! Ready for advanced AI analysis.")
            
            # Configuration Section
            st.markdown('<div class="section-header">Analysis Configuration</div>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                text_col = st.selectbox(
                    "Text Column",
                    df.columns,
                    help="Select the column containing your text content for analysis"
                )
            
            with col2:
                target_col = st.selectbox(
                    "Target Column", 
                    df.columns,
                    index=min(1, len(df.columns)-1) if len(df.columns) > 1 else 0,
                    help="Select the column containing your categories or labels"
                )
            
            with col3:
                feature_type = st.selectbox(
                    "Analysis Depth",
                    ["Lexical", "Semantic", "Syntactic", "Pragmatic", "Hybrid"],
                    help="Choose the depth of text analysis"
                )
            
            # Advanced Options
            with st.expander("Advanced Configuration"):
                col1, col2 = st.columns(2)
                with col1:
                    enable_smote = st.checkbox("Enable SMOTE (Handle Class Imbalance)", value=True,
                                             help="Synthetic Minority Over-sampling Technique to balance classes")
                with col2:
                    min_samples = st.number_input("Minimum Samples per Class", 
                                                min_value=2, value=5, 
                                                help="Minimum samples required per class")

            # Dataset Overview
            st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)
            
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
            with st.expander("Data Preview", expanded=True):
                st.dataframe(df.head(10), use_container_width=True)
                
            # Class Distribution
            if target_col in df.columns:
                with st.expander("Class Distribution"):
                    class_dist = df[target_col].value_counts()
                    st.bar_chart(class_dist)

            # Start Analysis Button
            if st.button("🚀 Launch Advanced AI Analysis", use_container_width=True):
                perform_advanced_analysis(df, text_col, target_col, feature_type, enable_smote)
                
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")

    else:
        # Features Section when no file is uploaded
        st.info("📁 Please upload a CSV file to begin analysis")

# ============================
# Enhanced Analysis Function with SMOTE
# ============================
def perform_advanced_analysis(df, text_col, target_col, feature_type, enable_smote):
    """Perform advanced analysis with SMOTE integration"""
    st.markdown('<div class="section-header">Advanced Analysis Results</div>', unsafe_allow_html=True)
    
    # Data validation
    if text_col not in df.columns or target_col not in df.columns:
        st.error("❌ Selected columns not found in dataset.")
        return

    # Handle missing values
    if df[text_col].isnull().any():
        df[text_col] = df[text_col].fillna('')

    if df[target_col].isnull().any():
        st.error("❌ Target column contains missing values.")
        return

    if len(df[target_col].unique()) < 2:
        st.error("❌ Target column must have at least 2 unique classes.")
        return

    # Advanced feature extraction
    with st.spinner("🔍 Extracting advanced features from text data..."):
        extractor = AdvancedFeatureExtractor()
        X = df[text_col].astype(str)
        y = df[target_col]

        # Select feature extraction method
        if feature_type == "Lexical":
            X_features = extractor.extract_lexical_features(X)
            st.info("Using advanced lexical features with TF-IDF and n-grams")
        elif feature_type == "Semantic":
            X_features = extractor.extract_semantic_features(X)
            st.info("Using comprehensive semantic features with sentiment and readability analysis")
        elif feature_type == "Syntactic":
            X_features = extractor.extract_syntactic_features(X)
            st.info("Using syntactic features with POS patterns and structural analysis")
        elif feature_type == "Pragmatic":
            X_features = extractor.extract_pragmatic_features(X)
            st.info("Using pragmatic features for fact verification and contextual analysis")
        else:  # Hybrid
            X_features = extractor.extract_hybrid_features(X)
            st.info("Using hybrid features combining all analysis types for maximum performance")

    st.success("✅ Feature extraction completed successfully!")

    # Enhanced model training with SMOTE
    with st.spinner("🤖 Training advanced AI models with SMOTE..."):
        trainer = AdvancedModelTrainer()
        results, label_encoder = trainer.train_and_evaluate(X_features, y, use_smote=enable_smote)

    # Display results
    successful_models = {k: v for k, v in results.items() if 'error' not in v}

    if successful_models:
        # Model Performance Overview
        st.markdown("#### 📊 Model Performance Overview")
        
        cols = st.columns(len(successful_models))
        for idx, (model_name, result) in enumerate(successful_models.items()):
            with cols[idx]:
                accuracy = result['accuracy']
                smote_status = "Yes" if result.get('smote_applied', False) else "No"
                
                st.markdown(f"""
                <div class="model-card">
                    <div style="font-size: 1.3rem; font-weight: 700; color: #F8FAFC; margin-bottom: 1rem;">{model_name}</div>
                    <div class="model-accuracy">{accuracy:.1%}</div>
                    <div style="color: #94A3B8; font-size: 14px; margin-bottom: 12px;">
                        Precision: {result['precision']:.3f}<br>
                        Recall: {result['recall']:.3f}<br>
                        F1-Score: {result['f1_score']:.3f}<br>
                        SMOTE: {smote_status}
                    </div>
                    <div class="progress-container">
                        <div class="progress-fill" style="width: {accuracy*100}%"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Performance Dashboard
        st.markdown("#### 📈 Performance Dashboard")
        viz = AdvancedVisualizer()
        dashboard_fig = viz.create_performance_dashboard(successful_models)
        st.pyplot(dashboard_fig)

        # Best Model Recommendation
        best_model = max(successful_models.items(), key=lambda x: x[1]['accuracy'])
        best_result = best_model[1]
        
        st.success(f"""
        🏆 **Recommended Model: {best_model[0]}**
        
        - **Accuracy**: {best_result['accuracy']:.1%}
        - **Precision**: {best_result['precision']:.3f}
        - **Recall**: {best_result['recall']:.3f}
        - **F1-Score**: {best_result['f1_score']:.3f}
        - **Cross-Validation Score**: {best_result.get('cv_mean', 0):.3f}
        - **SMOTE Applied**: {'Yes' if best_result.get('smote_applied', False) else 'No'}
        """)

    else:
        st.error("❌ No models were successfully trained. Please check your data and try again.")

if __name__ == "__main__":
    main()

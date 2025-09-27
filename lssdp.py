# ============================================
# 📌 Streamlit NLP Phase-wise with SMOTE & GloVe - PROFESSIONAL UI
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from textblob import TextBlob

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import requests
import io

# ============================
# Page Configuration
# ============================
st.set_page_config(
    page_title="Advanced NLP Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================
# Load SpaCy & Globals
# ============================
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("Please install the spaCy English model:")
    st.code("python -m spacy download en_core_web_sm")
    st.stop()

stop_words = STOP_WORDS

# ============================
# GloVe Embeddings Loader
# ============================
@st.cache_data
def load_glove_embeddings(embedding_dim=100):
    """Load GloVe embeddings with caching"""
    try:
        # Try to load from local file first
        try:
            glove_file = f"glove.6B.{embedding_dim}d.txt"
            embeddings_index = {}
            with open(glove_file, 'r', encoding='utf-8') as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    coefs = np.asarray(values[1:], dtype='float32')
                    embeddings_index[word] = coefs
            return embeddings_index
        except FileNotFoundError:
            # If file not found, use a sample for demonstration
            st.info("Using sample GloVe embeddings for demonstration")
            return create_sample_embeddings()
    except Exception as e:
        st.warning(f"GloVe loading failed: {str(e)}. Using TF-IDF instead.")
        return None

def create_sample_embeddings():
    """Create sample embeddings for demonstration"""
    sample_embeddings = {}
    common_words = ['the', 'and', 'is', 'in', 'to', 'of', 'for', 'with', 'on', 'at']
    for word in common_words:
        sample_embeddings[word] = np.random.randn(100).astype('float32')
    return sample_embeddings

def text_to_glove_embeddings(texts, embeddings_index, embedding_dim=100):
    """Convert texts to GloVe embeddings"""
    text_embeddings = []
    for text in texts:
        words = text.lower().split()
        word_vectors = []
        for word in words:
            if word in embeddings_index:
                word_vectors.append(embeddings_index[word])
        
        if len(word_vectors) > 0:
            text_vector = np.mean(word_vectors, axis=0)
        else:
            text_vector = np.zeros(embedding_dim)
        text_embeddings.append(text_vector)
    
    return np.array(text_embeddings)

# ============================
# Professional CSS Styling
# ============================
def load_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.8rem;
        color: #2563eb;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .section-header {
        font-size: 1.4rem;
        color: #1e40af;
        margin: 2rem 0 1rem 0;
        font-weight: 600;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #2563eb;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .success-card {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border-left: 4px solid #16a34a;
    }
    .warning-card {
        background: linear-gradient(135deg, #fefce8 0%, #fef9c3 100%);
        border-left: 4px solid #ca8a04;
    }
    .result-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        margin: 1rem 0;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
    }
    .stButton button {
        width: 100%;
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }
    .feature-preview {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        border-left: 3px solid #94a3b8;
        font-family: monospace;
        font-size: 0.9rem;
        margin: 0.5rem 0;
    }
    .balance-indicator {
        height: 8px;
        border-radius: 4px;
        margin: 5px 0;
        background: linear-gradient(90deg, #ef4444 0%, #f59e0b 50%, #10b981 100%);
    }
    </style>
    """, unsafe_allow_html=True)

load_css()

# ============================
# Phase Feature Extractors
# ============================
def lexical_preprocess(text):
    """Tokenization + Stopwords removal + Lemmatization"""
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.text not in stop_words and token.is_alpha]
    return " ".join(tokens)

def syntactic_features(text):
    """Part-of-Speech tags"""
    doc = nlp(text)
    pos_tags = " ".join([token.pos_ for token in doc])
    return pos_tags

def semantic_features(text):
    """Sentiment polarity & subjectivity"""
    blob = TextBlob(text)
    return [blob.sentiment.polarity, blob.sentiment.subjectivity]

def discourse_features(text):
    """Sentence count + first word of each sentence"""
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    return f"{len(sentences)} {' '.join([s.split()[0] for s in sentences if len(s.split()) > 0])}"

pragmatic_words = ["must", "should", "might", "could", "will", "?", "!"]
def pragmatic_features(text):
    """Counts of modality & special words"""
    text = text.lower()
    return [text.count(w) for w in pragmatic_words]

# ============================
# Enhanced Model Evaluation with SMOTE & GloVe
# ============================
def evaluate_models_with_enhancements(X_features, y, use_smote=False, use_glove=False, embedding_dim=100):
    results = {}
    
    # Load GloVe embeddings if requested
    embeddings_index = None
    if use_glove:
        embeddings_index = load_glove_embeddings(embedding_dim)
        if embeddings_index:
            # Convert to GloVe embeddings
            if hasattr(X_features, 'shape'):  # If it's a matrix
                texts = [" ".join([str(x) for x in row]) for row in X_features.toarray()]
            else:
                texts = X_features.astype(str)
            X_features = text_to_glove_embeddings(texts, embeddings_index, embedding_dim)
    
    models = {
        "Naive Bayes": MultinomialNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": SVC(random_state=42)
    }

    n_classes = len(np.unique(y))
    n_samples = len(y)
    
    # Calculate optimal test size
    min_test_size = n_classes * 2
    test_size = min(0.2, max(0.05, min_test_size / n_samples)) if min_test_size / n_samples > 0.05 else 0.2
    test_size = min(test_size, 0.3)
    
    # Encode labels if they're strings
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_features, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X_features, y_encoded, test_size=test_size, random_state=42, stratify=None
        )
    
    # Apply SMOTE if requested
    smote_info = ""
    if use_smote and n_classes > 1:
        try:
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            smote_info = f"SMOTE applied: {len(X_train)} → {len(X_train_resampled)} samples"
            X_train, y_train = X_train_resampled, y_train_resampled
        except Exception as e:
            smote_info = f"SMOTE failed: {str(e)}"
    
    progress_bar = st.progress(0)
    
    for i, (name, model) in enumerate(models.items()):
        try:
            # Handle different feature types
            if use_glove and embeddings_index:
                # GloVe features are already numerical
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            else:
                # For text features, use pipeline
                if hasattr(X_train, 'toarray'):  # Sparse matrix
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred) * 100
            
            # Calculate baseline
            baseline_acc = np.max(np.bincount(y_encoded)) / len(y_encoded) * 100
            improvement = acc - baseline_acc
            
            results[name] = {
                "accuracy": round(acc, 2),
                "baseline": round(baseline_acc, 2),
                "improvement": round(improvement, 2),
                "model": model,
                "predictions": y_pred,
                "true_labels": y_test
            }
                
        except Exception as e:
            results[name] = {
                "accuracy": 0.0,
                "error": str(e)
            }
        
        progress_bar.progress((i + 1) / len(models))
    
    progress_bar.empty()
    
    return results, n_classes, test_size, smote_info, le

# ============================
# Visualization Functions
# ============================
def plot_class_distribution(y, title="Class Distribution"):
    """Plot class distribution before and after SMOTE"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    
    value_counts = y.value_counts()
    ax.bar(range(len(value_counts)), value_counts.values)
    ax.set_xlabel('Classes')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=45)
    
    # Add balance indicator
    balance_ratio = value_counts.min() / value_counts.max()
    st.markdown(f"**Class Balance Ratio:** {balance_ratio:.3f}")
    st.markdown('<div class="balance-indicator" style="width: {}%"></div>'.format(balance_ratio * 100), 
                unsafe_allow_html=True)
    
    return fig

def plot_confusion_matrix_comparison(results, le):
    """Plot confusion matrices for all models"""
    n_models = len([r for r in results.values() if 'predictions' in r])
    if n_models == 0:
        return None
        
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, (name, result) in enumerate(results.items()):
        if idx >= 4:
            break
        if 'predictions' in result:
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(result['true_labels'], result['predictions'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
            axes[idx].set_title(f'{name}\nAccuracy: {result["accuracy"]}%')
    
    plt.tight_layout()
    return fig

# ============================
# Streamlit UI
# ============================

st.markdown("<div class='main-header'>Advanced NLP Analysis</div>", unsafe_allow_html=True)

# File upload section
st.markdown('<div class="section-header">Data Input</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Data overview
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Records", len(df))
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Columns", len(df.columns))
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Missing Values", df.isnull().sum().sum())
            st.markdown('</div>', unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            unique_classes = df.select_dtypes(include=['object']).nunique().max()
            st.metric("Max Unique Classes", unique_classes)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Configuration section
        st.markdown('<div class="section-header">Analysis Configuration</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            text_col = st.selectbox("Text Column", df.columns)
        with col2:
            target_col = st.selectbox("Target Column", df.columns)
        
        # Target analysis
        if target_col:
            target_unique = df[target_col].nunique()
            
            # Class distribution visualization
            st.markdown("### Class Distribution")
            fig_dist = plot_class_distribution(df[target_col])
            st.pyplot(fig_dist)
            
            if target_unique > 50:
                st.markdown('<div class="warning-card">', unsafe_allow_html=True)
                st.warning(f"High number of classes ({target_unique}). Consider grouping similar categories.")
                st.markdown('</div>', unsafe_allow_html=True)
        
        # NLP phase selection
        st.markdown('<div class="section-header">NLP Phase Selection</div>', unsafe_allow_html=True)
        
        phase_options = {
            "Lexical & Morphological": "Word-level processing (tokenization, lemmatization)",
            "Syntactic": "Grammar analysis (part-of-speech tagging)", 
            "Semantic": "Meaning analysis (sentiment, subjectivity)",
            "Discourse": "Multi-sentence analysis (structure, coherence)",
            "Pragmatic": "Context analysis (intent, modality)"
        }
        
        phase = st.selectbox("Select Analysis Phase", list(phase_options.keys()))
        
        # Advanced options
        st.markdown('<div class="section-header">Advanced Options</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            use_smote = st.checkbox("Apply SMOTE", value=False,
                                  help="Apply Synthetic Minority Over-sampling Technique for imbalanced data")
            if use_smote:
                smote_strategy = st.selectbox("SMOTE Strategy", 
                                            ["auto", "minority", "not minority"],
                                            help="Strategy for SMOTE sampling")
        
        with col2:
            use_glove = st.checkbox("Use GloVe Embeddings", value=False,
                                  help="Use pre-trained GloVe word embeddings (requires GloVe file)")
            if use_glove:
                embedding_dim = st.selectbox("Embedding Dimension", [50, 100, 200, 300],
                                           help="Dimension of GloVe embeddings")
        
        # Run analysis
        if st.button("Start Advanced Analysis", type="primary"):
            # Data validation
            if df[text_col].isnull().any():
                df[text_col] = df[text_col].fillna("")
            
            if df[target_col].isnull().any():
                st.error("Target column contains missing values. Please clean your data.")
                st.stop()
            
            if len(df[target_col].unique()) < 2:
                st.error("Target column must have at least 2 unique classes.")
                st.stop()
            
            with st.spinner(f"Processing {phase} features with advanced options..."):
                X = df[text_col].astype(str)
                y = df[target_col]
                
                # Feature extraction
                try:
                    if phase == "Lexical & Morphological":
                        X_processed = X.apply(lexical_preprocess)
                        X_features = CountVectorizer(max_features=5000).fit_transform(X_processed)

                    elif phase == "Syntactic":
                        X_processed = X.apply(syntactic_features)
                        X_features = CountVectorizer(max_features=5000).fit_transform(X_processed)

                    elif phase == "Semantic":
                        X_features = pd.DataFrame(X.apply(semantic_features).tolist(),
                                                columns=["polarity", "subjectivity"])

                    elif phase == "Discourse":
                        X_processed = X.apply(discourse_features)
                        X_features = CountVectorizer(max_features=5000).fit_transform(X_processed)

                    elif phase == "Pragmatic":
                        X_features = pd.DataFrame(X.apply(pragmatic_features).tolist(),
                                                columns=pragmatic_words)
                    
                    st.markdown('<div class="success-card">', unsafe_allow_html=True)
                    st.success(f"Feature extraction completed")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error during feature extraction: {str(e)}")
                    st.stop()
                
                # Model training with enhancements
                try:
                    results, n_classes, used_test_size, smote_info, label_encoder = evaluate_models_with_enhancements(
                        X_features, y, use_smote=use_smote, use_glove=use_glove, 
                        embedding_dim=embedding_dim if use_glove else 100
                    )
                    
                    # Display enhancement info
                    if smote_info:
                        st.info(f"🔄 {smote_info}")
                    if use_glove:
                        st.info("🔤 Using GloVe word embeddings")
                    
                    # Display results
                    st.markdown('<div class="section-header">Analysis Results</div>', unsafe_allow_html=True)
                    
                    # Results metrics
                    results_data = []
                    for model_name, result in results.items():
                        if "error" in result:
                            results_data.append({
                                "Model": model_name, 
                                "Accuracy": f"Error: {result['error']}", 
                                "Accuracy_float": 0
                            })
                        else:
                            results_data.append({
                                "Model": model_name, 
                                "Accuracy": f"{result['accuracy']}%",
                                "Baseline": f"{result['baseline']}%",
                                "Improvement": f"{result['improvement']}%",
                                "Accuracy_float": result['accuracy']
                            })
                    
                    results_df = pd.DataFrame(results_data)
                    results_df = results_df.sort_values("Accuracy_float", ascending=False)
                    
                    # Model performance cards
                    st.markdown("### Model Performance")
                    cols = st.columns(4)
                    
                    for i, (col, (_, row)) in enumerate(zip(cols, results_df.iterrows())):
                        with col:
                            st.markdown('<div class="result-card">', unsafe_allow_html=True)
                            if "Error" not in row["Accuracy"]:
                                st.metric(
                                    row["Model"], 
                                    row["Accuracy"],
                                    delta=f"{row['Improvement']}% vs baseline"
                                )
                            else:
                                st.error(row["Model"])
                            st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### Performance Comparison")
                        fig, ax = plt.subplots(figsize=(10, 5))
                        
                        successful_models = results_df[results_df["Accuracy_float"] > 0]
                        if len(successful_models) > 0:
                            colors = ['#2563eb', '#16a34a', '#dc2626', '#ea580c']
                            bars = ax.bar(successful_models["Model"], successful_models["Accuracy_float"], 
                                        color=colors[:len(successful_models)])
                            
                            ax.set_ylabel("Accuracy (%)")
                            ax.set_title(f"Model Performance - {phase}")
                            plt.xticks(rotation=45)
                            
                            for bar, v in zip(bars, successful_models["Accuracy_float"]):
                                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                                       f"{v:.1f}%", ha='center', va='bottom', fontsize=10)
                            
                            st.pyplot(fig)
                    
                    with col2:
                        st.markdown("### Enhancement Impact")
                        fig_enhance = plt.subplots(figsize=(10, 5))
                        
                        # Show baseline vs improved accuracy
                        baseline_acc = successful_models["Baseline"].iloc[0].replace('%', '')
                        best_acc = successful_models["Accuracy_float"].iloc[0]
                        
                        fig, ax = plt.subplots(figsize=(8, 5))
                        categories = ['Baseline', 'Best Model']
                        values = [float(baseline_acc), best_acc]
                        bars = ax.bar(categories, values, color=['#94a3b8', '#2563eb'])
                        
                        ax.set_ylabel("Accuracy (%)")
                        ax.set_title("Improvement from Enhancements")
                        
                        for bar, v in zip(bars, values):
                            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                                   f"{v:.1f}%", ha='center', va='bottom', fontsize=12)
                        
                        st.pyplot(fig)
                    
                    # Detailed results
                    with st.expander("📊 Detailed Results"):
                        st.dataframe(results_df.drop("Accuracy_float", axis=1), use_container_width=True)
                        
                        # Show confusion matrices if not too many classes
                        if n_classes <= 10:
                            st.markdown("### Confusion Matrices")
                            fig_cm = plot_confusion_matrix_comparison(results, label_encoder)
                            if fig_cm:
                                st.pyplot(fig_cm)
                        
                except Exception as e:
                    st.error(f"Error during model training: {str(e)}")
                    
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")

else:
    # Welcome state
    st.markdown("""
    <div style='text-align: center; padding: 3rem 1rem; background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); border-radius: 12px; margin: 2rem 0;'>
        <h3 style='color: #1e40af; margin-bottom: 1rem;'>Upload CSV File to Begin Advanced Analysis</h3>
        <p style='color: #475569; max-width: 600px; margin: 0 auto;'>
            Advanced NLP analysis with SMOTE for class balancing and GloVe embeddings for improved accuracy. 
            Upload a CSV file containing text data and labels to get started.
        </p>
    </div>
    
    <div style='margin: 2rem 0;'>
        <div class='section-header'>Advanced Features</div>
        <div class='result-card'>
            <h4>🔄 SMOTE (Synthetic Minority Over-sampling Technique)</h4>
            <p>Handles imbalanced datasets by generating synthetic samples for minority classes.</p>
        </div>
        <div class='result-card'>
            <h4>🔤 GloVe Embeddings</h4>
            <p>Uses pre-trained word vectors for better semantic understanding and improved accuracy.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #64748b; font-size: 0.9rem;'>"
           "Advanced NLP Analysis System | SMOTE + GloVe Enhancements | Professional UI"
           "</div>", unsafe_allow_html=True)

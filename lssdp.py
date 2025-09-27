
# ============================================
# 📌 Streamlit NLP Phase-wise with All Models - PROFESSIONAL UI
# ============================================

import streamlit as st
import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from textblob import TextBlob

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import numpy as np

# ============================
# Page Configuration
# ============================
st.set_page_config(
    page_title="NLP Phase Analyzer",
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
# Model Evaluation
# ============================
def evaluate_models(X_features, y):
    results = {}
    models = {
        "Naive Bayes": MultinomialNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": SVC(random_state=42)
    }

    n_classes = len(y.unique())
    n_samples = len(y)
    
    min_test_size = n_classes * 2
    max_test_size = n_samples - n_classes
    
    if min_test_size >= n_samples * 0.2:
        test_size = min(0.1, (n_samples - n_classes) / n_samples)
    else:
        test_size = 0.2
    
    test_size = min(test_size, 0.3)
    test_size = max(test_size, 0.05)
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_features, y, test_size=test_size, random_state=42, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X_features, y, test_size=test_size, random_state=42, stratify=None
        )
    
    progress_bar = st.progress(0)
    
    for i, (name, model) in enumerate(models.items()):
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred) * 100
            
            if n_classes > 10:
                baseline_acc = y.value_counts().max() / len(y) * 100
                improvement = acc - baseline_acc
                
                results[name] = {
                    "accuracy": round(acc, 2),
                    "baseline": round(baseline_acc, 2),
                    "improvement": round(improvement, 2),
                    "model": model
                }
            else:
                results[name] = {
                    "accuracy": round(acc, 2),
                    "model": model
                }
                
        except Exception as e:
            results[name] = {
                "accuracy": 0.0,
                "error": str(e)
            }
        
        progress_bar.progress((i + 1) / len(models))
    
    progress_bar.empty()
    
    return results, n_classes, test_size

# ============================
# Streamlit UI
# ============================

st.markdown("<div class='main-header'>NLP Phase Analysis</div>", unsafe_allow_html=True)

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
        
        # Run analysis
        if st.button("Start Analysis", type="primary"):
            # Data validation
            if df[text_col].isnull().any():
                df[text_col] = df[text_col].fillna("")
            
            if df[target_col].isnull().any():
                st.error("Target column contains missing values. Please clean your data.")
                st.stop()
            
            if len(df[target_col].unique()) < 2:
                st.error("Target column must have at least 2 unique classes.")
                st.stop()
            
            with st.spinner(f"Processing {phase} features..."):
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
                
                # Model training
                try:
                    results, n_classes, used_test_size = evaluate_models(X_features, y)
                    
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
                            if "improvement" in result:
                                results_data.append({
                                    "Model": model_name, 
                                    "Accuracy": f"{result['accuracy']}%",
                                    "Baseline": f"{result['baseline']}%",
                                    "Improvement": f"{result['improvement']}%",
                                    "Accuracy_float": result['accuracy']
                                })
                            else:
                                results_data.append({
                                    "Model": model_name, 
                                    "Accuracy": f"{result['accuracy']}%",
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
                                if "Improvement" in row:
                                    st.metric(
                                        row["Model"], 
                                        row["Accuracy"],
                                        delta=f"{row['Improvement']} vs baseline"
                                    )
                                else:
                                    st.metric(row["Model"], row["Accuracy"])
                            else:
                                st.error(row["Model"])
                            st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Visualization
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
                    
                    # Technical details
                    with st.expander("Technical Details"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Dataset Info**")
                            st.write(f"Feature Matrix: {X_features.shape}")
                            st.write(f"Number of Classes: {n_classes}")
                        with col2:
                            st.write("**Training Info**")
                            st.write(f"Test Size: {used_test_size:.1%}")
                            st.write(f"Models: 4 algorithms")
                        
                except Exception as e:
                    st.error(f"Error during model training: {str(e)}")
                    
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")

else:
    # Welcome state
    st.markdown("""
    <div style='text-align: center; padding: 3rem 1rem; background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); border-radius: 12px; margin: 2rem 0;'>
        <h3 style='color: #1e40af; margin-bottom: 1rem;'>Upload CSV File to Begin Analysis</h3>
        <p style='color: #475569; max-width: 600px; margin: 0 auto;'>
            Analyze text data across different NLP phases using machine learning models. 
            Upload a CSV file containing text data and labels to get started.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #64748b; font-size: 0.9rem;'>"
           "NLP Phase Analysis System | Professional UI"
           "</div>", unsafe_allow_html=True)

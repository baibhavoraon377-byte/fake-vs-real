# ============================================
# 📌 Streamlit NLP Phase-wise with All Models - SIMPLIFIED UI
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
import time

# ============================
# Page Configuration
# ============================
st.set_page_config(
    page_title="NLP Phase Analyzer",
    page_icon="🧠",
    layout="wide"
)

# ============================
# Load SpaCy & Globals
# ============================
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("⚠️ Please install the spaCy English model first:")
    st.code("python -m spacy download en_core_web_sm")
    st.stop()

stop_words = STOP_WORDS

# ============================
# Custom CSS Styling
# ============================
def load_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .phase-card {
        background: #667eea;
        color: white;
        padding: 1rem;
        border-radius: 10px;
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
# Enhanced Model Evaluation
# ============================
def evaluate_models(X_features, y):
    results = {}
    models = {
        "Naive Bayes": MultinomialNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": SVC(random_state=42)
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=0.2, random_state=42, stratify=y
    )

    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (name, model) in enumerate(models.items()):
        status_text.text(f"Training {name}... ({i+1}/{len(models)})")
        
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred) * 100
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
    
    status_text.text("Training completed!")
    time.sleep(0.5)
    status_text.empty()
    progress_bar.empty()
    
    return results

# ============================
# Simplified Streamlit UI
# ============================

st.markdown("<div class='main-header'>🧠 NLP Phase-wise Analysis</div>", unsafe_allow_html=True)

st.markdown("""
Welcome! This tool compares machine learning models across different NLP processing phases. 
Upload your CSV file with text data and labels to get started.
""")

# File upload section
uploaded_file = st.file_uploader("📤 Upload CSV File", type=["csv"], 
                               help="CSV should contain text data and labels")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        
        st.success(f"✅ File loaded successfully! Shape: {df.shape}")
        
        # Data overview
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        
        # Data preview
        with st.expander("📊 Data Preview"):
            st.dataframe(df.head(10))
            st.write("**Column Types:**")
            st.write(df.dtypes)
        
        # Configuration section
        st.markdown("---")
        st.subheader("⚙️ Analysis Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            text_col = st.selectbox("Select Text Column", df.columns, 
                                  help="Column containing text data")
        with col2:
            target_col = st.selectbox("Select Target Column", df.columns,
                                    help="Column containing labels/categories")
        
        # NLP phase selection with descriptions
        phase_options = {
            "Lexical & Morphological": "Word-level processing (tokenization, lemmatization)",
            "Syntactic": "Grammar analysis (part-of-speech tagging)", 
            "Semantic": "Meaning analysis (sentiment, subjectivity)",
            "Discourse": "Multi-sentence analysis (structure, coherence)",
            "Pragmatic": "Context analysis (intent, modality)"
        }
        
        phase = st.selectbox("Select NLP Phase", list(phase_options.keys()))
        st.info(f"**{phase}**: {phase_options[phase]}")
        
        # Run analysis
        if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
            # Validate data
            if df[text_col].isnull().any():
                st.warning("⚠️ Text column contains missing values. They will be filled with empty strings.")
                df[text_col] = df[text_col].fillna("")
            
            if df[target_col].isnull().any():
                st.error("❌ Target column contains missing values. Please clean your data first.")
                st.stop()
            
            if len(df[target_col].unique()) < 2:
                st.error("❌ Target column must have at least 2 unique classes.")
                st.stop()
            
            with st.spinner(f"Processing {phase} features..."):
                X = df[text_col].astype(str)
                y = df[target_col]
                
                # Feature extraction
                try:
                    if phase == "Lexical & Morphological":
                        X_processed = X.apply(lexical_preprocess)
                        X_features = CountVectorizer().fit_transform(X_processed)

                    elif phase == "Syntactic":
                        X_processed = X.apply(syntactic_features)
                        X_features = CountVectorizer().fit_transform(X_processed)

                    elif phase == "Semantic":
                        X_features = pd.DataFrame(X.apply(semantic_features).tolist(),
                                                columns=["polarity", "subjectivity"])

                    elif phase == "Discourse":
                        X_processed = X.apply(discourse_features)
                        X_features = CountVectorizer().fit_transform(X_processed)

                    elif phase == "Pragmatic":
                        X_features = pd.DataFrame(X.apply(pragmatic_features).tolist(),
                                                columns=pragmatic_words)
                    
                    st.success(f"✅ Feature extraction completed! Shape: {X_features.shape}")
                    
                except Exception as e:
                    st.error(f"❌ Error during feature extraction: {str(e)}")
                    st.stop()
                
                # Model training
                try:
                    results = evaluate_models(X_features, y)
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("📊 Results")
                    
                    # Convert results to DataFrame
                    results_data = []
                    for model_name, result in results.items():
                        if "error" in result:
                            results_data.append({"Model": model_name, "Accuracy": f"Error: {result['error']}", "Accuracy_float": 0})
                        else:
                            results_data.append({"Model": model_name, "Accuracy": f"{result['accuracy']}%", "Accuracy_float": result['accuracy']})
                    
                    results_df = pd.DataFrame(results_data)
                    results_df = results_df.sort_values("Accuracy_float", ascending=False)
                    
                    # Results metrics
                    st.subheader("🏆 Model Performance")
                    cols = st.columns(4)
                    for i, (col, (_, row)) in enumerate(zip(cols, results_df.iterrows())):
                        with col:
                            if "Error" not in row["Accuracy"]:
                                st.metric(row["Model"], row["Accuracy"])
                            else:
                                st.error(row["Model"])
                    
                    # Visualization
                    st.subheader("📈 Performance Chart")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = ax.bar(results_df["Model"], results_df["Accuracy_float"], 
                                color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
                    
                    ax.set_ylabel("Accuracy (%)")
                    ax.set_title(f"Model Performance - {phase}")
                    plt.xticks(rotation=45)
                    
                    # Add value labels on bars
                    for bar, v in zip(bars, results_df["Accuracy_float"]):
                        if v > 0:  # Only add labels for successful models
                            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                                   f"{v:.1f}%", ha='center', va='bottom')
                    
                    st.pyplot(fig)
                    
                    # Results table
                    st.subheader("📋 Detailed Results")
                    st.dataframe(results_df[["Model", "Accuracy"]], use_container_width=True)
                    
                    # Technical details
                    with st.expander("🔧 Technical Details"):
                        st.write(f"**Feature Matrix Shape:** {X_features.shape}")
                        st.write("**Target Distribution:**")
                        st.bar_chart(y.value_counts())
                        st.write(f"**Number of Classes:** {len(y.unique())}")
                        
                except Exception as e:
                    st.error(f"❌ Error during model training: {str(e)}")
                    
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        st.info("💡 Make sure your file is a valid CSV with proper formatting.")

else:
    # Instructions when no file is uploaded
    st.markdown("---")
    st.subheader("📋 How to Use")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **1. Prepare Your Data**
        - CSV format with text and label columns
        - Example structure:
        ```
        text,label
        "I love this product!",positive
        "This is terrible",negative
        ```
        """)
    
    with col2:
        st.markdown("""
        **2. Supported NLP Phases**
        - **Lexical**: Word-level features
        - **Syntactic**: Grammar features  
        - **Semantic**: Meaning features
        - **Discourse**: Structure features
        - **Pragmatic**: Context features
        """)
    
    st.markdown("""
    **3. Expected Results**
    - Accuracy scores for 4 ML models
    - Comparative visualization
    - Technical details and insights
    """)

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #666;'>"
           "NLP Phase Analyzer | Built with Streamlit 🎈"
           "</div>", unsafe_allow_html=True)

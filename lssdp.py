# ============================================
# 📌 Streamlit NLP Phase-wise with All Models - FIXED VERSION
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
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
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
# FIXED Model Evaluation with Smart Sampling
# ============================
def evaluate_models(X_features, y):
    results = {}
    models = {
        "Naive Bayes": MultinomialNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": SVC(random_state=42)
    }

    # Calculate optimal test size based on number of classes
    n_classes = len(y.unique())
    n_samples = len(y)
    
    # Ensure at least 2 samples per class in test set and at least 20% training data
    min_test_size = n_classes * 2  # At least 2 samples per class in test
    max_test_size = n_samples - n_classes  # Leave at least one sample per class in train
    
    if min_test_size >= n_samples * 0.2:
        # Too many classes, use smaller test size but ensure min samples per class
        test_size = min(0.1, (n_samples - n_classes) / n_samples)  # Use 10% or less
        st.warning(f"⚠️ Many classes detected ({n_classes}). Using smaller test size: {test_size:.1%}")
    else:
        test_size = 0.2  # Default 20% test size
    
    # Adjust test size if it's too large
    test_size = min(test_size, 0.3)  # Cap at 30%
    test_size = max(test_size, 0.05)  # At least 5%
    
    st.info(f"📊 Using test size: {test_size:.1%} (Samples: {int(n_samples * test_size)})")
    
    try:
        # Try stratified split first
        X_train, X_test, y_train, y_test = train_test_split(
            X_features, y, test_size=test_size, random_state=42, stratify=y
        )
    except ValueError:
        # If stratified fails, use random split
        st.warning("🔄 Stratified split failed. Using random split instead.")
        X_train, X_test, y_train, y_test = train_test_split(
            X_features, y, test_size=test_size, random_state=42, stratify=None
        )
    
    # Check if any class has only one sample in test set
    test_class_counts = y_test.value_counts()
    if (test_class_counts == 1).any():
        st.warning("⚠️ Some classes have only one sample in test set. Results may be unstable.")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (name, model) in enumerate(models.items()):
        status_text.text(f"Training {name}... ({i+1}/{len(models)})")
        
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred) * 100
            
            # Additional metrics for multi-class scenario
            if n_classes > 10:  # For high number of classes
                # Calculate baseline accuracy (majority class)
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
    
    status_text.text("Training completed!")
    progress_bar.empty()
    
    return results, n_classes, test_size

# ============================
# Streamlit UI
# ============================

st.markdown("<div class='main-header'>🧠 NLP Phase-wise Analysis</div>", unsafe_allow_html=True)

# File upload section
uploaded_file = st.file_uploader("📤 Upload CSV File", type=["csv"], 
                               help="CSV should contain text data and labels")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        
        st.success(f"✅ File loaded successfully! Shape: {df.shape}")
        
        # Data overview
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        with col4:
            unique_classes = df.select_dtypes(include=['object']).nunique().max()
            st.metric("Max Unique Classes", unique_classes)
        
        # Data preview
        with st.expander("📊 Data Preview & Statistics"):
            tab1, tab2, tab3 = st.tabs(["Data", "Info", "Descriptive Stats"])
            with tab1:
                st.dataframe(df.head(10))
            with tab2:
                st.write("**Data Types:**")
                st.write(df.dtypes)
                st.write("**Missing Values:**")
                st.write(df.isnull().sum())
            with tab3:
                st.write(df.describe(include='all'))
        
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
        
        # Show target distribution
        if target_col:
            target_unique = df[target_col].nunique()
            st.info(f"**Target Analysis**: {target_unique} unique classes")
            
            if target_unique > 50:
                st.warning(f"⚠️ High number of classes ({target_unique}). This is a multi-class classification problem. Consider grouping similar categories.")
            
            # Show top classes
            top_classes = df[target_col].value_counts().head(10)
            st.write("**Top 10 Classes:**")
            st.bar_chart(top_classes)
        
        # NLP phase selection
        phase_options = {
            "Lexical & Morphological": "Word-level processing (tokenization, lemmatization)",
            "Syntactic": "Grammar analysis (part-of-speech tagging)", 
            "Semantic": "Meaning analysis (sentiment, subjectivity)",
            "Discourse": "Multi-sentence analysis (structure, coherence)",
            "Pragmatic": "Context analysis (intent, modality)"
        }
        
        phase = st.selectbox("Select NLP Phase", list(phase_options.keys()))
        st.info(f"**{phase}**: {phase_options[phase]}")
        
        # Advanced options
        with st.expander("⚡ Advanced Options"):
            custom_test_size = st.slider("Custom Test Size", 0.05, 0.4, 0.2, 0.05,
                                       help="Adjust test set size for datasets with many classes")
            enable_stratified = st.checkbox("Enable Stratified Sampling", value=True,
                                          help="Maintain class distribution in train/test splits")
        
        # Run analysis
        if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
            # Validate data
            if df[text_col].isnull().any():
                st.warning("⚠️ Text column contains missing values. Filling with empty strings.")
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
                    
                    st.success(f"✅ Feature extraction completed! Shape: {X_features.shape}")
                    
                except Exception as e:
                    st.error(f"❌ Error during feature extraction: {str(e)}")
                    st.stop()
                
                # Model training
                try:
                    results, n_classes, used_test_size = evaluate_models(X_features, y)
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("📊 Results")
                    
                    # Convert results to DataFrame
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
                    
                    # Results metrics
                    st.subheader("🏆 Model Performance")
                    
                    if n_classes > 10:
                        # Multi-class results display
                        cols = st.columns(4)
                        for i, (col, (_, row)) in enumerate(zip(cols, results_df.iterrows())):
                            with col:
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
                    else:
                        # Binary/multi-class results
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
                    
                    successful_models = results_df[results_df["Accuracy_float"] > 0]
                    if len(successful_models) > 0:
                        bars = ax.bar(successful_models["Model"], successful_models["Accuracy_float"], 
                                    color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62727'])
                        
                        ax.set_ylabel("Accuracy (%)")
                        ax.set_title(f"Model Performance - {phase}\n({n_classes} classes, test size: {used_test_size:.1%})")
                        plt.xticks(rotation=45)
                        
                        # Add value labels on bars
                        for bar, v in zip(bars, successful_models["Accuracy_float"]):
                            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                                   f"{v:.1f}%", ha='center', va='bottom', fontsize=9)
                        
                        st.pyplot(fig)
                    
                    # Results table
                    st.subheader("📋 Detailed Results")
                    st.dataframe(results_df.drop("Accuracy_float", axis=1), use_container_width=True)
                    
                    # Technical details
                    with st.expander("🔧 Technical Details"):
                        st.write(f"**Feature Matrix Shape:** {X_features.shape}")
                        st.write(f"**Number of Classes:** {n_classes}")
                        st.write(f"**Test Size Used:** {used_test_size:.1%}")
                        st.write("**Class Distribution:**")
                        st.bar_chart(y.value_counts().head(20))  # Show top 20 classes
                        
                except Exception as e:
                    st.error(f"❌ Error during model training: {str(e)}")
                    
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")

else:
    # Instructions when no file is uploaded
    st.info("👆 Please upload a CSV file to begin analysis")
    st.markdown("""
    ### 📋 Expected Data Format:
    - **CSV file** with text data and labels
    - **Text column**: Contains the text to analyze
    - **Target column**: Contains categories/labels
    
    ### 💡 Tips for Better Results:
    - Ensure your target column has reasonable number of classes (<50 ideal)
    - Clean your text data before uploading
    - For many classes, consider grouping similar categories
    """)

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #666;'>"
           "NLP Phase Analyzer | Handles multi-class datasets | Built with Streamlit 🎈"
           "</div>", unsafe_allow_html=True)

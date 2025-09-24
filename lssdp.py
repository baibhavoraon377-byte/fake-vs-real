# ============================================
# 📌 Streamlit NLP Phase-wise with All Models - MODERN UI
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
import plotly.graph_objects as go
import plotly.express as px
from streamlit_lottie import st_lottie
import requests
import time

# ============================
# Page Configuration
# ============================
st.set_page_config(
    page_title="NLP Phase Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================
# Load SpaCy & Globals
# ============================
@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()
stop_words = STOP_WORDS

# ============================
# Custom CSS Styling
# ============================
def load_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2e86ab;
        border-bottom: 2px solid #2e86ab;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .phase-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)

load_css()

# ============================
# Phase Feature Extractors (Same as before)
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
# Modern Streamlit UI
# ============================

# Sidebar for navigation
with st.sidebar:
    st.markdown("<div class='phase-card'>", unsafe_allow_html=True)
    st.title("🧠 NLP Analyzer")
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("### 📊 Navigation")
    page = st.radio("Go to:", ["🏠 Home", "📈 Analysis", "ℹ️ About"])
    
    st.markdown("---")
    st.markdown("### 🔧 Settings")
    show_details = st.checkbox("Show technical details", value=True)
    
    st.markdown("---")
    st.markdown("### 📚 NLP Phases Info")
    with st.expander("Learn about each phase"):
        st.markdown("""
        - **Lexical**: Word-level processing
        - **Syntactic**: Grammar and structure  
        - **Semantic**: Meaning and sentiment
        - **Discourse**: Multi-sentence analysis
        - **Pragmatic**: Context and intent
        """)

# Main content area
if page == "🏠 Home":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("<div class='main-header'>NLP Phase-wise Analysis</div>", unsafe_allow_html=True)
        st.markdown("""
        ### Welcome to the Advanced NLP Analyzer! 🚀
        
        This tool allows you to compare machine learning models across different 
        **Natural Language Processing phases**. Upload your text data and discover 
        which linguistic features work best for your specific task.
        
        ### How to use:
        1. **Upload** your CSV file with text data
        2. **Select** the text and target columns
        3. **Choose** an NLP phase to analyze
        4. **Compare** model performances
        5. **Visualize** results with interactive charts
        """)
        
        with st.expander("📋 Supported File Format"):
            st.markdown("""
            Your CSV should contain:
            - At least one **text column** (reviews, articles, comments)
            - One **target column** (labels, categories, sentiments)
            - Example structure:
            ```csv
            text,label
            "I love this product!",positive
            "This is terrible",negative
            ```
            """)
    
    with col2:
        # You can add a Lottie animation here or an image
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=200)
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Supported Models", "4")
        st.metric("NLP Phases", "5")
        st.metric("Output Types", "Charts + Metrics")
        st.markdown("</div>", unsafe_allow_html=True)

elif page == "📈 Analysis":
    st.markdown("<div class='section-header'>Data Upload & Analysis</div>", unsafe_allow_html=True)
    
    # File upload section
    uploaded_file = st.file_uploader("📤 Upload your CSV file", type=["csv"], 
                                   help="Upload a CSV file with text data and labels")
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Data overview in columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.metric("Total Records", len(df))
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.metric("Columns", len(df.columns))
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col3:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.metric("Data Type", "Structured")
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Data preview with tabs
            tab1, tab2, tab3 = st.tabs(["📊 Data Preview", "📈 Statistics", "🔍 Sample Data"])
            
            with tab1:
                st.dataframe(df.head(10), use_container_width=True)
            
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Column Types:**")
                    st.write(df.dtypes)
                with col2:
                    st.write("**Basic Statistics:**")
                    st.write(df.describe())
            
            with tab3:
                if st.checkbox("Show random samples"):
                    st.write(df.sample(5))
            
            # Configuration section
            st.markdown("<div class='section-header'>Analysis Configuration</div>", unsafe_allow_html=True)
            
            config_col1, config_col2, config_col3 = st.columns(3)
            
            with config_col1:
                text_col = st.selectbox("📝 Text Column", df.columns, 
                                      help="Select the column containing text data")
            
            with config_col2:
                target_col = st.selectbox("🎯 Target Column", df.columns,
                                        help="Select the column containing labels/categories")
            
            with config_col3:
                phase = st.selectbox("🔬 NLP Phase", [
                    "Lexical & Morphological",
                    "Syntactic", 
                    "Semantic",
                    "Discourse",
                    "Pragmatic"
                ], help="Choose the linguistic level to analyze")
            
            # Phase description
            phase_descriptions = {
                "Lexical & Morphological": "Word-level analysis: tokenization, lemmatization, stopword removal",
                "Syntactic": "Grammar analysis: part-of-speech tagging, sentence structure",
                "Semantic": "Meaning analysis: sentiment, subjectivity, semantic features", 
                "Discourse": "Multi-sentence analysis: coherence, structure, flow",
                "Pragmatic": "Context analysis: intent, modality, pragmatic markers"
            }
            
            st.info(f"**Selected Phase**: {phase_descriptions[phase]}")
            
            # Run analysis button
            if st.button("🚀 Run Comprehensive Analysis", type="primary", use_container_width=True):
                with st.spinner(f"Analyzing {phase} features..."):
                    X = df[text_col].astype(str)
                    y = df[target_col]
                    
                    # Feature extraction
                    feature_extractors = {
                        "Lexical & Morphological": lambda x: CountVectorizer().fit_transform(x.apply(lexical_preprocess)),
                        "Syntactic": lambda x: CountVectorizer().fit_transform(x.apply(syntactic_features)),
                        "Semantic": lambda x: pd.DataFrame(x.apply(semantic_features).tolist(), 
                                                         columns=["polarity", "subjectivity"]),
                        "Discourse": lambda x: CountVectorizer().fit_transform(x.apply(discourse_features)),
                        "Pragmatic": lambda x: pd.DataFrame(x.apply(pragmatic_features).tolist(), 
                                                          columns=pragmatic_words)
                    }
                    
                    X_features = feature_extractors[phase](X)
                    
                    # Model training
                    results = evaluate_models(X_features, y)
                    
                    # Display results
                    st.markdown("<div class='section-header'>📊 Results & Visualization</div>", unsafe_allow_html=True)
                    
                    # Convert results to DataFrame
                    results_data = []
                    for model_name, result in results.items():
                        if "error" in result:
                            results_data.append({"Model": model_name, "Accuracy": f"Error: {result['error']}", "Accuracy_float": 0})
                        else:
                            results_data.append({"Model": model_name, "Accuracy": f"{result['accuracy']}%", "Accuracy_float": result['accuracy']})
                    
                    results_df = pd.DataFrame(results_data)
                    results_df = results_df.sort_values("Accuracy_float", ascending=False)
                    
                    # Results in columns
                    res_col1, res_col2, res_col3, res_col4 = st.columns(4)
                    models = results_df.head(4).itertuples()
                    
                    for col, model in zip([res_col1, res_col2, res_col3, res_col4], models):
                        with col:
                            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                            if "Error" not in model.Accuracy:
                                st.metric(f"🏆 {model.Model}", model.Accuracy)
                            else:
                                st.error(f"{model.Model}: Error")
                            st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Visualization tabs
                    viz_tab1, viz_tab2 = st.tabs(["📈 Interactive Chart", "📋 Results Table"])
                    
                    with viz_tab1:
                        # Plotly interactive chart
                        fig = px.bar(results_df, x='Model', y='Accuracy_float', 
                                   title=f'Model Performance - {phase}',
                                   color='Accuracy_float',
                                   color_continuous_scale='viridis')
                        fig.update_layout(xaxis_title="Models", yaxis_title="Accuracy (%)")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with viz_tab2:
                        st.dataframe(results_df[['Model', 'Accuracy']], use_container_width=True)
                    
                    # Technical details expander
                    if show_details:
                        with st.expander("🔧 Technical Details"):
                            st.write("**Feature Matrix Shape:**", X_features.shape)
                            st.write("**Target Distribution:**")
                            st.bar_chart(y.value_counts())
                            st.write("**Sample Processed Features:**")
                            if hasattr(X_features, 'shape'):
                                st.write(f"Sparse matrix with {X_features.shape[0]} samples and {X_features.shape[1]} features")
                            else:
                                st.write(X_features.head())
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    else:
        st.info("👆 Please upload a CSV file to begin analysis")

elif page == "ℹ️ About":
    st.markdown("<div class='section-header'>About This Application</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🧠 NLP Phase-wise Analysis Tool
        
        This application demonstrates how different levels of linguistic analysis 
        affect machine learning model performance on text data.
        
        #### 🔬 Supported NLP Phases:
        1. **Lexical & Morphological** - Word-level processing
        2. **Syntactic** - Grammar and sentence structure  
        3. **Semantic** - Meaning and sentiment analysis
        4. **Discourse** - Multi-sentence coherence
        5. **Pragmatic** - Context and intent analysis
        
        #### 🤖 Machine Learning Models:
        - **Naive Bayes** - Fast and efficient for text
        - **Decision Tree** - Interpretable and simple
        - **Logistic Regression** - Balanced performance
        - **Support Vector Machine** - Powerful for complex patterns
        
        #### 🛠️ Built With:
        - Streamlit for the web interface
        - spaCy for advanced NLP processing
        - scikit-learn for machine learning
        - Plotly for interactive visualizations
        """)
    
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("### 📊 Application Stats")
        st.markdown("- **Version**: 2.0")
        st.markdown("- **Last Updated**: 2024")
        st.markdown("- **License**: MIT")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("### 👨‍💻 Developer Info")
        st.markdown("""
        This tool is designed for:
        - NLP researchers
        - Data scientists  
        - Machine learning engineers
        - Students learning NLP
        """)

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #666;'>"
           "NLP Phase Analyzer © 2024 | Built with Streamlit 🎈"
           "</div>", unsafe_allow_html=True)

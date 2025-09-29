# ============================================
# 🎨 NLP Analysis Suite
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from textblob import TextBlob

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import altair as alt

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
import time

# ============================
# Configuration
# ============================
st.set_page_config(
    page_title="NLP Canvas | Canva-Style Analytics",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================
# Canva-Style CSS with Google Fonts
# ============================
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">

<style>
    /* Canva Color Palette */
    :root {
        --canva-bg: #F9FAFB;
        --canva-white: #FFFFFF;
        --canva-card: #FFFFFF;
        --canva-purple: #8B5CF6;
        --canva-teal: #14B8A6;
        --canva-blue: #3B82F6;
        --canva-pink: #EC4899;
        --canva-orange: #F59E0B;
        --canva-text: #1F2937;
        --canva-text-light: #6B7280;
        --canva-border: #E5E7EB;
        --canva-success: #10B981;
    }
    
    /* Main styles */
    .main {
        background-color: var(--canva-bg);
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-family: 'Poppins', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, var(--canva-purple) 0%, var(--canva-teal) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        letter-spacing: -0.02em;
    }
    
    .subheader {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        color: var(--canva-text-light);
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 400;
    }
    
    .canva-card {
        background: var(--canva-white);
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        border: 1px solid var(--canva-border);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .canva-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.08), 0 10px 10px -5px rgba(0, 0, 0, 0.02);
    }
    
    .metric-card {
        background: linear-gradient(135deg, var(--canva-white) 0%, #F8FAFC 100%);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border-left: 4px solid var(--canva-purple);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        border-left-color: var(--canva-teal);
        transform: scale(1.02);
    }
    
    .section-title {
        font-family: 'Poppins', sans-serif;
        font-size: 1.8rem;
        font-weight: 600;
        color: var(--canva-text);
        margin: 3rem 0 1.5rem 0;
        padding-bottom: 0.8rem;
        border-bottom: 2px solid var(--canva-border);
    }
    
    .feature-pill {
        display: inline-block;
        background: linear-gradient(135deg, var(--canva-purple) 0%, var(--canva-blue) 100%);
        color: white;
        padding: 0.5rem 1.2rem;
        border-radius: 20px;
        margin: 0.3rem;
        font-size: 0.9rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .feature-pill:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(139, 92, 246, 0.2);
    }
    
    .upload-area {
        border: 2px dashed var(--canva-border);
        border-radius: 16px;
        padding: 3rem 2rem;
        text-align: center;
        background: var(--canva-white);
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .upload-area:hover {
        border-color: var(--canva-purple);
        background: #F8FAFC;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, var(--canva-purple) 0%, var(--canva-blue) 100%);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(139, 92, 246, 0.3);
    }
    
    /* Custom tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: var(--canva-white);
        padding: 1rem;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: var(--canva-white);
        border-radius: 8px 8px 0px 0px;
        gap: 1rem;
        padding: 0 2rem;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--canva-purple);
        color: white;
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
# Enhanced Feature Engineering
# ============================
class CanvaFeatureExtractor:
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
# Canva Model Trainer
# ============================
class CanvaModelTrainer:
    def __init__(self):
        self.models = {
            "🎯 Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
            "🌲 Random Forest": RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced'),
            "⚡ Support Vector Machine": SVC(random_state=42, probability=True, class_weight='balanced'),
            "📊 Naive Bayes": MultinomialNB()
        }
    
    def train_and_evaluate(self, X, y):
        """Canva-style model training with comprehensive evaluation"""
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
# Interactive Visualizations with Plotly
# ============================
class CanvaVisualizer:
    @staticmethod
    def create_performance_radar(results):
        """Create an interactive radar chart for model performance"""
        models = []
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = []
        
        for model_name, result in results.items():
            if 'error' not in result:
                clean_name = model_name.replace('🎯 ', '').replace('🌲 ', '').replace('⚡ ', '').replace('📊 ', '')
                models.append(clean_name)
                values.append([
                    result['accuracy'],
                    result['precision'],
                    result['recall'],
                    result['f1_score']
                ])
        
        fig = go.Figure()
        
        colors = ['#8B5CF6', '#14B8A6', '#3B82F6', '#EC4899']
        
        for i, (model, metric_values) in enumerate(zip(models, values)):
            fig.add_trace(go.Scatterpolar(
                r=metric_values + [metric_values[0]],  # Close the radar
                theta=metrics + [metrics[0]],
                fill='toself',
                name=model,
                line=dict(color=colors[i % len(colors)], width=3),
                fillcolor=colors[i % len(colors)] + '40',  # Add transparency
                hovertemplate='<b>%{theta}</b>: %{r:.3f}<extra></extra>'
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickfont=dict(size=11),
                    gridcolor='#E5E7EB'
                ),
                angularaxis=dict(
                    tickfont=dict(size=12),
                    gridcolor='#E5E7EB'
                ),
                bgcolor='rgba(0,0,0,0)'
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                font=dict(size=12)
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=500,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        return fig

    @staticmethod
    def create_metric_comparison(results):
        """Create interactive bar chart for metric comparison"""
        models = []
        metrics_data = {
            'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-Score': []
        }
        
        for model_name, result in results.items():
            if 'error' not in result:
                clean_name = model_name.replace('🎯 ', '').replace('🌲 ', '').replace('⚡ ', '').replace('📊 ', '')
                models.append(clean_name)
                metrics_data['Accuracy'].append(result['accuracy'])
                metrics_data['Precision'].append(result['precision'])
                metrics_data['Recall'].append(result['recall'])
                metrics_data['F1-Score'].append(result['f1_score'])
        
        fig = go.Figure()
        
        colors = ['#8B5CF6', '#14B8A6', '#3B82F6', '#EC4899']
        metrics = list(metrics_data.keys())
        
        for i, metric in enumerate(metrics):
            fig.add_trace(go.Bar(
                name=metric,
                x=models,
                y=metrics_data[metric],
                marker_color=colors[i],
                hovertemplate='<b>%{x}</b><br>%{fullData.name}: %{y:.3f}<extra></extra>',
                text=[f'{val:.3f}' for val in metrics_data[metric]],
                textposition='auto',
            ))
        
        fig.update_layout(
            barmode='group',
            xaxis_title="Models",
            yaxis_title="Score",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12),
            height=500,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )
        
        return fig

    @staticmethod
    def create_confusion_matrix_heatmap(results, selected_model):
        """Create interactive confusion matrix heatmap"""
        if selected_model in results and 'error' not in results[selected_model]:
            result = results[selected_model]
            cm = confusion_matrix(result['true_labels'], result['predictions'])
            
            fig = ff.create_annotated_heatmap(
                z=cm,
                x=[f'Class {i}' for i in range(result['n_classes'])],
                y=[f'Class {i}' for i in range(result['n_classes'])],
                colorscale='Blues',
                showscale=True,
                hoverinfo='z'
            )
            
            fig.update_layout(
                title=f'Confusion Matrix - {selected_model}',
                xaxis_title='Predicted',
                yaxis_title='Actual',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=400
            )
            
            return fig
        return None

    @staticmethod
    def create_performance_gauge(results, selected_model):
        """Create gauge chart for individual model performance"""
        if selected_model in results and 'error' not in results[selected_model]:
            result = results[selected_model]
            accuracy = result['accuracy']
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = accuracy,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': f"Accuracy - {selected_model}", 'font': {'size': 20}},
                delta = {'reference': 0.5, 'increasing': {'color': "#10B981"}},
                gauge = {
                    'axis': {'range': [None, 1], 'tickwidth': 1, 'tickcolor': "#1F2937"},
                    'bar': {'color': "#8B5CF6"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 0.5], 'color': '#FEF3C7'},
                        {'range': [0.5, 0.8], 'color': '#FDE68A'},
                        {'range': [0.8, 1], 'color': '#D9F99D'}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.9}}))
            
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                height=300,
                font={'color': "#1F2937", 'family': "Inter"}
            )
            
            return fig
        return None

# ============================
# Drill-down Analysis Components
# ============================
class DrillDownAnalyzer:
    @staticmethod
    def create_entity_explorer(texts, selected_text_index=None):
        """Create interactive entity explorer with drill-down"""
        if selected_text_index is not None and 0 <= selected_text_index < len(texts):
            text = texts.iloc[selected_text_index] if hasattr(texts, 'iloc') else texts[selected_text_index]
            doc = nlp(str(text))
            
            entities = []
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                })
            
            return entities
        return []

    @staticmethod
    def create_keyword_context(texts, keyword, n_examples=3):
        """Show context around specific keywords"""
        examples = []
        for text in texts[:100]:  # Limit for performance
            if keyword.lower() in str(text).lower():
                context = str(text)
                examples.append(context)
                if len(examples) >= n_examples:
                    break
        return examples

# ============================
# Main Application
# ============================
def main():
    # Header with Canva-style design
    st.markdown("""
    <div style='text-align: center; padding: 3rem 2rem; background: linear-gradient(135deg, #F9FAFB 0%, #FFFFFF 100%); 
                border-radius: 24px; margin: 2rem 0; border: 1px solid #E5E7EB;'>
        <h1 class='main-header'>🎨 NLP Canvas</h1>
        <p class='subheader'>Beautiful, interactive text analytics powered by Canva-inspired design</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature Highlights with interactive pills
    st.markdown("""
    <div style='text-align: center; margin: 2rem 0;'>
        <span class='feature-pill'>📖 Lexical Analysis</span>
        <span class='feature-pill'>🎭 Semantic Understanding</span>
        <span class='feature-pill'>🔧 Syntactic Parsing</span>
        <span class='feature-pill'>🎯 Pragmatic Insights</span>
        <span class='feature-pill'>📊 Interactive Visuals</span>
        <span class='feature-pill'>🔍 Drill-down Explorer</span>
    </div>
    """, unsafe_allow_html=True)
    
    # File Upload Section
    st.markdown('<div class="section-title">📁 Upload Your Dataset</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader("", type=["csv"], 
                                       help="Upload your CSV file for analysis",
                                       label_visibility="collapsed")
    
    with col2:
        st.markdown("""
        <div style='text-align: center; color: #6B7280; font-size: 0.9rem;'>
            📝 Supports CSV files with text columns<br>
            🎯 Automatic feature detection<br>
            ⚡ Real-time processing
        </div>
        """, unsafe_allow_html=True)
    
    if uploaded_file is not None:
        try:
            # Progress animation
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(100):
                progress_bar.progress(i + 1)
                status_text.text(f"Loading and processing data... {i+1}%")
                time.sleep(0.01)
            
            status_text.text("Data loaded successfully!")
            time.sleep(0.5)
            status_text.empty()
            
            df = pd.read_csv(uploaded_file)
            
            # Data Explorer in Canva Cards
            st.markdown('<div class="section-title">🔍 Data Explorer</div>', unsafe_allow_html=True)
            
            with st.container():
                tab1, tab2, tab3 = st.tabs(["📊 Data Preview", "📈 Quick Stats", "🎯 Data Quality"])
                
                with tab1:
                    st.markdown('<div class="canva-card">', unsafe_allow_html=True)
                    st.dataframe(df.head(10), use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with tab2:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3 style="color: #8B5CF6; margin: 0; font-size: 2rem;">{df.shape[0]:,}</h3>
                            <p style="color: #6B7280; margin: 0;">Total Records</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3 style="color: #14B8A6; margin: 0; font-size: 2rem;">{df.shape[1]}</h3>
                            <p style="color: #6B7280; margin: 0;">Features</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with col3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3 style="color: #3B82F6; margin: 0; font-size: 2rem;">{df.isnull().sum().sum()}</h3>
                            <p style="color: #6B7280; margin: 0;">Missing Values</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with col4:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3 style="color: #EC4899; margin: 0; font-size: 2rem;">{len(df.dtypes.unique())}</h3>
                            <p style="color: #6B7280; margin: 0;">Data Types</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with tab3:
                    st.markdown('<div class="canva-card">', unsafe_allow_html=True)
                    col1, col2 = st.columns(2)
                    with col1:
                        # Data quality metrics
                        completeness = (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
                        st.metric("Data Completeness", f"{completeness:.1f}%")
                        
                        text_cols = df.select_dtypes(include=['object']).columns
                        st.metric("Text Columns", len(text_cols))
                    
                    with col2:
                        duplicate_rows = df.duplicated().sum()
                        st.metric("Duplicate Rows", duplicate_rows)
                        
                        memory_usage = df.memory_usage(deep=True).sum() / 1024**2
                        st.metric("Memory Usage", f"{memory_usage:.2f} MB")
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Analysis Configuration
            st.markdown('<div class="section-title">⚙️ Analysis Configuration</div>', unsafe_allow_html=True)
            
            config_col1, config_col2, config_col3 = st.columns([2, 2, 1])
            
            with config_col1:
                text_col = st.selectbox("Select Text Column", df.columns, 
                                      help="Choose the column containing text data for analysis")
            
            with config_col2:
                target_col = st.selectbox("Select Target Column", df.columns,
                                        help="Choose the column for classification target")
            
            with config_col3:
                st.markdown("<br>", unsafe_allow_html=True)
                analyze_btn = st.button("🚀 Launch Analysis", use_container_width=True)
            
            # Feature type selection with visual cards
            st.markdown("### 🎨 Select Analysis Type")
            feature_cols = st.columns(4)
            
            with feature_cols[0]:
                st.markdown("""
                <div class="canva-card" style='text-align: center; cursor: pointer;' onclick='alert("Lexical Analysis Selected")'>
                    <h3 style='color: #8B5CF6;'>📖</h3>
                    <h4 style='color: #1F2937;'>Lexical</h4>
                    <p style='color: #6B7280; font-size: 0.9rem;'>Word-level analysis & patterns</p>
                </div>
                """, unsafe_allow_html=True)
                lexical_selected = st.radio("", ["Lexical"], key="lexical", label_visibility="collapsed")
            
            with feature_cols[1]:
                st.markdown("""
                <div class="canva-card" style='text-align: center; cursor: pointer;'>
                    <h3 style='color: #14B8A6;'>🎭</h3>
                    <h4 style='color: #1F2937;'>Semantic</h4>
                    <p style='color: #6B7280; font-size: 0.9rem;'>Meaning & sentiment analysis</p>
                </div>
                """, unsafe_allow_html=True)
                semantic_selected = st.radio("", ["Semantic"], key="semantic", label_visibility="collapsed")
            
            with feature_cols[2]:
                st.markdown("""
                <div class="canva-card" style='text-align: center; cursor: pointer;'>
                    <h3 style='color: #3B82F6;'>🔧</h3>
                    <h4 style='color: #1F2937;'>Syntactic</h4>
                    <p style='color: #6B7280; font-size: 0.9rem;'>Grammar & structure analysis</p>
                </div>
                """, unsafe_allow_html=True)
                syntactic_selected = st.radio("", ["Syntactic"], key="syntactic", label_visibility="collapsed")
            
            with feature_cols[3]:
                st.markdown("""
                <div class="canva-card" style='text-align: center; cursor: pointer;'>
                    <h3 style='color: #EC4899;'>🎯</h3>
                    <h4 style='color: #1F2937;'>Pragmatic</h4>
                    <p style='color: #6B7280; font-size: 0.9rem;'>Context & intent analysis</p>
                </div>
                """, unsafe_allow_html=True)
                pragmatic_selected = st.radio("", ["Pragmatic"], key="pragmatic", label_visibility="collapsed")
            
            if analyze_btn:
                if df[text_col].isnull().any():
                    df[text_col] = df[text_col].fillna('')
                
                if df[target_col].isnull().any():
                    st.error("🎯 Target column contains missing values. Please clean your data.")
                    return
                
                if len(df[target_col].unique()) < 2:
                    st.error("🎯 Target column must have at least 2 unique classes for classification.")
                    return
                
                # Feature Extraction with progress
                with st.spinner("🔄 Extracting advanced features..."):
                    progress_text = st.empty()
                    progress_bar = st.progress(0)
                    
                    extractor = CanvaFeatureExtractor()
                    X = df[text_col].astype(str)
                    y = df[target_col]
                    
                    # Simulate progress for feature extraction
                    for i in range(4):
                        progress_bar.progress((i + 1) * 25)
                        progress_text.text(f"Extracting features... Step {i + 1}/4")
                        time.sleep(0.5)
                    
                    # Use lexical features for demonstration
                    X_features = extractor.extract_lexical_features(X)
                    progress_bar.progress(100)
                    progress_text.text("Feature extraction complete!")
                    time.sleep(0.5)
                    progress_text.empty()
                
                # Model Training
                with st.spinner("🤖 Training machine learning models..."):
                    trainer = CanvaModelTrainer()
                    results, label_encoder = trainer.train_and_evaluate(X_features, y)
                
                successful_models = {k: v for k, v in results.items() if 'error' not in v}
                
                if successful_models:
                    # Enhanced Visualization Section
                    st.markdown('<div class="section-title">📊 Interactive Results Dashboard</div>', unsafe_allow_html=True)
                    
                    # Performance Overview
                    col1, col2 = st.columns([3, 2])
                    
                    with col1:
                        st.markdown("#### 📈 Performance Radar")
                        radar_fig = CanvaVisualizer.create_performance_radar(successful_models)
                        st.plotly_chart(radar_fig, use_container_width=True)
                    
                    with col2:
                        st.markdown("#### 🎯 Model Comparison")
                        bar_fig = CanvaVisualizer.create_metric_comparison(successful_models)
                        st.plotly_chart(bar_fig, use_container_width=True)
                    
                    # Drill-down Analysis
                    st.markdown('<div class="section-title">🔍 Drill-down Analysis</div>', unsafe_allow_html=True)
                    
                    drill_col1, drill_col2 = st.columns(2)
                    
                    with drill_col1:
                        st.markdown("#### 🎛️ Model Details")
                        selected_model = st.selectbox("Choose a model to explore:", list(successful_models.keys()))
                        
                        if selected_model:
                            gauge_fig = CanvaVisualizer.create_performance_gauge(successful_models, selected_model)
                            if gauge_fig:
                                st.plotly_chart(gauge_fig, use_container_width=True)
                    
                    with drill_col2:
                        st.markdown("#### 🎪 Confusion Matrix")
                        cm_fig = CanvaVisualizer.create_confusion_matrix_heatmap(successful_models, selected_model)
                        if cm_fig:
                            st.plotly_chart(cm_fig, use_container_width=True)
                    
                    # Entity Explorer
                    st.markdown("#### 🔍 Text Entity Explorer")
                    entity_col1, entity_col2 = st.columns([1, 2])
                    
                    with entity_col1:
                        sample_texts = df[text_col].head(10)
                        selected_text_index = st.selectbox("Select a text sample:", 
                                                         range(len(sample_texts)),
                                                         format_func=lambda x: f"Sample {x+1}: {sample_texts.iloc[x][:50]}...")
                    
                    with entity_col2:
                        if selected_text_index is not None:
                            entities = DrillDownAnalyzer.create_entity_explorer(sample_texts, selected_text_index)
                            if entities:
                                st.markdown("**Discovered Entities:**")
                                for entity in entities:
                                    st.markdown(f"""
                                    <div style='background: #F3F4F6; padding: 0.5rem 1rem; border-radius: 8px; margin: 0.3rem 0; 
                                                border-left: 4px solid #8B5CF6;'>
                                        <strong>{entity['text']}</strong> → <span style='color: #8B5CF6;'>{entity['label']}</span>
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.info("No named entities found in this text sample.")
                    
                    # Best Model Recommendation
                    best_model_name, best_model_result = max(successful_models.items(), 
                                                           key=lambda x: x[1]['accuracy'])
                    
                    st.success(f"""
                    🎉 **Analysis Complete!** 
                    
                    **Recommended Model**: {best_model_name}  
                    **Accuracy**: {best_model_result['accuracy']:.1%}  
                    **F1-Score**: {best_model_result['f1_score']:.1%}
                    
                    This model shows the best overall performance for your dataset.
                    """)
                
                else:
                    st.error("❌ No models were successfully trained. Please check your data and try again.")
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("💡 Please ensure your CSV file is properly formatted and contains text data.")
    
    else:
        # Welcome Section with Canva-style design
        st.markdown("""
        <div class='canva-card' style='text-align: center; padding: 4rem 2rem;'>
            <div style='font-size: 4rem; margin-bottom: 2rem;'>🎨</div>
            <h2 style='color: #1F2937; margin-bottom: 1.5rem;'>Welcome to NLP Canvas</h2>
            <p style='color: #6B7280; font-size: 1.2rem; line-height: 1.6; margin-bottom: 2.5rem;'>
                Upload your CSV file to unlock beautiful, interactive text analysis with 
                <strong>drill-down capabilities</strong> and <strong>Canva-inspired visuals</strong>.
            </p>
            <div style='background: linear-gradient(135deg, #8B5CF6 0%, #3B82F6 100%); 
                        padding: 1.5rem; border-radius: 16px; color: white;'>
                <h3 style='margin: 0 0 1rem 0;'>🚀 Get Started</h3>
                <p style='margin: 0; opacity: 0.9;'>Upload a CSV file above to begin your analysis</p>
            </div>
        </div>
        
        <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5rem; margin: 3rem 0;'>
            <div class='canva-card'>
                <h3 style='color: #8B5CF6;'>📊 Interactive Dashboards</h3>
                <p style='color: #6B7280;'>Beautiful, responsive visualizations with hover effects and animations</p>
            </div>
            <div class='canva-card'>
                <h3 style='color: #14B8A6;'>🔍 Drill-down Analysis</h3>
                <p style='color: #6B7280;'>Click on entities and metrics to explore detailed insights</p>
            </div>
            <div class='canva-card'>
                <h3 style='color: #3B82F6;'>🤖 Smart ML Models</h3>
                <p style='color: #6B7280;'>Multiple algorithms with automatic performance comparison</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

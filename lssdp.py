# ============================================
# 🎨 Canva-Style NLP Analysis Suite - Complete
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

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
# Canva-Style CSS
# ============================
st.markdown("""
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
    
    .main {
        background-color: var(--canva-bg);
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, var(--canva-purple) 0%, var(--canva-teal) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .subheader {
        font-size: 1.1rem;
        color: var(--canva-text-light);
        text-align: center;
        margin-bottom: 3rem;
    }
    
    .canva-card {
        background: var(--canva-white);
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        border: 1px solid var(--canva-border);
        transition: all 0.3s ease;
    }
    
    .canva-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.08);
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
    }
    
    .upload-area {
        border: 2px dashed var(--canva-border);
        border-radius: 16px;
        padding: 3rem 2rem;
        text-align: center;
        background: var(--canva-white);
        transition: all 0.3s ease;
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
</style>
""", unsafe_allow_html=True)

# ============================
# Text Analysis Functions
# ============================
class TextAnalyzer:
    @staticmethod
    def analyze_sentiment(text):
        """Analyze text sentiment using TextBlob"""
        try:
            blob = TextBlob(str(text))
            return {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity,
                'sentiment': 'positive' if blob.sentiment.polarity > 0.1 else 'negative' if blob.sentiment.polarity < -0.1 else 'neutral'
            }
        except:
            return {'polarity': 0, 'subjectivity': 0, 'sentiment': 'neutral'}
    
    @staticmethod
    def extract_entities_simple(text):
        """Simple entity extraction using regex patterns"""
        text = str(text)
        entities = {
            'emails': re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text),
            'urls': re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text),
            'hashtags': re.findall(r'#\w+', text),
            'mentions': re.findall(r'@\w+', text)
        }
        return entities
    
    @staticmethod
    def get_text_stats(text):
        """Get basic text statistics"""
        text = str(text)
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
            'avg_sentence_length': np.mean([len(sent.split()) for sent in sentences]) if sentences else 0,
            'unique_words': len(set(words)),
            'readability_score': len(words) / len(sentences) if sentences else 0
        }

# ============================
# Visualization Functions
# ============================
class CanvaVisualizer:
    @staticmethod
    def create_sentiment_chart(sentiment_data):
        """Create sentiment distribution chart"""
        sentiment_counts = pd.Series(sentiment_data).value_counts()
        
        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="Sentiment Distribution",
            color=sentiment_counts.index,
            color_discrete_map={
                'positive': '#10B981',
                'negative': '#EF4444', 
                'neutral': '#6B7280'
            }
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(showlegend=False)
        return fig
    
    @staticmethod
    def create_word_length_chart(length_data):
        """Create word length distribution chart"""
        fig = px.histogram(
            x=length_data,
            title="Word Length Distribution",
            labels={'x': 'Word Count', 'y': 'Frequency'},
            color_discrete_sequence=['#8B5CF6']
        )
        fig.update_layout(
            xaxis_title="Word Count",
            yaxis_title="Number of Texts",
            showlegend=False
        )
        return fig
    
    @staticmethod
    def create_sentiment_over_time(time_data, sentiment_data):
        """Create sentiment over time chart"""
        if len(time_data) > 1:
            fig = px.line(
                x=time_data,
                y=sentiment_data,
                title="Sentiment Over Time",
                labels={'x': 'Text Index', 'y': 'Sentiment Polarity'},
                color_discrete_sequence=['#14B8A6']
            )
            fig.update_layout(
                xaxis_title="Text Sequence",
                yaxis_title="Sentiment Polarity",
                showlegend=False
            )
            return fig
        return None
    
    @staticmethod
    def create_entity_chart(entity_counts):
        """Create entity type distribution chart"""
        if entity_counts:
            fig = px.bar(
                x=list(entity_counts.keys()),
                y=list(entity_counts.values()),
                title="Entity Type Distribution",
                color=list(entity_counts.keys()),
                color_discrete_sequence=['#8B5CF6', '#14B8A6', '#3B82F6', '#EC4899']
            )
            fig.update_layout(
                xaxis_title="Entity Type",
                yaxis_title="Count",
                showlegend=False
            )
            return fig
        return None

# ============================
# Drill-down Analysis
# ============================
class DrillDownAnalyzer:
    def __init__(self, df, text_column):
        self.df = df
        self.text_column = text_column
    
    def get_text_samples(self, sentiment_type=None, min_words=0, max_words=1000):
        """Get text samples filtered by criteria"""
        samples = self.df.copy()
        
        # Filter by word count
        samples['word_count'] = samples[self.text_column].astype(str).str.split().str.len()
        samples = samples[(samples['word_count'] >= min_words) & (samples['word_count'] <= max_words)]
        
        # Filter by sentiment if specified
        if sentiment_type:
            samples['sentiment'] = samples[self.text_column].apply(
                lambda x: TextAnalyzer.analyze_sentiment(x)['sentiment']
            )
            samples = samples[samples['sentiment'] == sentiment_type]
        
        return samples.head(10)
    
    def find_texts_with_keyword(self, keyword, n_samples=5):
        """Find texts containing specific keywords"""
        keyword = keyword.lower()
        matches = self.df[
            self.df[self.text_column].astype(str).str.lower().str.contains(keyword, na=False)
        ].head(n_samples)
        return matches

# ============================
# Main Application
# ============================
def main():
    # Header
    st.markdown("""
    <div style='text-align: center; padding: 3rem 2rem; background: linear-gradient(135deg, #F9FAFB 0%, #FFFFFF 100%); 
                border-radius: 24px; margin: 2rem 0; border: 1px solid #E5E7EB;'>
        <h1 class='main-header'>🎨 NLP Canvas</h1>
        <p class='subheader'>Beautiful, interactive text analytics powered by Canva-inspired design</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature Highlights
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
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file with text data", 
        type=["csv"],
        help="Upload a CSV file containing at least one text column"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            
            # Show success message
            st.success(f"✅ Successfully loaded {len(df):,} rows and {len(df.columns)} columns")
            
            # Data Explorer
            st.markdown('<div class="section-title">🔍 Data Explorer</div>', unsafe_allow_html=True)
            
            # Quick Stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="color: #8B5CF6; margin: 0; font-size: 2rem;">{len(df):,}</h3>
                    <p style="color: #6B7280; margin: 0;">Total Rows</p>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="color: #14B8A6; margin: 0; font-size: 2rem;">{len(df.columns)}</h3>
                    <p style="color: #6B7280; margin: 0;">Columns</p>
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
                    <h3 style="color: #EC4899; margin: 0; font-size: 2rem;">{len(df.select_dtypes(include=['object']).columns)}</h3>
                    <p style="color: #6B7280; margin: 0;">Text Columns</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Data Preview
            tab1, tab2, tab3 = st.tabs(["📊 Data Preview", "📈 Column Info", "🎯 Sample Data"])
            
            with tab1:
                st.markdown('<div class="canva-card">', unsafe_allow_html=True)
                st.dataframe(df.head(10), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with tab2:
                st.markdown('<div class="canva-card">', unsafe_allow_html=True)
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Data Type': df.dtypes.values,
                    'Non-Null Count': df.notnull().sum().values,
                    'Null Count': df.isnull().sum().values,
                    'Unique Values': [df[col].nunique() for col in df.columns]
                })
                st.dataframe(col_info, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with tab3:
                st.markdown('<div class="canva-card">', unsafe_allow_html=True)
                if len(df) > 0:
                    sample_idx = st.slider("Select sample row:", 0, len(df)-1, 0)
                    selected_col = st.selectbox("Select column to view:", df.columns)
                    st.text_area(
                        f"Sample text from row {sample_idx}:",
                        df.iloc[sample_idx][selected_col] if selected_col in df.columns else "No data",
                        height=150
                    )
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Text Analysis Configuration
            st.markdown('<div class="section-title">🔧 Text Analysis Configuration</div>', unsafe_allow_html=True)
            
            config_col1, config_col2 = st.columns(2)
            
            with config_col1:
                text_column = st.selectbox(
                    "Select text column for analysis:",
                    df.columns,
                    help="Choose the column containing text data"
                )
            
            with config_col2:
                analysis_type = st.selectbox(
                    "Select analysis type:",
                    ["Basic Text Analysis", "Sentiment Analysis", "Entity Extraction", "Comprehensive Analysis"],
                    help="Choose the type of analysis to perform"
                )
            
            # Perform Analysis
            if st.button("🚀 Perform Analysis", use_container_width=True):
                if text_column not in df.columns:
                    st.error("❌ Selected text column not found in dataset")
                    return
                
                # Clean text data
                df_clean = df.dropna(subset=[text_column]).copy()
                df_clean[text_column] = df_clean[text_column].astype(str)
                
                if len(df_clean) == 0:
                    st.error("❌ No valid text data found in selected column")
                    return
                
                # Show analysis progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Perform text analysis
                status_text.text("📊 Analyzing text data...")
                
                # Basic text statistics
                text_stats = df_clean[text_column].apply(TextAnalyzer.get_text_stats)
                stats_df = pd.json_normalize(text_stats.tolist())
                
                progress_bar.progress(25)
                
                # Sentiment analysis
                status_text.text("🎭 Analyzing sentiment...")
                sentiment_results = df_clean[text_column].apply(TextAnalyzer.analyze_sentiment)
                sentiment_df = pd.json_normalize(sentiment_results.tolist())
                
                progress_bar.progress(50)
                
                # Entity extraction
                status_text.text("🔍 Extracting entities...")
                entity_results = df_clean[text_column].apply(TextAnalyzer.extract_entities_simple)
                
                progress_bar.progress(75)
                
                # Combine results
                analysis_df = pd.concat([df_clean, stats_df, sentiment_df], axis=1)
                
                progress_bar.progress(100)
                status_text.text("✅ Analysis complete!")
                
                # Display Results
                st.markdown('<div class="section-title">📊 Analysis Results</div>', unsafe_allow_html=True)
                
                # Results in tabs
                results_tab1, results_tab2, results_tab3, results_tab4 = st.tabs([
                    "📈 Overview", "🎭 Sentiment", "📖 Text Stats", "🔍 Drill-down"
                ])
                
                with results_tab1:
                    st.markdown('<div class="canva-card">', unsafe_allow_html=True)
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        avg_words = stats_df['word_count'].mean()
                        st.metric("Average Word Count", f"{avg_words:.1f}")
                    
                    with col2:
                        avg_sentiment = sentiment_df['polarity'].mean()
                        st.metric("Average Sentiment", f"{avg_sentiment:.3f}")
                    
                    with col3:
                        unique_ratio = stats_df['unique_words'].sum() / stats_df['word_count'].sum()
                        st.metric("Vocabulary Diversity", f"{unique_ratio:.2%}")
                    
                    with col4:
                        positive_ratio = (sentiment_df['sentiment'] == 'positive').mean()
                        st.metric("Positive Texts", f"{positive_ratio:.1%}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with results_tab2:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Sentiment distribution
                        sentiment_chart = CanvaVisualizer.create_sentiment_chart(sentiment_df['sentiment'])
                        st.plotly_chart(sentiment_chart, use_container_width=True)
                    
                    with col2:
                        # Sentiment over text sequence
                        time_chart = CanvaVisualizer.create_sentiment_over_time(
                            range(len(sentiment_df)),
                            sentiment_df['polarity']
                        )
                        if time_chart:
                            st.plotly_chart(time_chart, use_container_width=True)
                        else:
                            st.info("Not enough data for time series analysis")
                
                with results_tab3:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Word length distribution
                        word_chart = CanvaVisualizer.create_word_length_chart(stats_df['word_count'])
                        st.plotly_chart(word_chart, use_container_width=True)
                    
                    with col2:
                        # Sentence length distribution
                        if 'sentence_count' in stats_df.columns:
                            sent_chart = px.histogram(
                                stats_df, 
                                x='sentence_count',
                                title="Sentence Count Distribution",
                                color_discrete_sequence=['#14B8A6']
                            )
                            st.plotly_chart(sent_chart, use_container_width=True)
                
                with results_tab4:
                    st.markdown('<div class="canva-card">', unsafe_allow_html=True)
                    st.subheader("🔍 Text Explorer")
                    
                    drill_down = DrillDownAnalyzer(df_clean, text_column)
                    
                    # Filter options
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        sentiment_filter = st.selectbox(
                            "Filter by sentiment:",
                            ["All", "positive", "negative", "neutral"]
                        )
                    
                    with col2:
                        min_words = st.number_input("Min words:", min_value=0, value=0)
                    
                    with col3:
                        max_words = st.number_input("Max words:", min_value=1, value=1000)
                    
                    # Get filtered samples
                    filtered_samples = drill_down.get_text_samples(
                        sentiment_filter if sentiment_filter != "All" else None,
                        min_words,
                        max_words
                    )
                    
                    # Display samples
                    if len(filtered_samples) > 0:
                        st.write(f"Found {len(filtered_samples)} samples:")
                        
                        for idx, sample in filtered_samples.iterrows():
                            with st.expander(f"Sample {idx} - {len(str(sample[text_column]).split())} words"):
                                st.text_area("Text:", sample[text_column], height=100, key=f"text_{idx}")
                                
                                # Show analysis for this sample
                                sample_stats = TextAnalyzer.get_text_stats(sample[text_column])
                                sample_sentiment = TextAnalyzer.analyze_sentiment(sample[text_column])
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Word Count", sample_stats['word_count'])
                                with col2:
                                    st.metric("Sentiment", sample_sentiment['sentiment'])
                                with col3:
                                    st.metric("Polarity", f"{sample_sentiment['polarity']:.3f}")
                    else:
                        st.info("No samples found matching the criteria")
                    
                    # Keyword search
                    st.subheader("🔎 Keyword Search")
                    keyword = st.text_input("Enter keyword to search:")
                    
                    if keyword:
                        keyword_matches = drill_down.find_texts_with_keyword(keyword, 3)
                        if len(keyword_matches) > 0:
                            st.write(f"Found {len(keyword_matches)} texts containing '{keyword}':")
                            
                            for idx, match in keyword_matches.iterrows():
                                text = str(match[text_column])
                                # Highlight keyword in text
                                highlighted_text = text.replace(
                                    keyword, 
                                    f"**{keyword}**"
                                )
                                st.markdown(f"**Text {idx}:** {highlighted_text}")
                                st.divider()
                        else:
                            st.info(f"No texts found containing '{keyword}'")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Download results
                st.markdown('<div class="canva-card">', unsafe_allow_html=True)
                st.subheader("📥 Export Results")
                
                # Create downloadable DataFrame
                export_df = analysis_df.copy()
                
                col1, col2 = st.columns(2)
                with col1:
                    csv = export_df.to_csv(index=False)
                    st.download_button(
                        label="📊 Download Analysis Results (CSV)",
                        data=csv,
                        file_name="nlp_analysis_results.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    # Summary statistics
                    summary_stats = {
                        'total_texts': len(analysis_df),
                        'avg_word_count': stats_df['word_count'].mean(),
                        'avg_sentiment': sentiment_df['polarity'].mean(),
                        'positive_ratio': (sentiment_df['sentiment'] == 'positive').mean(),
                        'negative_ratio': (sentiment_df['sentiment'] == 'negative').mean(),
                        'neutral_ratio': (sentiment_df['sentiment'] == 'neutral').mean()
                    }
                    
                    summary_df = pd.DataFrame([summary_stats])
                    summary_csv = summary_df.to_csv(index=False)
                    
                    st.download_button(
                        label="📈 Download Summary Stats (CSV)",
                        data=summary_csv,
                        file_name="nlp_summary_stats.csv",
                        mime="text/csv"
                    )
                st.markdown('</div>', unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("💡 Please ensure your CSV file is properly formatted and contains text data.")
    
    else:
        # Welcome Section
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
                <h3 style='color: #3B82F6;'>🎭 Sentiment Analysis</h3>
                <p style='color: #6B7280;'>Understand the emotional tone and polarity of your text data</p>
            </div>
            <div class='canva-card'>
                <h3 style='color: #EC4899;'>📈 Text Statistics</h3>
                <p style='color: #6B7280;'>Comprehensive analysis of word counts, readability, and more</p>
            </div>
        </div>
        
        <div class='canva-card'>
            <h3 style='color: #1F2937; text-align: center;'>📋 Sample Data Format</h3>
            <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; text-align: center;'>
                <div>
                    <h4 style='color: #8B5CF6;'>CSV Format</h4>
                    <p style='color: #6B7280;'>Comma-separated values with header row</p>
                </div>
                <div>
                    <h4 style='color: #14B8A6;'>Text Column</h4>
                    <p style='color: #6B7280;'>At least one column containing text data</p>
                </div>
                <div>
                    <h4 style='color: #3B82F6;'>Optional Metadata</h4>
                    <p style='color: #6B7280;'>Additional columns for filtering and analysis</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

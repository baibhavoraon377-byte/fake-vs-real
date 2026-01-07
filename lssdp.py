# lssdp.py
# Complete Streamlit app with simplified Google API verification display
import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import csv
from urllib.parse import urljoin
import time
import random
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from datetime import datetime, timedelta

# NLP & ML
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from textblob import TextBlob
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from scipy import sparse
import io
import os
from joblib import dump, load
import json
import pathlib

# --- Try to install spaCy model if not present ---
try:
    import subprocess
    import sys
    import spacy
    spacy.load("en_core_web_sm")
    print("spaCy model loaded successfully")
except OSError:
    print("Downloading spaCy model...")
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])

# --- Configuration ---
SCRAPED_DATA_PATH = 'politifact_data.csv'
N_SPLITS = 5
MODELS_DIR = pathlib.Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# Google Fact Check API rating mappings
GOOGLE_TRUE_RATINGS = ["True", "Mostly True", "Accurate", "Correct", "Factual", "Verified", "Real"]
GOOGLE_FALSE_RATINGS = ["False", "Mostly False", "Pants on Fire", "Pants on Fire!", "Fake", "Incorrect", "Baseless", "Misleading", "Unverified", "Debunked"]

# --- SpaCy Loading Function ---
@st.cache_resource
def load_spacy_model():
    try:
        nlp = spacy.load("en_core_web_sm")
        print("spaCy model loaded successfully via @st.cache_resource")
        return nlp
    except OSError as e:
        st.error("SpaCy model 'en_core_web_sm' not found.")
        try:
            import subprocess, sys
            st.info("Attempting to download spaCy model automatically...")
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm")
            return spacy.load("en_core_web_sm")
        except Exception as ee:
            st.error("Failed to download spaCy model. Please install manually:")
            st.code("python -m spacy download en_core_web_sm")
            return None

NLP_MODEL = load_spacy_model()
if NLP_MODEL is None:
    st.error("CRITICAL: Could not load spaCy model. The app cannot function properly.")
    st.stop()

stop_words = STOP_WORDS
pragmatic_words = ["must", "should", "might", "could", "will", "?", "!"]

# --------------------------
# GOOGLE API FUNCTIONS
# --------------------------
def get_google_api_key():
    """Get Google API key from secrets.toml"""
    try:
        # Try to load from secrets.toml in current directory
        if os.path.exists('secrets.toml'):
            import toml
            with open('secrets.toml', 'r') as f:
                secrets = toml.load(f)
            if 'GOOGLE_API_KEY' in secrets:
                return secrets['GOOGLE_API_KEY']
        
        # Try Streamlit secrets (for cloud deployment)
        if hasattr(st, 'secrets') and 'GOOGLE_API_KEY' in st.secrets:
            return st.secrets['GOOGLE_API_KEY']
        
        # Try environment variable
        import os
        if 'GOOGLE_API_KEY' in os.environ:
            return os.environ['GOOGLE_API_KEY']
            
    except Exception as e:
        st.error(f"Error loading API key: {e}")
    
    return None

def verify_single_claim_with_google(claim_text, max_results=3):
    """Verify a single claim with Google Fact Check API"""
    api_key = get_google_api_key()
    if not api_key:
        return {"error": "Google API Key not found. Please add it to secrets.toml"}
    
    base_url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    
    try:
        params = {
            'key': api_key,
            'query': claim_text[:300],
            'languageCode': 'en',
            'pageSize': max_results,
            'maxAgeDays': 365
        }
        
        response = requests.get(base_url, params=params, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            if 'claims' in data and data['claims']:
                best_match = None
                best_score = 0
                
                for claim_obj in data['claims']:
                    google_text = claim_obj.get('text', '')
                    if not google_text:
                        continue
                    
                    original_lower = claim_text.lower()
                    google_lower = google_text.lower()
                    
                    original_words = set(re.findall(r'\w+', original_lower))
                    google_words = set(re.findall(r'\w+', google_lower))
                    
                    if original_words and google_words:
                        overlap = len(original_words.intersection(google_words))
                        total = len(original_words.union(google_words))
                        word_score = overlap / total if total > 0 else 0
                        
                        len_similarity = 1 - abs(len(original_lower) - len(google_lower)) / max(len(original_lower), len(google_lower))
                        combined_score = (word_score * 0.7) + (len_similarity * 0.3)
                        
                        if combined_score > best_score:
                            best_score = combined_score
                            best_match = claim_obj
                
                if best_match and best_score > 0.3:
                    claim_reviews = best_match.get('claimReview', [])
                    if claim_reviews:
                        review = claim_reviews[0]
                        return {
                            'found': True,
                            'google_text': best_match.get('text', ''),
                            'google_rating': review.get('textualRating', ''),
                            'google_publisher': review.get('publisher', {}).get('name', ''),
                            'google_url': review.get('url', ''),
                            'match_score': best_score,
                            'confidence': 'High' if best_score >= 0.7 else 'Medium' if best_score >= 0.5 else 'Low'
                        }
        
        return {'found': False, 'message': 'No matching claim found in Google Fact Check database'}
        
    except Exception as e:
        return {"error": f"API request failed: {str(e)}"}

def verify_batch_claims_with_google(df, text_column='statement', max_claims=20, delay=0.5):
    """Verify multiple claims with Google API"""
    api_key = get_google_api_key()
    if not api_key:
        st.error("Google API Key not found in secrets.toml")
        return pd.DataFrame()
    
    if df.empty or text_column not in df.columns:
        st.error("No data to verify")
        return pd.DataFrame()
    
    sample_size = min(max_claims, len(df))
    df_sample = df.head(sample_size).copy()
    
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, (_, row) in enumerate(df_sample.iterrows()):
        status_text.text(f"Verifying claim {idx+1}/{sample_size}...")
        progress_bar.progress((idx + 1) / sample_size)
        
        claim_text = str(row[text_column])
        result = {
            'original_statement': claim_text,
            'original_label': row.get('label', 'Unknown'),
            'original_binary_label': row.get('binary_label', -1),
            'google_verification': 'Not Found',
            'google_rating': '',
            'google_publisher': '',
            'match_score': 0.0,
            'verification_url': '',
            'verification_confidence': 'Low'
        }
        
        api_result = verify_single_claim_with_google(claim_text)
        
        if 'error' not in api_result and api_result.get('found', False):
            result.update({
                'google_verification': 'Found',
                'google_rating': api_result['google_rating'],
                'google_publisher': api_result['google_publisher'],
                'match_score': api_result['match_score'],
                'verification_url': api_result.get('google_url', ''),
                'verification_confidence': api_result['confidence']
            })
        
        results.append(result)
        time.sleep(delay)
    
    progress_bar.empty()
    status_text.empty()
    
    if results:
        verification_df = pd.DataFrame(results)
        return verification_df
    else:
        return pd.DataFrame()

# --------------------------
# SIMPLIFIED VERIFICATION RESULTS DISPLAY
# --------------------------
def show_verification_results_simple(verification_df):
    """Display verification results in simplified format"""
    if verification_df.empty:
        st.warning("No verification results to display.")
        return
    
    st.subheader("Google API Verification Results")
    
    found_count = len(verification_df[verification_df['google_verification'] == 'Found'])
    total_count = len(verification_df)
    
    st.info(f"Summary: Found {found_count} out of {total_count} claims ({found_count/total_count*100:.1f}% verification rate)")
    
    for index, row in verification_df.iterrows():
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**Claim:** \"{row['original_statement'][:200]}{'...' if len(row['original_statement']) > 200 else ''}\"")
                
                original_label = str(row['original_label'])
                label_display = original_label.title() if len(original_label) < 20 else original_label[:20] + "..."
                st.markdown(f"**Your label:** {label_display}")
                
                if row['google_verification'] == 'Found':
                    google_rating = row['google_rating']
                    google_publisher = row['google_publisher']
                    st.markdown(f"**Google says:** {google_rating.upper()} (from {google_publisher})")
                else:
                    st.markdown("**Google says:** NOT FOUND (No match in Google Fact Check database)")
            
            with col2:
                if row['google_verification'] == 'Found':
                    result_text = compare_labels_simple(row['original_label'], row['google_rating'])
                    if "CORRECT" in result_text:
                        st.success(f"{result_text}")
                    elif "WRONG" in result_text:
                        st.error(f"{result_text}")
                    else:
                        st.info(f"{result_text}")
                else:
                    st.warning("UNVERIFIED")
            
            st.markdown("---")

def compare_labels_simple(your_label, google_rating):
    """Compare your label with Google's rating and return simple result"""
    your_label_lower = str(your_label).lower().strip()
    google_rating_lower = str(google_rating).lower().strip()
    
    true_indicators = ['true', 'mostly true', 'accurate', 'correct', 'fact', 'real', '1', 'yes']
    false_indicators = ['false', 'mostly false', 'pants on fire', 'fake', 'incorrect', 'baseless', '0', 'no']
    
    your_is_true = any(indicator in your_label_lower for indicator in true_indicators)
    your_is_false = any(indicator in your_label_lower for indicator in false_indicators)
    google_is_true = any(indicator in google_rating_lower for indicator in true_indicators)
    google_is_false = any(indicator in google_rating_lower for indicator in false_indicators)
    
    if your_is_true and google_is_true:
        return "CORRECT"
    elif your_is_false and google_is_false:
        return "CORRECT"
    elif your_is_true and google_is_false:
        return "WRONG"
    elif your_is_false and google_is_true:
        return "WRONG"
    elif not (your_is_true or your_is_false):
        return "UNKNOWN LABEL"
    elif not (google_is_true or google_is_false):
        return "AMBIGUOUS RATING"
    else:
        return "CHECK NEEDED"

# --------------------------
# DEMO DATA
# --------------------------
def get_demo_google_claims():
    demo_claims = [
        {'claim_text': 'The earth is flat and NASA is hiding the truth from us.', 'rating': 'False'},
        {'claim_text': 'Vaccines are completely safe and effective for 95% of the population.', 'rating': 'Mostly True'},
        {'claim_text': 'The moon landing was filmed in a Hollywood studio in 1969.', 'rating': 'False'},
        {'claim_text': 'Climate change is primarily caused by human activities and carbon emissions.', 'rating': 'True'},
        {'claim_text': 'You can cure COVID-19 by drinking bleach and taking horse medication.', 'rating': 'False'},
    ]
    return demo_claims

# --------------------------
# CSV PROCESSING FUNCTIONS
# --------------------------
def create_binary_label_column(df, label_column):
    """Automatically create binary labels (0/1) from various label formats"""
    if df.empty or label_column not in df.columns:
        return df
    
    df = df.copy()
    
    TRUE_LABELS = ["true", "mostly true", "accurate", "correct", "real", "fact", "1", "yes", "y"]
    FALSE_LABELS = ["false", "mostly false", "pants on fire", "fake", "incorrect", "baseless", "misleading", "0", "no", "n"]
    
    df['binary_label'] = -1
    
    for idx, row in df.iterrows():
        label = str(row[label_column]).lower().strip()
        
        if any(true_label in label for true_label in TRUE_LABELS):
            df.at[idx, 'binary_label'] = 1
        elif any(false_label in label for false_label in FALSE_LABELS):
            df.at[idx, 'binary_label'] = 0
    
    true_count = len(df[df['binary_label'] == 1])
    false_count = len(df[df['binary_label'] == 0])
    unknown_count = len(df[df['binary_label'] == -1])
    
    if true_count + false_count > 0:
        st.success(f"Created binary labels: {true_count} True (1), {false_count} False (0), {unknown_count} Unknown")
        df['label_readable'] = df['binary_label'].map({1: 'True', 0: 'False', -1: 'Unknown'})
    else:
        st.warning("Could not automatically create binary labels from the selected column.")
    
    return df

def filter_by_date_range(df, date_column, start_date, end_date):
    """Filter DataFrame by date range"""
    if df.empty or date_column not in df.columns:
        return df
    
    df = df.copy()
    
    try:
        df['date_parsed'] = pd.to_datetime(df[date_column], errors='coerce')
        mask = (df['date_parsed'] >= pd.Timestamp(start_date)) & (df['date_parsed'] <= pd.Timestamp(end_date))
        filtered_df = df[mask].copy()
        st.info(f"Filtered to {len(filtered_df)} claims from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        return filtered_df
    except Exception as e:
        st.error(f"Error parsing dates: {e}")
        return df

def get_politifact_csv_template():
    """Generate a template CSV structure for Politifact data"""
    template = {
        'statement': 'The claim text goes here',
        'rating': 'False',
        'binary_rating': 0,
        'date': '2023-01-15',
        'source': 'Politifact',
        'author': 'Author Name',
        'url': 'https://politifact.com/factchecks/...'
    }
    return pd.DataFrame([template])

# --------------------------
# FEATURE EXTRACTION
# --------------------------
def lexical_features(text):
    doc = NLP_MODEL(text.lower())
    tokens = [token.lemma_ for token in doc if token.text not in stop_words and token.is_alpha]
    return " ".join(tokens)

def syntactic_features(text):
    doc = NLP_MODEL(text)
    pos_tags = " ".join([token.pos_ for token in doc])
    return pos_tags

def semantic_features(text):
    blob = TextBlob(text)
    return [blob.sentiment.polarity, blob.sentiment.subjectivity]

def discourse_features(text):
    doc = NLP_MODEL(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    return f"{len(sentences)} {' '.join([s.split()[0].lower() for s in sentences if len(s.split()) > 0])}"

def pragmatic_features(text):
    text = text.lower()
    return [text.count(w) for w in pragmatic_words]

# --------------------------
# MODEL TRAINING FUNCTIONS
# --------------------------
def apply_feature_extraction(X, phase, vectorizer=None):
    try:
        if phase == "Lexical & Morphological":
            X_processed = X.apply(lexical_features)
            vectorizer = vectorizer if vectorizer else CountVectorizer(binary=True, ngram_range=(1,2), max_features=1000)
            X_features = vectorizer.fit_transform(X_processed)
            return X_features, vectorizer
        elif phase == "Syntactic":
            X_processed = X.apply(syntactic_features)
            vectorizer = vectorizer if vectorizer else TfidfVectorizer(max_features=500)
            X_features = vectorizer.fit_transform(X_processed)
            return X_features, vectorizer
        elif phase == "Semantic":
            X_features = pd.DataFrame(X.apply(semantic_features).tolist(), columns=["polarity", "subjectivity"])
            return X_features.values, None
        elif phase == "Discourse":
            X_processed = X.apply(discourse_features)
            vectorizer = vectorizer if vectorizer else CountVectorizer(ngram_range=(1,2), max_features=500)
            X_features = vectorizer.fit_transform(X_processed)
            return X_features, vectorizer
        elif phase == "Pragmatic":
            X_features = pd.DataFrame(X.apply(pragmatic_features).tolist(), columns=pragmatic_words)
            return X_features.values, None
        return None, None
    except Exception as e:
        st.error(f"Error in apply_feature_extraction: {e}")
        return None, None

def evaluate_models(df: pd.DataFrame, selected_phase: str, text_column: str = 'statement', label_column: str = 'label', use_smote: bool = True):
    try:
        if df.empty:
            st.error("DataFrame is empty! Please load data first.")
            return pd.DataFrame(), {}, None
        
        st.info(f"Data shape: {df.shape[0]} rows, {df.shape[1]} columns")
        
        if text_column not in df.columns:
            st.error(f"Text column '{text_column}' not found in data!")
            st.info(f"Available columns: {list(df.columns)}")
            return pd.DataFrame(), {}, None
        
        if 'binary_label' not in df.columns:
            st.warning("'binary_label' column not found. Creating binary labels...")
            df = create_binary_label_column(df, label_column)
        
        df_clean = df[df['binary_label'] != -1].copy()
        if df_clean.empty:
            st.error("No valid binary labels (0 or 1) found after filtering!")
            return pd.DataFrame(), {}, None
        
        class_counts = df_clean['binary_label'].value_counts()
        st.info(f"Class distribution: {dict(class_counts)}")
        
        if len(class_counts) < 2:
            st.error(f"Only one class found ({class_counts.index[0]}). Need at least 2 classes for training.")
            return pd.DataFrame(), {}, None
        
        X_raw = df_clean[text_column].astype(str)
        y = df_clean['binary_label'].values.astype(int)
        
        st.info(f"Extracting {selected_phase} features...")
        X_features, vectorizer = apply_feature_extraction(X_raw, selected_phase)
        
        if X_features is None:
            st.error("Feature extraction failed!")
            return pd.DataFrame(), {}, None
        
        models_config = {
            "Naive Bayes": MultinomialNB(alpha=0.1),
            "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, solver='liblinear'),
            "SVM": SVC(kernel='linear', random_state=42, probability=True, max_iter=1000)
        }
        
        results = []
        trained_models_final = {}
        
        for name, model in models_config.items():
            st.info(f"Training {name}...")
            
            try:
                start_time = time.time()
                model.fit(X_features, y)
                train_time = time.time() - start_time
                
                y_pred = model.predict(X_features)
                
                accuracy = accuracy_score(y, y_pred) * 100
                f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
                precision = precision_score(y, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y, y_pred, average='weighted', zero_division=0)
                
                results.append({
                    'Model': name,
                    'Accuracy': round(accuracy, 2),
                    'F1-Score': round(f1, 4),
                    'Precision': round(precision, 4),
                    'Recall': round(recall, 4),
                    'Training Time (s)': round(train_time, 2),
                    'Inference Latency (ms)': 0
                })
                
                trained_models_final[name] = model
                st.success(f"{name} trained successfully! Accuracy: {accuracy:.2f}%")
                
            except Exception as e:
                st.error(f"Failed to train {name}: {str(e)[:200]}")
                results.append({
                    'Model': name,
                    'Accuracy': 0,
                    'F1-Score': 0,
                    'Precision': 0,
                    'Recall': 0,
                    'Training Time (s)': 0,
                    'Inference Latency (ms)': 0
                })
                trained_models_final[name] = None
        
        df_results = pd.DataFrame(results)
        
        try:
            save_trained_models(trained_models_final, vectorizer, selected_phase)
        except Exception as e:
            st.warning(f"Could not save models: {e}")
        
        return df_results, trained_models_final, vectorizer
        
    except Exception as e:
        st.error(f"CRITICAL ERROR in evaluate_models: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return pd.DataFrame(), {}, None

def save_trained_models(trained_models: dict, vectorizer, selected_phase):
    try:
        MODELS_DIR.mkdir(exist_ok=True)
        for name, model in trained_models.items():
            if model is not None:
                path = MODELS_DIR / f"{name.replace(' ', '_')}.joblib"
                try:
                    dump(model, path)
                except Exception as e:
                    st.warning(f"Failed to save {name}: {e}")
        if vectorizer is not None:
            try:
                dump(vectorizer, MODELS_DIR / "vectorizer.joblib")
            except Exception as e:
                st.warning(f"Failed to save vectorizer: {e}")
        with open(MODELS_DIR / "metadata.json", "w") as f:
            json.dump({"selected_phase": selected_phase}, f)
        st.success("Trained models and vectorizer saved to disk.")
    except Exception as e:
        st.error(f"Saving models failed: {e}")

def load_trained_models(expected_model_names=None):
    trained_models = {}
    vectorizer = None
    selected_phase = None
    try:
        meta_path = MODELS_DIR / "metadata.json"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                meta = json.load(f)
                selected_phase = meta.get("selected_phase")
        vec_path = MODELS_DIR / "vectorizer.joblib"
        if vec_path.exists():
            try:
                vectorizer = load(vec_path)
            except Exception:
                vectorizer = None
        if expected_model_names:
            names = expected_model_names
        else:
            names = []
            for p in MODELS_DIR.glob("*.joblib"):
                if p.name != "vectorizer.joblib":
                    names.append(p.stem)
        for name in names:
            filename = (MODELS_DIR / f"{name.replace(' ', '_')}.joblib")
            if not filename.exists():
                filename = MODELS_DIR / f"{name}.joblib"
            if filename.exists():
                try:
                    loaded = load(filename)
                    trained_models[name] = loaded
                except Exception as e:
                    st.warning(f"Failed to load model {name}: {e}")
                    trained_models[name] = None
            else:
                trained_models[name] = None
        return trained_models, vectorizer, selected_phase
    except Exception as e:
        st.warning(f"Loading trained models failed: {e}")
        return {}, None, None

# --------------------------
# SINGLE CLAIM VERIFICATION
# --------------------------
def verify_single_claim_ui():
    """UI for verifying a single claim with Google API"""
    st.subheader("Verify a Single Claim")
    
    claim_text = st.text_area(
        "Enter a claim to verify:",
        placeholder="e.g., 'Eating chocolate helps weight loss'",
        height=100,
        key="single_claim_input"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        user_label = st.selectbox(
            "Your label for this claim:",
            ["True", "False", "Mostly True", "Mostly False", "Unknown"],
            key="user_label_select"
        )
    
    if st.button("Verify with Google API", key="verify_single_btn", type="primary"):
        if not claim_text.strip():
            st.error("Please enter a claim to verify")
            return
        
        api_key = get_google_api_key()
        if not api_key:
            st.error("Google API Key not found. Please add it to secrets.toml")
            return
        
        with st.spinner("Verifying claim with Google Fact Check API..."):
            result = verify_single_claim_with_google(claim_text)
            
            if 'error' in result:
                st.error(f"Error: {result['error']}")
            else:
                st.markdown("---")
                st.subheader("Verification Results")
                
                with st.container():
                    st.markdown(f"**Claim:** \"{claim_text}\"")
                    st.markdown(f"**Your label:** {user_label}")
                    
                    if result.get('found', False):
                        google_rating = result['google_rating']
                        google_publisher = result['google_publisher']
                        st.markdown(f"**Google says:** {google_rating.upper()} (from {google_publisher})")
                        
                        result_text = compare_labels_simple(user_label, google_rating)
                        
                        if result_text == "CORRECT":
                            st.success(f"**Result:** {result_text}")
                        elif result_text == "WRONG":
                            st.error(f"**Result:** {result_text}")
                        else:
                            st.info(f"**Result:** {result_text}")
                    else:
                        st.markdown("**Google says:** NOT FOUND")
                        st.warning("**Result:** UNVERIFIED")

# --------------------------
# BATCH VERIFICATION
# --------------------------
def batch_verification_ui():
    """UI for batch verification of claims"""
    st.subheader("Batch Verify Your Data")
    
    if 'scraped_df' not in st.session_state or st.session_state['scraped_df'].empty:
        st.warning("Please load data first in the Data Collection section.")
        return
    
    data_to_verify = st.session_state.get('filtered_df', st.session_state['scraped_df'])
    
    st.info(f"Data ready for verification: {len(data_to_verify)} claims available")
    
    col1, col2 = st.columns(2)
    with col1:
        max_claims = st.number_input(
            "Max claims to verify:",
            min_value=1,
            max_value=50,
            value=10,
            step=5
        )
    
    with col2:
        text_column = st.selectbox(
            "Column containing claim text:",
            data_to_verify.columns.tolist(),
            index=data_to_verify.columns.tolist().index('statement') if 'statement' in data_to_verify.columns else 0
        )
    
    if st.button("Run Batch Verification", key="batch_verify_btn", type="primary"):
        api_key = get_google_api_key()
        if not api_key:
            st.error("Google API Key not found. Please add it to secrets.toml")
            return
        
        with st.spinner(f"Verifying up to {max_claims} claims with Google API..."):
            verification_df = verify_batch_claims_with_google(
                data_to_verify, 
                text_column=text_column,
                max_claims=max_claims
            )
            
            if not verification_df.empty:
                st.session_state['verification_df'] = verification_df
                st.session_state['verification_performed'] = True
                st.success(f"Verification complete! Processed {len(verification_df)} claims.")
                
                show_verification_results_simple(verification_df)
            else:
                st.error("Verification failed or returned no results.")

# --------------------------
# STREAMLIT APP MAIN FUNCTION
# --------------------------
def app():
    st.set_page_config(
        page_title='FactChecker: AI Fact-Checking Platform', 
        layout='wide', 
        initial_sidebar_state='expanded'
    )
    
    # Simple CSS
    st.markdown("""
    <style>
    .main-header {
        color: #1e88e5;
        padding-bottom: 10px;
        border-bottom: 2px solid #bbdefb;
    }
    .stButton button {
        background-color: #1e88e5;
        color: white;
        font-weight: 500;
    }
    .stButton button:hover {
        background-color: #0d47a1;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    session_vars = [
        'scraped_df', 'df_results', 'trained_models', 'trained_vectorizer',
        'google_benchmark_results', 'google_df', 'selected_phase_run',
        'use_smote', 'verification_df', 'filtered_verification_df',
        'selected_text_column', 'selected_label_column', 'selected_date_column',
        'csv_columns_selected', 'filtered_df', 'verification_performed',
        'models_loaded_attempted'
    ]
    
    for var in session_vars:
        if var not in st.session_state:
            if 'df' in var:
                st.session_state[var] = pd.DataFrame()
            elif 'models' in var:
                st.session_state[var] = {}
            elif 'vectorizer' in var:
                st.session_state[var] = None
            elif 'phase' in var:
                st.session_state[var] = None
            elif 'smote' in var:
                st.session_state[var] = True
            elif 'selected' in var:
                st.session_state[var] = None
            elif 'columns_selected' in var:
                st.session_state[var] = False
            elif 'performed' in var:
                st.session_state[var] = False
            elif 'loaded_attempted' in var:
                st.session_state[var] = False
    
    # Load models on startup
    if not st.session_state['models_loaded_attempted']:
        trained_models_loaded, vec_loaded, selected_phase_loaded = load_trained_models(
            expected_model_names=["Naive Bayes","Decision Tree","Logistic Regression","SVM"]
        )
        any_model_loaded = any(v is not None for v in trained_models_loaded.values()) if trained_models_loaded else False
        if any_model_loaded:
            st.session_state['trained_models'] = trained_models_loaded
            st.session_state['trained_vectorizer'] = vec_loaded
            if selected_phase_loaded:
                st.session_state['selected_phase_run'] = selected_phase_loaded
        st.session_state['models_loaded_attempted'] = True
    
    # Sidebar navigation
    st.sidebar.markdown("<h1 style='text-align: center;'>FactChecker</h1>", unsafe_allow_html=True)
    page = st.sidebar.radio(
        "Navigation", 
        ["Dashboard", "Data Collection", "Model Training", "Google API Verification", "Benchmark Testing", "Results & Analysis"],
        key='navigation'
    )
    
    # Dashboard
    if page == "Dashboard":
        st.markdown("<h1 class='main-header'>FactChecker Dashboard</h1>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_claims = len(st.session_state.get('filtered_df', pd.DataFrame()))
            if total_claims == 0:
                total_claims = len(st.session_state['scraped_df']) if not st.session_state['scraped_df'].empty else 0
            st.metric("Data Claims", total_claims)
        
        with col2:
            trained_model_count = sum(1 for m in st.session_state['trained_models'].values() if m is not None)
            st.metric("Trained Models", trained_model_count)
        
        with col3:
            st.metric("API Status", "Ready" if get_google_api_key() else "Missing")
        
        with col4:
            verification_count = 0
            if not st.session_state.get('verification_df', pd.DataFrame()).empty:
                verification_count = len(st.session_state['verification_df'])
            st.metric("Verified Claims", verification_count)
    
    # Data Collection
    elif page == "Data Collection":
        st.markdown("<h1 class='main-header'>Data Collection</h1>", unsafe_allow_html=True)
        
        st.info("Upload your CSV data in the format: 'statement', 'label', (optional: 'date')")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"Successfully loaded {len(df)} claims")
                
                st.dataframe(df.head(), use_container_width=True)
                
                st.session_state['scraped_df'] = df
                st.session_state['filtered_df'] = df.copy()
                st.session_state['csv_columns_selected'] = True
                
            except Exception as e:
                st.error(f"Error loading file: {e}")
    
    # Model Training
    elif page == "Model Training":
        st.markdown("<h1 class='main-header'>Model Training</h1>", unsafe_allow_html=True)
        
        if st.session_state['scraped_df'].empty:
            st.warning("Please load data first in Data Collection section.")
        else:
            phases = ["Lexical & Morphological", "Syntactic", "Semantic", "Discourse", "Pragmatic"]
            selected_phase = st.selectbox("Feature Extraction Method:", phases)
            
            if st.button("Train Models", type="primary"):
                with st.spinner("Training models..."):
                    df_results, trained_models, vectorizer = evaluate_models(
                        st.session_state['scraped_df'], 
                        selected_phase
                    )
                    
                    if not df_results.empty:
                        st.session_state['df_results'] = df_results
                        st.session_state['trained_models'] = trained_models
                        st.session_state['trained_vectorizer'] = vectorizer
                        st.session_state['selected_phase_run'] = selected_phase
                        
                        st.success("Training complete!")
                        st.dataframe(df_results, use_container_width=True)
    
    # Google API Verification Section
    elif page == "Google API Verification":
        st.markdown("<h1 class='main-header'>Google API Verification</h1>", unsafe_allow_html=True)
        
        api_key = get_google_api_key()
        if not api_key:
            st.error("""
            Google API Key not found!
            
            Please create a `secrets.toml` file in your project directory with:
            ```toml
            GOOGLE_API_KEY = "AIzaSyBOl5iIVor1xf1yz8SA0vDBAt7V6_nUV8M"
            ```
            
            Then restart the Streamlit app.
            """)
            return
        
        st.success("Google API Key found and ready to use!")
        
        tab1, tab2, tab3 = st.tabs(["Single Claim", "Batch Verify", "View Results"])
        
        with tab1:
            verify_single_claim_ui()
        
        with tab2:
            batch_verification_ui()
        
        with tab3:
            if 'verification_df' in st.session_state and not st.session_state['verification_df'].empty:
                st.subheader("Previous Verification Results")
                show_verification_results_simple(st.session_state['verification_df'])
            else:
                st.info("No verification results available. Run a verification first.")
    
    # Benchmark Testing
    elif page == "Benchmark Testing":
        st.markdown("<h1 class='main-header'>Benchmark Testing</h1>", unsafe_allow_html=True)
        
        if not st.session_state['trained_models']:
            st.warning("Please train models first in the Model Training section.")
        else:
            st.write("Benchmark your trained models against external data sources.")
    
    # Results & Analysis
    elif page == "Results & Analysis":
        st.markdown("<h1 class='main-header'>Results & Analysis</h1>", unsafe_allow_html=True)
        
        if st.session_state['df_results'].empty:
            st.warning("No training results found. Please train models first.")
        else:
            st.dataframe(st.session_state['df_results'], use_container_width=True)
            
            if 'verification_df' in st.session_state and not st.session_state['verification_df'].empty:
                st.markdown("---")
                st.subheader("Verification Results Summary")
                
                verification_df = st.session_state['verification_df']
                found_df = verification_df[verification_df['google_verification'] == 'Found']
                
                if not found_df.empty:
                    matches = 0
                    total_comparable = 0
                    
                    for _, row in found_df.iterrows():
                        your_label = str(row['original_label']).lower()
                        google_rating = str(row['google_rating']).lower()
                        
                        your_is_true = any(ind in your_label for ind in ['true', 'mostly true', 'accurate'])
                        your_is_false = any(ind in your_label for ind in ['false', 'mostly false', 'fake'])
                        google_is_true = any(ind in google_rating for ind in ['true', 'mostly true', 'accurate'])
                        google_is_false = any(ind in google_rating for ind in ['false', 'mostly false', 'fake'])
                        
                        if (your_is_true and google_is_true) or (your_is_false and google_is_false):
                            matches += 1
                        if your_is_true or your_is_false:
                            total_comparable += 1
                    
                    if total_comparable > 0:
                        accuracy = matches / total_comparable * 100
                        st.metric("Google Verification Accuracy", f"{accuracy:.1f}%")

if __name__ == '__main__':
    app()

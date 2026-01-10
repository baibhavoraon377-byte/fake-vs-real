# lssdp.py
# Updated Streamlit app with Google API verification via secrets.toml
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

# --- Configuration ---
SCRAPED_DATA_PATH = 'politifact_data.csv'
N_SPLITS = 5
MODELS_DIR = pathlib.Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# Google Fact Check API rating mappings (binary)
GOOGLE_TRUE_RATINGS = ["True", "Mostly True", "Accurate", "Correct"]
GOOGLE_FALSE_RATINGS = ["False", "Mostly False", "Pants on Fire", "Pants on Fire!", "Fake", "Incorrect", "Baseless", "Misleading"]

# --- SpaCy Loading Function ---
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError as e:
        st.error("SpaCy model 'en_core_web_sm' not found. Ensure it's in requirements (or add the wheel URL) and that the model is installed.")
        st.code("""
# Example to place in requirements.txt:
https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
""", language='text')
        # try to auto-download
        try:
            import subprocess, sys
            st.info("Attempting to download spaCy model automatically...")
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            return spacy.load("en_core_web_sm")
        except Exception as ee:
            st.error("Automatic spaCy model download failed. Please install manually.")
            raise e

NLP_MODEL = None
try:
    NLP_MODEL = load_spacy_model()
except Exception:
    # If model couldn't be loaded, stop the app (it won't function properly).
    st.stop()

stop_words = STOP_WORDS
pragmatic_words = ["must", "should", "might", "could", "will", "?", "!"]

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
# GOOGLE FACT CHECK INTEGRATION - DEDICATED VERIFICATION FUNCTIONS
# --------------------------
def fetch_google_claims(api_key, num_claims=100):
    base_url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    collected_claims = []
    page_token = None
    placeholder = st.empty()

    try:
        while len(collected_claims) < num_claims:
            params = {
                'key': api_key,
                'languageCode': 'en',
                'pageSize': min(100, num_claims - len(collected_claims))
            }
            if page_token:
                params['pageToken'] = page_token
            placeholder.text(f"Fetching Google claims... {len(collected_claims)} collected so far")
            response = requests.get(base_url, params=params, timeout=15)
            if response.status_code == 401:
                st.error("Invalid API key. Please check your GOOGLE_API_KEY in secrets.toml")
                return []
            elif response.status_code == 403:
                st.error("API access forbidden. Ensure 'Fact Check Tools API' is enabled in Google Cloud Console.")
                return []
            elif response.status_code == 429:
                st.error("API rate limit exceeded. Please try again later with fewer claims.")
                return []
            response.raise_for_status()
            data = response.json()
            if 'claims' not in data or not data['claims']:
                placeholder.success(f"Fetched {len(collected_claims)} claims (no more available)")
                break
            for claim_obj in data['claims']:
                if len(collected_claims) >= num_claims:
                    break
                claim_text = claim_obj.get('text', '')
                claim_reviews = claim_obj.get('claimReview', [])
                if not claim_reviews:
                    continue
                textual_rating = claim_reviews[0].get('textualRating', '')
                if not claim_text or not textual_rating:
                    continue
                collected_claims.append({'claim_text': claim_text, 'rating': textual_rating})
            page_token = data.get('nextPageToken')
            if not page_token:
                placeholder.success(f"Fetched {len(collected_claims)} claims (all pages processed)")
                break
        placeholder.success(f"Successfully fetched {len(collected_claims)} claims from Google Fact Check API")
        return collected_claims
    except requests.exceptions.RequestException as e:
        placeholder.error(f"Network error while fetching Google claims: {e}")
        return collected_claims if collected_claims else []
    except Exception as e:
        placeholder.error(f"Error processing Google API response: {e}")
        return collected_claims if collected_claims else []

def process_and_map_google_claims(api_results):
    if not api_results:
        return pd.DataFrame(columns=['claim_text', 'ground_truth'])
    processed_claims = []
    true_count = 0
    false_count = 0
    discarded_count = 0
    for claim_data in api_results:
        claim_text = claim_data.get('claim_text', '').strip()
        rating = claim_data.get('rating', '').strip()
        if not claim_text or len(claim_text) < 10:
            discarded_count += 1
            continue
        if not rating:
            discarded_count += 1
            continue
        rating_normalized = rating.lower().strip().rstrip('!').rstrip('?')
        is_true = any(rating_normalized == r.lower() for r in GOOGLE_TRUE_RATINGS)
        is_false = any(rating_normalized == r.lower() for r in GOOGLE_FALSE_RATINGS)
        if is_true:
            processed_claims.append({'claim_text': claim_text, 'ground_truth': 1})
            true_count += 1
        elif is_false:
            processed_claims.append({'claim_text': claim_text, 'ground_truth': 0})
            false_count += 1
        else:
            discarded_count += 1
    google_df = pd.DataFrame(processed_claims)
    if not google_df.empty:
        google_df = google_df.drop_duplicates(subset=['claim_text'], keep='first')
    total_processed = len(api_results)
    st.info(f"Processed {total_processed} claims: {true_count} True, {false_count} False, {discarded_count} ambiguous (discarded)")
    if not google_df.empty and len(google_df['ground_truth'].unique()) < 2:
        st.warning("Only one class found in processed claims. Results may not be meaningful.")
    return google_df

# --------------------------
# DEDICATED GOOGLE API VERIFICATION FUNCTIONS
# --------------------------
def verify_single_claim_with_google(claim_text, api_key):
    """
    Verify a single claim using Google Fact Check API
    """
    base_url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    
    result = {
        'claim_text': claim_text,
        'verification_status': 'Not Found',
        'google_rating': '',
        'google_rating_normalized': '',
        'match_score': 0.0,
        'verification_url': '',
        'google_claim_text': '',
        'google_publisher': '',
        'google_claim_date': '',
        'google_review_date': '',
        'verification_confidence': 'Low',
        'google_binary_label': -1
    }
    
    try:
        params = {
            'key': api_key,
            'query': claim_text[:300],
            'languageCode': 'en',
            'pageSize': 3,
            'maxAgeDays': 365
        }
        
        response = requests.get(base_url, params=params, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            if 'claims' in data and data['claims']:
                best_match = None
                best_score = 0
                
                for claim_obj in data['claims'][:3]:
                    google_text = claim_obj.get('text', '')
                    if not google_text:
                        continue
                    
                    # Calculate similarity
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
                        result['verification_status'] = 'Found'
                        result['google_rating'] = review.get('textualRating', '')
                        result['google_rating_normalized'] = result['google_rating'].lower().strip().rstrip('!').rstrip('?')
                        result['match_score'] = best_score
                        result['verification_url'] = review.get('url', '')
                        result['google_claim_text'] = best_match.get('text', '')
                        result['google_publisher'] = review.get('publisher', {}).get('name', '')
                        result['google_claim_date'] = best_match.get('claimDate', '')
                        result['google_review_date'] = review.get('reviewDate', '')
                        
                        if best_score >= 0.7:
                            result['verification_confidence'] = 'High'
                        elif best_score >= 0.5:
                            result['verification_confidence'] = 'Medium'
                        
                        # Determine binary label
                        rating_normalized = result['google_rating_normalized']
                        is_true = any(rating_normalized == r.lower() for r in GOOGLE_TRUE_RATINGS)
                        is_false = any(rating_normalized == r.lower() for r in GOOGLE_FALSE_RATINGS)
                        
                        if is_true:
                            result['google_binary_label'] = 1
                        elif is_false:
                            result['google_binary_label'] = 0
        elif response.status_code == 429:
            result['verification_status'] = 'Rate Limited'
            
    except Exception as e:
        result['verification_status'] = f'Error: {str(e)[:100]}'
    
    return result

def verify_batch_claims_with_google(df, api_key, max_verifications=50, text_column='statement'):
    """
    Verify a batch of claims from DataFrame using Google Fact Check API
    """
    if df.empty or text_column not in df.columns:
        return pd.DataFrame()
    
    verified_claims = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    sample_size = min(max_verifications, len(df))
    df_sample = df.head(sample_size).copy()
    
    for idx, (_, row) in enumerate(df_sample.iterrows()):
        status_text.text(f"Verifying claim {idx+1}/{sample_size}...")
        progress_bar.progress((idx + 1) / sample_size)
        
        claim_text = str(row[text_column])
        result = {
            'original_statement': claim_text,
            'original_label': row.get('label', ''),
            'original_binary_label': row.get('binary_label', ''),
            'original_date': row.get('date', ''),
            'original_source': row.get('source', ''),
            'google_verification': 'Not Found',
            'google_rating': '',
            'google_rating_normalized': '',
            'match_score': 0.0,
            'verification_url': '',
            'google_claim_text': '',
            'google_publisher': '',
            'google_claim_date': '',
            'google_review_date': '',
            'verification_confidence': 'Low',
            'google_binary_label': -1
        }
        
        try:
            verification_result = verify_single_claim_with_google(claim_text, api_key)
            
            if verification_result['verification_status'] == 'Found':
                result.update({
                    'google_verification': 'Found',
                    'google_rating': verification_result['google_rating'],
                    'google_rating_normalized': verification_result['google_rating_normalized'],
                    'match_score': verification_result['match_score'],
                    'verification_url': verification_result['verification_url'],
                    'google_claim_text': verification_result['google_claim_text'],
                    'google_publisher': verification_result['google_publisher'],
                    'google_claim_date': verification_result['google_claim_date'],
                    'google_review_date': verification_result['google_review_date'],
                    'verification_confidence': verification_result['verification_confidence'],
                    'google_binary_label': verification_result['google_binary_label']
                })
        except Exception as e:
            if idx % 10 == 0:
                st.warning(f"Error verifying claim {idx+1}: {str(e)[:100]}")
        
        verified_claims.append(result)
        time.sleep(0.3)  # Rate limiting
    
    progress_bar.empty()
    status_text.empty()
    
    if verified_claims:
        verification_df = pd.DataFrame(verified_claims)
        return verification_df
    else:
        return pd.DataFrame()

def run_google_benchmark(google_df, trained_models, vectorizer, selected_phase):
    if google_df.empty:
        st.error("No Google claims available for benchmarking.")
        return pd.DataFrame()
    X_raw = google_df['claim_text']
    y_true = google_df['ground_truth'].values
    try:
        if selected_phase == "Lexical & Morphological":
            X_processed = X_raw.apply(lexical_features)
            if vectorizer is None:
                st.error("Vectorizer not found for Lexical phase. Please retrain models.")
                return pd.DataFrame()
            X_features = vectorizer.transform(X_processed)
        elif selected_phase == "Syntactic":
            X_processed = X_raw.apply(syntactic_features)
            if vectorizer is None:
                st.error("Vectorizer not found for Syntactic phase. Please retrain models.")
                return pd.DataFrame()
            X_features = vectorizer.transform(X_processed)
        elif selected_phase == "Discourse":
            X_processed = X_raw.apply(discourse_features)
            if vectorizer is None:
                st.error("Vectorizer not found for Discourse phase. Please retrain models.")
                return pd.DataFrame()
            X_features = vectorizer.transform(X_processed)
        elif selected_phase == "Semantic":
            X_features = pd.DataFrame(X_raw.apply(semantic_features).tolist(), columns=["polarity", "subjectivity"]).values
        elif selected_phase == "Pragmatic":
            X_features = pd.DataFrame(X_raw.apply(pragmatic_features).tolist(), columns=pragmatic_words).values
        else:
            st.error(f"Unknown feature phase: {selected_phase}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Feature extraction failed for Google claims: {e}")
        return pd.DataFrame()
    results_list = []
    for model_name, model in trained_models.items():
        try:
            if model is None:
                st.warning(f"Model {model_name} is None, skipping prediction")
                results_list.append({
                    'Model': model_name,
                    'Accuracy': 0,
                    'F1-Score': 0,
                    'Precision': 0,
                    'Recall': 0,
                    'Inference Latency (ms)': 9999
                })
                continue
                
            if model_name == "Naive Bayes":
                X_features_model = np.abs(X_features).astype(float)
            else:
                X_features_model = X_features
            start_inference = time.time()
            y_pred = model.predict(X_features_model)
            inference_time = (time.time() - start_inference) * 1000
            accuracy = accuracy_score(y_true, y_pred) * 100
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            results_list.append({
                'Model': model_name,
                'Accuracy': accuracy,
                'F1-Score': f1,
                'Precision': precision,
                'Recall': recall,
                'Inference Latency (ms)': round(inference_time, 2)
            })
        except Exception as e:
            st.error(f"Prediction failed for {model_name}: {e}")
            results_list.append({
                'Model': model_name,
                'Accuracy': 0,
                'F1-Score': 0,
                'Precision': 0,
                'Recall': 0,
                'Inference Latency (ms)': 9999
            })
    return pd.DataFrame(results_list)

# --------------------------
# CSV PROCESSING FUNCTIONS
# --------------------------
def create_binary_label_column(df, label_column):
    """
    Automatically create binary labels (0/1) from various label formats
    """
    if df.empty or label_column not in df.columns:
        return df
    
    df = df.copy()
    
    # Define label mappings
    TRUE_LABELS = ["true", "mostly true", "accurate", "correct", "real", "fact", "1", "yes", "y"]
    FALSE_LABELS = ["false", "mostly false", "pants on fire", "fake", "incorrect", "baseless", "misleading", "0", "no", "n"]
    
    # Initialize binary label column
    df['binary_label'] = -1  # Default to unknown
    
    for idx, row in df.iterrows():
        label = str(row[label_column]).lower().strip()
        
        # Check for True labels
        if any(true_label in label for true_label in TRUE_LABELS):
            df.at[idx, 'binary_label'] = 1
        # Check for False labels
        elif any(false_label in label for false_label in FALSE_LABELS):
            df.at[idx, 'binary_label'] = 0
        # Check for numeric labels
        elif label.isdigit():
            num_label = int(label)
            if num_label == 1:
                df.at[idx, 'binary_label'] = 1
            elif num_label == 0:
                df.at[idx, 'binary_label'] = 0
    
    # Count statistics
    true_count = len(df[df['binary_label'] == 1])
    false_count = len(df[df['binary_label'] == 0])
    unknown_count = len(df[df['binary_label'] == -1])
    
    if true_count + false_count > 0:
        st.success(f"Created binary labels: {true_count} True (1), {false_count} False (0), {unknown_count} Unknown")
        
        # Create readable label column
        df['label_readable'] = df['binary_label'].map({1: 'True', 0: 'False', -1: 'Unknown'})
    else:
        st.warning("Could not automatically create binary labels from the selected column.")
    
    return df

def filter_by_date_range(df, date_column, start_date, end_date):
    """
    Filter DataFrame by date range
    """
    if df.empty or date_column not in df.columns:
        return df
    
    df = df.copy()
    
    # Convert date column to datetime
    try:
        df['date_parsed'] = pd.to_datetime(df[date_column], errors='coerce')
        
        # Filter by date range
        mask = (df['date_parsed'] >= pd.Timestamp(start_date)) & (df['date_parsed'] <= pd.Timestamp(end_date))
        filtered_df = df[mask].copy()
        
        st.info(f"Filtered to {len(filtered_df)} claims from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        return filtered_df
    except Exception as e:
        st.error(f"Error parsing dates: {e}")
        return df

def apply_date_filter_to_verification(verification_df, original_df, date_column, start_date, end_date):
    """
    Apply date filter to verification results by matching with original data
    """
    if verification_df.empty or original_df.empty:
        return pd.DataFrame()
    
    try:
        # Create a copy of original data with date parsing
        original_with_dates = original_df.copy()
        original_with_dates['date_parsed'] = pd.to_datetime(original_with_dates[date_column], errors='coerce')
        
        # Filter original data by date
        mask = (original_with_dates['date_parsed'] >= pd.Timestamp(start_date)) & \
               (original_with_dates['date_parsed'] <= pd.Timestamp(end_date))
        filtered_original = original_with_dates[mask]
        
        if filtered_original.empty:
            return pd.DataFrame()
        
        # Match verification results with filtered data using statement text
        filtered_statements = set(filtered_original['statement'].astype(str).str.lower().str.strip())
        
        # Filter verification results
        filtered_verification = verification_df.copy()
        filtered_verification['statement_lower'] = filtered_verification['original_statement'].astype(str).str.lower().str.strip()
        filtered_verification = filtered_verification[filtered_verification['statement_lower'].isin(filtered_statements)]
        
        # Drop temporary column
        filtered_verification = filtered_verification.drop('statement_lower', axis=1)
        
        st.info(f"Filtered verification results: {len(filtered_verification)} claims match the date range")
        
        return filtered_verification
    except Exception as e:
        st.error(f"Error filtering verification results: {e}")
        return verification_df

# --------------------------
# CSV TEMPLATE FUNCTION
# --------------------------
def get_politifact_csv_template():
    """Generate a template CSV structure for Politifact data"""
    template = {
        'statement': 'The claim text goes here',
        'rating': 'False',  # Can be True/False/Mostly True/etc
        'binary_rating': 0,  # 0 for False, 1 for True
        'date': '2023-01-15',
        'source': 'Politifact',
        'author': 'Author Name',
        'url': 'https://politifact.com/factchecks/...'
    }
    return pd.DataFrame([template])

# --------------------------
# WEB SCRAPING (Politifact) - KEEPING FOR DATE-WISE EXTRACTION
# --------------------------
def scrape_data_by_date_range(start_date: pd.Timestamp, end_date: pd.Timestamp):
    base_url = "https://www.politifact.com/factchecks/list/"
    current_url = base_url
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["author", "statement", "source", "date", "label"])
    scraped_rows_count = 0
    page_count = 0
    st.caption(f"Starting scrape from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    placeholder = st.empty()

    while current_url and page_count < 100:
        page_count += 1
        placeholder.text(f"Fetching page {page_count}... Scraped {scraped_rows_count} claims so far.")
        try:
            response = requests.get(current_url, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
        except requests.exceptions.RequestException as e:
            placeholder.error(f"Network Error during request: {e}. Stopping scrape.")
            break
        rows_to_add = []
        for card in soup.find_all("li", class_="o-listicle__item"):
            date_div = card.find("div", class_="m-statement__desc")
            date_text = date_div.get_text(strip=True) if date_div else None
            claim_date = None
            if date_text:
                match = re.search(r"stated on ([A-Za-z]+\s+\d{1,2},\s+\d{4})", date_text)
                if match:
                    try:
                        claim_date = pd.to_datetime(match.group(1), format='%B %d, %Y')
                    except ValueError:
                        continue
            if claim_date:
                if start_date <= claim_date <= end_date:
                    statement_block = card.find("div", class_="m-statement__quote")
                    statement = statement_block.find("a", href=True).get_text(strip=True) if statement_block and statement_block.find("a", href=True) else None
                    source_a = card.find("a", class_="m-statement__name")
                    source = source_a.get_text(strip=True) if source_a else None
                    footer = card.find("footer", class_="m-statement__footer")
                    author = None
                    if footer:
                        author_match = re.search(r"By\s+([^•]+)", footer.get_text(strip=True))
                        if author_match:
                            author = author_match.group(1).strip()
                    label_img = card.find("img", alt=True)
                    label = label_img['alt'].replace('-', ' ').title() if label_img and 'alt' in label_img.attrs else None
                    rows_to_add.append([author, statement, source, claim_date.strftime('%Y-%m-%d'), label])
                elif claim_date < start_date:
                    placeholder.warning(f"Encountered claim older than start date ({start_date.strftime('%Y-%m-%d')}). Stopping scrape.")
                    current_url = None
                    break
        if current_url is None:
            break
        writer.writerows(rows_to_add)
        scraped_rows_count += len(rows_to_add)
        next_link = soup.find("a", class_="c-button c-button--hollow", string=re.compile(r"Next", re.I))
        if next_link and 'href' in next_link.attrs:
            next_href = next_link['href'].rstrip('&').rstrip('?')
            current_url = urljoin(base_url, next_href)
        else:
            placeholder.success("No more pages found or last page reached.")
            current_url = None
    placeholder.success(f"Scraping finished! Total claims processed: {scraped_rows_count}")
    output.seek(0)
    df = pd.read_csv(output, header=0, keep_default_na=False)
    df = df.dropna(subset=['statement', 'label'])
    df.to_csv(SCRAPED_DATA_PATH, index=False)
    return df

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
# CLASSIFIERS & FEATURE APPLICATION
# --------------------------
def get_classifier(name):
    if name == "Naive Bayes":
        return MultinomialNB()
    elif name == "Decision Tree":
        return DecisionTreeClassifier(random_state=42, class_weight='balanced')
    elif name == "Logistic Regression":
        return LogisticRegression(max_iter=1000, solver='liblinear', random_state=42, class_weight='balanced')
    elif name == "SVM":
        return SVC(kernel='linear', C=0.5, random_state=42, class_weight='balanced')
    return None

def apply_feature_extraction(X, phase, vectorizer=None):
    if phase == "Lexical & Morphological":
        X_processed = X.apply(lexical_features)
        vectorizer = vectorizer if vectorizer else CountVectorizer(binary=True, ngram_range=(1,2), max_features=5000)
        X_features = vectorizer.fit_transform(X_processed)
        return X_features, vectorizer
    elif phase == "Syntactic":
        X_processed = X.apply(syntactic_features)
        vectorizer = vectorizer if vectorizer else TfidfVectorizer(max_features=5000)
        X_features = vectorizer.fit_transform(X_processed)
        return X_features, vectorizer
    elif phase == "Semantic":
        try:
            X_features = pd.DataFrame(X.apply(semantic_features).tolist(), columns=["polarity", "subjectivity"])
            # Handle NaN values
            X_features = X_features.fillna(0)
            return X_features, None
        except Exception as e:
            st.error(f"Semantic feature extraction failed: {e}")
            return None, None
    elif phase == "Discourse":
        X_processed = X.apply(discourse_features)
        vectorizer = vectorizer if vectorizer else CountVectorizer(ngram_range=(1,2), max_features=5000)
        X_features = vectorizer.fit_transform(X_processed)
        return X_features, vectorizer
    elif phase == "Pragmatic":
        try:
            X_features = pd.DataFrame(X.apply(pragmatic_features).tolist(), columns=pragmatic_words)
            # Handle NaN values
            X_features = X_features.fillna(0)
            return X_features, None
        except Exception as e:
            st.error(f"Pragmatic feature extraction failed: {e}")
            return None, None
    return None, None

# --------------------------
# Save & Load trained models (joblib)
# --------------------------
def save_trained_models(trained_models: dict, vectorizer, selected_phase):
    try:
        MODELS_DIR.mkdir(exist_ok=True)
        # save each model separately using joblib
        for name, model in trained_models.items():
            if model is not None:  # Only save if model is not None
                path = MODELS_DIR / f"{name.replace(' ', '_')}.joblib"
                try:
                    dump(model, path)
                except Exception as e:
                    st.warning(f"Failed to save {name}: {e}")
        # save vectorizer if present
        if vectorizer is not None:
            try:
                dump(vectorizer, MODELS_DIR / "vectorizer.joblib")
            except Exception as e:
                st.warning(f"Failed to save vectorizer: {e}")
        # save metadata (selected phase)
        with open(MODELS_DIR / "metadata.json", "w") as f:
            json.dump({"selected_phase": selected_phase}, f)
        st.success("Trained models and vectorizer saved to disk.")
    except Exception as e:
        st.error(f"Saving models failed: {e}")

def load_trained_models(expected_model_names=None):
    """
    Attempt to load saved models/vectorizer/metadata from models/ directory.
    Returns (trained_models_dict, vectorizer_or_none, selected_phase_or_none)
    """
    trained_models = {}
    vectorizer = None
    selected_phase = None
    try:
        meta_path = MODELS_DIR / "metadata.json"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                meta = json.load(f)
                selected_phase = meta.get("selected_phase")
        # load vectorizer
        vec_path = MODELS_DIR / "vectorizer.joblib"
        if vec_path.exists():
            try:
                vectorizer = load(vec_path)
            except Exception:
                vectorizer = None
        # load model files for expected names or discover joblib files
        if expected_model_names:
            names = expected_model_names
        else:
            # discover all joblib files except vectorizer
            names = []
            for p in MODELS_DIR.glob("*.joblib"):
                if p.name != "vectorizer.joblib":
                    names.append(p.stem)
        # load each model if available
        for name in names:
            filename = (MODELS_DIR / f"{name.replace(' ', '_')}.joblib")
            # also try alternative stem names
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
# Single-text prediction helper
# --------------------------
def predict_single_text(text, trained_models, vectorizer, selected_phase):
    if not text or not trained_models:
        return {"error": "No trained models available"}
    
    # Check if any models are actually trained
    if not any(model is not None for model in trained_models.values()):
        return {"error": "All models are None. Please train models first."}
    
    try:
        X_raw_series = pd.Series([str(text)])
        
        if selected_phase == "Lexical & Morphological":
            X_proc = X_raw_series.apply(lexical_features)
            if vectorizer is None:
                raise ValueError("Vectorizer required for Lexical phase is missing.")
            X_features = vectorizer.transform(X_proc)
        elif selected_phase == "Syntactic":
            X_proc = X_raw_series.apply(syntactic_features)
            if vectorizer is None:
                raise ValueError("Vectorizer required for Syntactic phase is missing.")
            X_features = vectorizer.transform(X_proc)
        elif selected_phase == "Discourse":
            X_proc = X_raw_series.apply(discourse_features)
            if vectorizer is None:
                raise ValueError("Vectorizer required for Discourse phase is missing.")
            X_features = vectorizer.transform(X_proc)
        elif selected_phase == "Semantic":
            X_features = pd.DataFrame(X_raw_series.apply(semantic_features).tolist(), 
                                     columns=["polarity", "subjectivity"]).fillna(0).values
        elif selected_phase == "Pragmatic":
            X_features = pd.DataFrame(X_raw_series.apply(pragmatic_features).tolist(), 
                                     columns=pragmatic_words).fillna(0).values
        else:
            raise ValueError(f"Unknown selected_phase: {selected_phase}")
    except Exception as e:
        return {"error": f"Feature extraction failed: {e}"}
    
    results = {}
    for model_name, model in trained_models.items():
        if model is None:
            results[model_name] = {"error": "Model not trained"}
            continue
        try:
            if model_name == "Naive Bayes":
                # Convert to dense array for Naive Bayes if sparse
                if hasattr(X_features, "toarray"):
                    X_for_model = np.abs(X_features.toarray()).astype(float)
                else:
                    X_for_model = np.abs(X_features).astype(float)
            elif model_name == "Decision Tree" and hasattr(X_features, "toarray"):
                # Convert sparse to dense for Decision Tree
                X_for_model = X_features.toarray()
            else:
                X_for_model = X_features
            
            # Ensure X_for_model has correct shape
            if X_for_model.ndim == 1:
                X_for_model = X_for_model.reshape(1, -1)
            elif X_for_model.ndim > 2:
                X_for_model = X_for_model.reshape(1, -1)
            
            pred = model.predict(X_for_model)
            label = int(pred[0]) if hasattr(pred, "__len__") else int(pred)
            results[model_name] = {"prediction": label}
        except Exception as e:
            # Try alternative approach
            try:
                if hasattr(X_features, "toarray"):
                    X_alt = X_features.toarray()
                else:
                    X_alt = X_features
                
                if X_alt.ndim == 1:
                    X_alt = X_alt.reshape(1, -1)
                elif X_alt.ndim > 2:
                    X_alt = X_alt.reshape(1, -1)
                
                pred = model.predict(X_alt)
                label = int(pred[0]) if hasattr(pred, "__len__") else int(pred)
                results[model_name] = {"prediction": label}
            except Exception as e2:
                results[model_name] = {"error": f"Prediction failed: {str(e2)[:100]}"}
    return results

# --------------------------
# IMPROVED Model training & evaluation with feature extraction optimization
# --------------------------
def evaluate_models(df: pd.DataFrame, selected_phase: str, text_column: str = 'statement', label_column: str = 'label', use_smote: bool = True):
    # Use binary_label column if it exists, otherwise use label column
    if 'binary_label' in df.columns:
        label_col_to_use = 'binary_label'
        # Filter out unknown labels (-1)
        df = df[df['binary_label'] != -1].copy()
    else:
        label_col_to_use = label_column
    
    # Map labels to binary if needed
    REAL_LABELS = ["true", "mostly true", "accurate", "correct", "real", "fact", "1"]
    FAKE_LABELS = ["false", "mostly false", "pants on fire", "fake", "incorrect", "baseless", "misleading", "0"]

    def create_binary_target(label):
        if pd.isna(label):
            return np.nan
        label_str = str(label).lower().strip()
        if any(true_label in label_str for true_label in REAL_LABELS):
            return 1
        elif any(false_label in label_str for false_label in FAKE_LABELS):
            return 0
        elif label_str.isdigit():
            num_label = int(label_str)
            if num_label == 1:
                return 1
            elif num_label == 0:
                return 0
        return np.nan

    if label_col_to_use != 'binary_label':
        df['target_label'] = df[label_col_to_use].apply(create_binary_target)
    else:
        df['target_label'] = df[label_col_to_use]
    
    # Drop NaN values
    df = df.dropna(subset=['target_label'])
    df = df[df[text_column].astype(str).str.len() > 10]
    
    if df.empty:
        st.error("No valid data after processing. Check your labels and data.")
        return pd.DataFrame(), {}, None
    
    X_raw = df[text_column].astype(str)
    y_raw = df['target_label'].astype(int)
    
    if len(np.unique(y_raw)) < 2:
        st.error(f"After binary mapping, only one class remains (class {y_raw.iloc[0]}). Cannot train classifier.")
        return pd.DataFrame(), {}, None

    # Extract features ONCE for the entire dataset
    X_features_full, vectorizer = apply_feature_extraction(X_raw, selected_phase)
    if X_features_full is None:
        st.error("Feature extraction failed.")
        return pd.DataFrame(), {}, None
    
    # Convert to appropriate format
    if isinstance(X_features_full, pd.DataFrame):
        X_features_full = X_features_full.values
    elif sparse.issparse(X_features_full):
        # Keep sparse for efficiency
        pass
    elif hasattr(X_features_full, "toarray"):
        X_features_full = X_features_full.toarray()
    
    y = y_raw.values
    
    # Adjust hyperparameters based on feature phase for better accuracy
    model_params = {
        "Naive Bayes": {
            "alpha": 0.1 if selected_phase in ["Lexical & Morphological", "Syntactic"] else 0.5,
            "fit_prior": True
        },
        "Decision Tree": {
            "max_depth": 20 if selected_phase in ["Semantic", "Pragmatic"] else 15,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": 'sqrt' if selected_phase in ["Lexical & Morphological", "Discourse"] else None,
            "random_state": 42
        },
        "Logistic Regression": {
            "C": 1.0 if selected_phase in ["Semantic", "Pragmatic"] else 0.5,
            "penalty": 'l2',
            "solver": 'liblinear',
            "max_iter": 2000,
            "random_state": 42
        },
        "SVM": {
            "C": 0.5 if selected_phase in ["Lexical & Morphological", "Syntactic"] else 1.0,
            "kernel": 'linear',
            "gamma": 'scale',
            "random_state": 42
        }
    }
    
    models_to_run = {
        "Naive Bayes": MultinomialNB(**model_params["Naive Bayes"]),
        "Decision Tree": DecisionTreeClassifier(
            **model_params["Decision Tree"]
        ),
        "Logistic Regression": LogisticRegression(
            **model_params["Logistic Regression"],
            class_weight='balanced'
        ),
        "SVM": SVC(
            **model_params["SVM"],
            class_weight='balanced'
        )
    }
    
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    model_metrics = {name: [] for name in models_to_run.keys()}
    
    for name, model in models_to_run.items():
        st.caption(f"Training {name} with {N_SPLITS}-Fold CV...")
        fold_metrics = {'accuracy': [], 'f1': [], 'precision': [], 'recall': [], 'train_time': [], 'inference_time': []}
        
        for fold, (train_index, test_index) in enumerate(skf.split(X_features_full, y)):
            try:
                # Split the ALREADY PROCESSED features
                X_train = X_features_full[train_index]
                X_test = X_features_full[test_index]
                y_train = y[train_index]
                y_test = y[test_index]
                
                # Prepare data for specific models
                if name == "Naive Bayes":
                    # Naive Bayes needs non-negative values
                    if sparse.issparse(X_train):
                        X_train_nb = X_train.toarray()
                        X_test_nb = X_test.toarray()
                    else:
                        X_train_nb = X_train
                        X_test_nb = X_test
                    X_train_nb = np.abs(X_train_nb).astype(float)
                    X_test_nb = np.abs(X_test_nb).astype(float)
                    
                    start_time = time.time()
                    
                    if use_smote and len(np.unique(y_train)) > 1:
                        min_class_count = np.min(np.bincount(y_train))
                        k_neighbors = min(3, min_class_count - 1) if min_class_count > 1 else 1
                        
                        smote_pipeline = ImbPipeline([
                            ('sampler', SMOTE(random_state=42, k_neighbors=k_neighbors)), 
                            ('classifier', model)
                        ])
                        smote_pipeline.fit(X_train_nb, y_train)
                        clf = smote_pipeline
                    else:
                        clf = model
                        clf.fit(X_train_nb, y_train)
                    
                    train_time = time.time() - start_time
                    start_inference = time.time()
                    y_pred = clf.predict(X_test_nb)
                    inference_time = (time.time() - start_inference) * 1000
                    
                else:
                    # For other models, convert sparse to dense if needed
                    if name == "Decision Tree" and sparse.issparse(X_train):
                        X_train_dt = X_train.toarray()
                        X_test_dt = X_test.toarray()
                    else:
                        X_train_dt = X_train
                        X_test_dt = X_test
                    
                    start_time = time.time()
                    
                    if use_smote and len(np.unique(y_train)) > 1:
                        min_class_count = np.min(np.bincount(y_train))
                        k_neighbors = min(3, min_class_count - 1) if min_class_count > 1 else 1
                        
                        smote_pipeline = ImbPipeline([
                            ('sampler', SMOTE(random_state=42, k_neighbors=k_neighbors)), 
                            ('classifier', model)
                        ])
                        smote_pipeline.fit(X_train_dt, y_train)
                        clf = smote_pipeline
                    else:
                        clf = model
                        clf.fit(X_train_dt, y_train)
                    
                    train_time = time.time() - start_time
                    start_inference = time.time()
                    y_pred = clf.predict(X_test_dt)
                    inference_time = (time.time() - start_inference) * 1000
                
                # Calculate metrics
                fold_metrics['accuracy'].append(accuracy_score(y_test, y_pred))
                fold_metrics['f1'].append(f1_score(y_test, y_pred, average='weighted', zero_division=0))
                fold_metrics['precision'].append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
                fold_metrics['recall'].append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
                fold_metrics['train_time'].append(train_time)
                fold_metrics['inference_time'].append(inference_time)
                
            except Exception as e:
                st.warning(f"Fold {fold+1} failed for {name}: {str(e)[:100]}")
                # Append zeros for this fold
                for key in ['accuracy', 'f1', 'precision', 'recall']:
                    fold_metrics[key].append(0)
                fold_metrics['train_time'].append(0)
                fold_metrics['inference_time'].append(0)
        
        # Calculate final metrics for this model
        if fold_metrics['accuracy'] and any(acc > 0 for acc in fold_metrics['accuracy']):
            model_metrics[name] = {
                "Model": name,
                "Accuracy": np.mean(fold_metrics['accuracy']) * 100,
                "F1-Score": np.mean(fold_metrics['f1']),
                "Precision": np.mean(fold_metrics['precision']),
                "Recall": np.mean(fold_metrics['recall']),
                "Training Time (s)": round(np.mean(fold_metrics['train_time']), 2),
                "Inference Latency (ms)": round(np.mean(fold_metrics['inference_time']), 2),
            }
        else:
            model_metrics[name] = {
                "Model": name, "Accuracy": 0, "F1-Score": 0, "Precision": 0, "Recall": 0,
                "Training Time (s)": 0, "Inference Latency (ms)": 9999,
            }
    
    # Train final models on full dataset
    st.caption("Training final models on complete dataset...")
    trained_models_final = {}
    
    for name in models_to_run.keys():
        try:
            # Get fresh model with optimized parameters
            if name == "Naive Bayes":
                final_model = MultinomialNB(**model_params["Naive Bayes"])
                # Prepare final training data for Naive Bayes
                if sparse.issparse(X_features_full):
                    X_final_nb = X_features_full.toarray()
                else:
                    X_final_nb = X_features_full
                X_final_nb = np.abs(X_final_nb).astype(float)
                
                if use_smote and len(np.unique(y)) > 1:
                    min_class_count = np.min(np.bincount(y))
                    k_neighbors = min(3, min_class_count - 1) if min_class_count > 1 else 1
                    
                    smote_pipeline_final = ImbPipeline([
                        ('sampler', SMOTE(random_state=42, k_neighbors=k_neighbors)), 
                        ('classifier', final_model)
                    ])
                    smote_pipeline_final.fit(X_final_nb, y)
                    trained_models_final[name] = smote_pipeline_final
                else:
                    final_model.fit(X_final_nb, y)
                    trained_models_final[name] = final_model
                    
            elif name == "Decision Tree":
                final_model = DecisionTreeClassifier(**model_params["Decision Tree"])
                # Prepare final training data for Decision Tree
                if sparse.issparse(X_features_full):
                    X_final_dt = X_features_full.toarray()
                else:
                    X_final_dt = X_features_full
                
                if use_smote and len(np.unique(y)) > 1:
                    min_class_count = np.min(np.bincount(y))
                    k_neighbors = min(3, min_class_count - 1) if min_class_count > 1 else 1
                    
                    smote_pipeline_final = ImbPipeline([
                        ('sampler', SMOTE(random_state=42, k_neighbors=k_neighbors)), 
                        ('classifier', final_model)
                    ])
                    smote_pipeline_final.fit(X_final_dt, y)
                    trained_models_final[name] = smote_pipeline_final
                else:
                    final_model.fit(X_final_dt, y)
                    trained_models_final[name] = final_model
                    
            else:  # Logistic Regression and SVM
                if name == "Logistic Regression":
                    final_model = LogisticRegression(**model_params["Logistic Regression"], class_weight='balanced')
                else:  # SVM
                    final_model = SVC(**model_params["SVM"], class_weight='balanced')
                
                # Prepare final training data
                if sparse.issparse(X_features_full):
                    X_final = X_features_full
                else:
                    X_final = X_features_full
                
                if use_smote and len(np.unique(y)) > 1:
                    min_class_count = np.min(np.bincount(y))
                    k_neighbors = min(3, min_class_count - 1) if min_class_count > 1 else 1
                    
                    smote_pipeline_final = ImbPipeline([
                        ('sampler', SMOTE(random_state=42, k_neighbors=k_neighbors)), 
                        ('classifier', final_model)
                    ])
                    smote_pipeline_final.fit(X_final, y)
                    trained_models_final[name] = smote_pipeline_final
                else:
                    final_model.fit(X_final, y)
                    trained_models_final[name] = final_model
            
        except Exception as e:
            st.error(f"Failed to train final {name} model: {str(e)[:200]}")
            trained_models_final[name] = None
    
    # Save models
    try:
        save_trained_models(trained_models_final, vectorizer, selected_phase)
    except Exception as e:
        st.warning(f"Saving after training failed: {e}")
    
    results_list = list(model_metrics.values())
    return pd.DataFrame(results_list), trained_models_final, vectorizer

# --------------------------
# Humor & critique functions
# --------------------------
def get_phase_critique(best_phase: str) -> str:
    critiques = {
        "Lexical & Morphological": ["Ah, the Lexical phase. Proving that sometimes, all you need is raw vocabulary and minimal effort. It's the high-school dropout that won the Nobel Prize.", "Just words, nothing fancy. This phase decided to ditch the deep thought and focus on counting. Turns out, quantity has a quality all its own.", "The Lexical approach: when in doubt, just scream the words louder. It lacks elegance but gets the job done."],
        "Syntactic": ["Syntactic features won? So grammar actually matters! We must immediately inform Congress. This phase is the meticulous editor who corrects everyone's texts.", "The grammar police have prevailed. This model focused purely on structure, proving that sentence construction is more important than meaning... wait, is that how politics works?", "It passed the grammar check! This phase is the sensible adult in the room, refusing to process any nonsense until the parts of speech align."],
        "Semantic": ["The Semantic phase won by feeling its feelings. It's highly emotional, heavily relying on vibes and tone. Surprisingly effective, just like a good political ad.", "It turns out sentiment polarity is the secret sauce! This model just needed to know if the statement felt 'good' or 'bad.' Zero complex reasoning required.", "Semantic victory! The model simply asked, 'Are they being optimistic or negative?' and apparently that was enough to crush the competition."],
        "Discourse": ["Discourse features won! This phase is the over-analyzer, counting sentences and focusing on the rhythm of the argument. It knows the debate structure better than the content.", "The long-winded champion! This model cared about how the argument was *structured*—the thesis, the body, the conclusion. It's basically the high school debate team captain.", "Discourse is the winner! It successfully mapped the argument's flow, proving that presentation beats facts."],
        "Pragmatic": ["The Pragmatic phase won by focusing on keywords like 'must' and '?'. It just needed to know the speaker's intent. It's the Sherlock Holmes of NLP.", "It's all about intent! This model ignored the noise and hunted for specific linguistic tells. It's concise, ruthless, and apparently correct.", "Pragmatic features for the win! The model knows that if someone uses three exclamation marks, they're either lying or selling crypto. Either way, it's a clue."],
    }
    return random.choice(critiques.get(best_phase, ["The results are in, and the system is speechless. It seems we need to hire a better comedian."]))

def get_model_critique(best_model: str) -> str:
    critiques = {
        "Naive Bayes": ["Naive Bayes: It's fast, it's simple, and it assumes every feature is independent. The model is either brilliant or blissfully unaware, but hey, it works!", "The Simpleton Savant has won! Naive Bayes brings zero drama and just counts things. It's the least complicated tool in the box, which is often the best.", "NB pulled off a victory. It's the 'less-is-more' philosopher who manages to outperform all the complex math majors."],
        "Decision Tree": ["The Decision Tree won by asking a series of simple yes/no questions until it got tired. It's transparent, slightly judgmental, and surprisingly effective.", "The Hierarchical Champion! It built a beautiful, intricate set of if/then statements. It's the most organized person in the office, and the accuracy shows it.", "Decision Tree victory! It achieved success by splitting the data until it couldn't be split anymore. A classic strategy in science and divorce."],
        "Logistic Regression": ["Logistic Regression: The veteran politician of ML. It draws a clean, straight line to victory. Boring, reliable, and hard to beat.", "The Straight-Line Stunner. It uses simple math to predict complex reality. It's predictable, efficient, and definitely got tenure.", "LogReg prevails! The model's philosophy is: 'Probability is all you need.' It's the safest bet, and the accuracy score agrees."],
        "SVM": ["SVM: It found the biggest, widest gap between the truth and the lies, and parked its hyperplane right there. Aggressive but effective boundary enforcement.", "The Maximizing Margin Master! SVM doesn't just separate classes; it builds a fortress between them. It's the most dramatic and highly paid algorithm here.", "SVM crushed it! It's the model that believes in extreme boundaries. No fuzzy logic, just a hard, clean, dividing line."],
    }
    return random.choice(critiques.get(best_model, ["This model broke the simulation, so we have nothing funny to say."]))

def generate_humorous_critique(df_results: pd.DataFrame, selected_phase: str) -> str:
    if df_results.empty:
        return "The system failed to train anything. We apologize; our ML models are currently on strike demanding better data and less existential dread."
    df_results['F1-Score'] = pd.to_numeric(df_results['F1-Score'], errors='coerce').fillna(0)
    best_model_row = df_results.loc[df_results['F1-Score'].idxmax()]
    best_model = best_model_row['Model']
    max_f1 = best_model_row['F1-Score']
    max_acc = best_model_row['Accuracy']
    phase_critique = get_phase_critique(selected_phase)
    model_critique = get_model_critique(best_model)
    headline = f"The Golden Snitch Award goes to the {best_model}!"
    summary = (
        f"**Accuracy Report Card:** {headline}\n\n"
        f"This absolute unit achieved a **{max_acc:.2f}% Accuracy** (and {max_f1:.2f} F1-Score) on the `{selected_phase}` feature set. "
        f"It beat its rivals, proving that when faced with political statements, the winning strategy was to rely on: **{selected_phase} features!**\n\n"
    )
    roast = (
        f"### The AI Roast (Certified by a Data Scientist):\n"
        f"**Phase Performance:** {phase_critique}\n\n"
        f"**Model Personality:** {model_critique}\n\n"
        f"*(Disclaimer: All models were equally confused by the 'Mostly True' label, which they collectively deemed an existential threat.)*"
    )
    return summary + roast

# --------------------------
# Helper function to show verification results
# --------------------------
def show_verification_results(verification_df):
    """Helper function to display verification results"""
    if verification_df.empty:
        st.warning("No verification results to display.")
        return
    
    with st.expander("Verification Results", expanded=True):
        # Show summary statistics
        found_df = verification_df[verification_df['google_verification'] == 'Found']
        if not found_df.empty:
            # Calculate accuracy if we have original binary labels
            if 'binary_label' in found_df.columns and 'google_binary_label' in found_df.columns:
                # Filter only where both labels exist
                valid_comparisons = found_df.dropna(subset=['binary_label', 'google_binary_label'])
                valid_comparisons = valid_comparisons[valid_comparisons['google_binary_label'] != -1]
                
                if len(valid_comparisons) > 0:
                    matches = (valid_comparisons['binary_label'] == valid_comparisons['google_binary_label']).sum()
                    accuracy = matches / len(valid_comparisons) * 100
                    st.info(f"Google API vs Original Labels Accuracy: {accuracy:.1f}% ({matches}/{len(valid_comparisons)} matches)")
        
        # Show detailed results
        display_cols = ['original_statement', 'original_label', 'google_verification', 
                       'google_rating', 'match_score', 'verification_confidence']
        if 'google_binary_label' in verification_df.columns:
            display_cols.append('google_binary_label')
        
        st.dataframe(verification_df[display_cols], use_container_width=True)
        
        # Visualization of verification results
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Pie chart: Found vs Not Found
        found_count = len(verification_df[verification_df['google_verification'] == 'Found'])
        not_found_count = len(verification_df) - found_count
        
        if found_count + not_found_count > 0:
            ax1.pie([found_count, not_found_count], 
                   labels=['Verified', 'Not Found'], 
                   autopct='%1.1f%%',
                   colors=['#4CAF50', '#F44336'])
            ax1.set_title('Google API Verification Results')
        else:
            ax1.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax1.set_title('No Verification Data')
        
        # 2. Bar chart: Confidence levels
        if found_count > 0:
            conf_counts = verification_df['verification_confidence'].value_counts()
            colors_conf = {'High': '#4CAF50', 'Medium': '#FF9800', 'Low': '#F44336'}
            conf_colors = [colors_conf.get(conf, '#9E9E9E') for conf in conf_counts.index]
            
            bars = ax2.bar(range(len(conf_counts)), conf_counts.values, 
                          color=conf_colors, edgecolor='black')
            ax2.set_xlabel('Confidence Level')
            ax2.set_ylabel('Count')
            ax2.set_title('Verification Confidence Levels')
            ax2.set_xticks(range(len(conf_counts)))
            ax2.set_xticklabels(conf_counts.index, fontsize=9)
            
            # Add count labels
            for bar, count in zip(bars, conf_counts.values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{count}', ha='center', va='bottom', fontsize=9)
        else:
            ax2.text(0.5, 0.5, 'No matches found', ha='center', va='center')
            ax2.set_title('No Verification Matches')
        
        # 3. Histogram: Match scores
        if found_count > 0:
            match_scores = verification_df['match_score'].dropna()
            ax3.hist(match_scores, bins=10, color='#2196F3', edgecolor='black', alpha=0.7)
            ax3.set_xlabel('Match Score')
            ax3.set_ylabel('Count')
            ax3.set_title('Claim Match Scores Distribution')
            ax3.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Threshold (0.5)')
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'No match scores', ha='center', va='center')
            ax3.set_title('No Match Scores')
        
        # 4. Google rating distribution
        if found_count > 0:
            rating_counts = verification_df['google_rating'].value_counts().head(10)
            if len(rating_counts) > 0:
                bars4 = ax4.bar(range(len(rating_counts)), rating_counts.values, 
                               color='#9C27B0', edgecolor='black', alpha=0.7)
                ax4.set_xlabel('Google Rating')
                ax4.set_ylabel('Count')
                ax4.set_title('Top 10 Google Ratings')
                ax4.set_xticks(range(len(rating_counts)))
                ax4.set_xticklabels(rating_counts.index, rotation=45, ha='right', fontsize=8)
                
                # Add count labels
                for bar, count in zip(bars4, rating_counts.values):
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                            f'{count}', ha='center', va='bottom', fontsize=8)
            else:
                ax4.text(0.5, 0.5, 'No ratings', ha='center', va='center')
                ax4.set_title('No Ratings Found')
        else:
            ax4.text(0.5, 0.5, 'No ratings', ha='center', va='center')
            ax4.set_title('No Ratings Found')
        
        plt.tight_layout()
        st.pyplot(fig)

# --------------------------
# GOOGLE API KEY MANAGEMENT
# --------------------------
def get_google_api_key():
    """
    Get Google API key from secrets.toml in root directory
    """
    try:
        # First try to read from root directory (where your secrets.toml is)
        secrets_path = pathlib.Path("secrets.toml")
        
        if secrets_path.exists():
            # Read the file directly
            with open(secrets_path, 'r') as f:
                content = f.read()
                # Simple parsing for GOOGLE_API_KEY
                import re
                match = re.search(r'GOOGLE_API_KEY\s*=\s*"([^"]+)"', content)
                if match:
                    api_key = match.group(1)
                    if api_key and api_key.strip() and api_key != "your_actual_api_key_here":
                        st.success("API Key loaded from secrets.toml")
                        return api_key.strip()
                    else:
                        st.error("Invalid API key in secrets.toml. It's still using the placeholder value.")
                else:
                    st.error("GOOGLE_API_KEY not found in secrets.toml")
        else:
            st.error(f"secrets.toml not found at: {secrets_path.absolute()}")
            
        # If we get here, provide instructions
        st.info("""
        To fix this:
        
        1. **Check your secrets.toml file** - Make sure it contains:
        ```toml
        GOOGLE_API_KEY = "your_actual_api_key_here"
        ```
        
        2. **Move it to .streamlit/ folder** (Optional but recommended):
        ```bash
        mkdir -p .streamlit
        mv secrets.toml .streamlit/
        ```
        
        3. **Or keep it in root** - This modified code should work with it in root directory
        
        4. **Restart the app**
        """)
        return None
        
    except Exception as e:
        st.error(f"Error reading API key: {e}")
        return None

# --------------------------
# STREAMLIT APP
# --------------------------
def app():
    st.set_page_config(page_title='FactChecker: AI Fact-Checking Platform', layout='wide', initial_sidebar_state='expanded')

    # Clean Blue and White Theme CSS with improved sidebar
    st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --primary-blue: #1e88e5;
        --secondary-blue: #64b5f6;
        --light-blue: #bbdefb;
        --accent-blue: #0d47a1;
        --white: #ffffff;
        --light-gray: #f5f5f5;
        --dark-text: #212121;
        --sidebar-blue: #e3f2fd;
    }
    
    /* Overall page styling */
    .stApp {
        background-color: #ffffff;
        color: var(--dark-text);
    }
    
    /* Sidebar styling - Lighter Blue with improved text */
    section[data-testid="stSidebar"] {
        background-color: var(--sidebar-blue);
        border-right: 1px solid var(--light-blue);
    }
    
    /* Sidebar title - BOLDER and LARGER */
    section[data-testid="stSidebar"] h1 {
        color: var(--accent-blue) !important;
        font-size: 28px !important;
        font-weight: 800 !important;
        text-align: center !important;
        margin-bottom: 30px !important;
        padding-bottom: 15px !important;
        border-bottom: 3px solid var(--primary-blue) !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    /* Navigation menu - LARGER and CLEARER */
    section[data-testid="stSidebar"] .stRadio label {
        font-size: 18px !important;
        font-weight: 600 !important;
        color: var(--dark-text) !important;
        padding: 12px 18px !important;
        margin: 8px 0 !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
        background-color: transparent !important;
    }
    
    section[data-testid="stSidebar"] .stRadio label:hover {
        background-color: rgba(30, 136, 229, 0.15) !important;
        color: var(--accent-blue) !important;
        transform: translateX(5px);
    }
    
    section[data-testid="stSidebar"] .stRadio label div {
        font-size: 18px !important;
        font-weight: 600 !important;
    }
    
    /* Selected navigation item */
    section[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"]:has(input:checked) {
        background-color: rgba(30, 136, 229, 0.2) !important;
        color: var(--accent-blue) !important;
        border-left: 4px solid var(--primary-blue) !important;
    }
    
    /* Main headers - Clean without borders */
    h1, h2, h3 {
        color: var(--accent-blue) !important;
        padding-bottom: 0px !important;
        margin-bottom: 15px !important;
    }
    
    /* Buttons */
    .stButton button {
        background-color: var(--primary-blue) !important;
        color: white !important;
        font-weight: 500 !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 10px 20px !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton button:hover {
        background-color: var(--accent-blue) !important;
        box-shadow: 0 2px 6px rgba(30, 136, 229, 0.2) !important;
    }
    
    /* Dataframes and tables */
    .stDataFrame {
        border: 1px solid var(--light-blue) !important;
        border-radius: 6px !important;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: var(--accent-blue) !important;
        font-size: 22px !important;
        font-weight: bold !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--dark-text) !important;
        font-weight: 500 !important;
    }
    
    /* Input fields */
    .stTextInput input, .stSelectbox select, .stDateInput input {
        border: 1px solid var(--light-blue) !important;
        border-radius: 6px !important;
        padding: 8px 12px !important;
    }
    
    .stTextInput input:focus, .stSelectbox select:focus, .stDateInput input:focus {
        border-color: var(--primary-blue) !important;
        box-shadow: 0 0 0 2px rgba(30, 136, 229, 0.1) !important;
    }
    
    /* Alert messages */
    .stAlert {
        border-radius: 6px !important;
        border: 1px solid !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: var(--light-blue) !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        border: 1px solid var(--light-blue) !important;
    }
    
    /* Progress bars */
    .stProgress > div > div > div {
        background-color: var(--primary-blue) !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        border-bottom: 1px solid var(--light-blue);
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        font-weight: 500;
    }
    
    /* Remove borders from specific headers */
    .main-header {
        border-bottom: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'scraped_df' not in st.session_state:
        st.session_state['scraped_df'] = pd.DataFrame()
    if 'df_results' not in st.session_state:
        st.session_state['df_results'] = pd.DataFrame()
    if 'trained_models' not in st.session_state:
        st.session_state['trained_models'] = {}
    if 'trained_vectorizer' not in st.session_state:
        st.session_state['trained_vectorizer'] = None
    if 'google_benchmark_results' not in st.session_state:
        st.session_state['google_benchmark_results'] = pd.DataFrame()
    if 'google_df' not in st.session_state:
        st.session_state['google_df'] = pd.DataFrame()
    if 'selected_phase_run' not in st.session_state:
        st.session_state['selected_phase_run'] = None
    if 'use_smote' not in st.session_state:
        st.session_state['use_smote'] = True
    if 'verification_df' not in st.session_state:
        st.session_state['verification_df'] = pd.DataFrame()
    if 'filtered_verification_df' not in st.session_state:
        st.session_state['filtered_verification_df'] = pd.DataFrame()
    if 'selected_text_column' not in st.session_state:
        st.session_state['selected_text_column'] = 'statement'
    if 'selected_label_column' not in st.session_state:
        st.session_state['selected_label_column'] = 'label'
    if 'selected_date_column' not in st.session_state:
        st.session_state['selected_date_column'] = None
    if 'csv_columns_selected' not in st.session_state:
        st.session_state['csv_columns_selected'] = False
    if 'filtered_df' not in st.session_state:
        st.session_state['filtered_df'] = pd.DataFrame()
    if 'verification_performed' not in st.session_state:
        st.session_state['verification_performed'] = False

    # Attempt to load previously saved models on startup (only once)
    if 'models_loaded_attempted' not in st.session_state:
        expected_names = ["Naive_Bayes", "Decision_Tree", "Logistic_Regression", "SVM"]
        trained_models_loaded, vec_loaded, selected_phase_loaded = load_trained_models(expected_model_names=["Naive Bayes","Decision Tree","Logistic Regression","SVM"])
        any_model_loaded = any(v is not None for v in trained_models_loaded.values()) if trained_models_loaded else False
        if any_model_loaded:
            st.session_state['trained_models'] = trained_models_loaded
            st.session_state['trained_vectorizer'] = vec_loaded
            if selected_phase_loaded:
                st.session_state['selected_phase_run'] = selected_phase_loaded
        st.session_state['models_loaded_attempted'] = True

    # Sidebar and navigation with IMPROVED styling - REMOVED Benchmark Testing
    st.sidebar.markdown("<h1 class='main-header'>FactChecker</h1>", unsafe_allow_html=True)
    page = st.sidebar.radio("", ["Dashboard", "Data Collection", "Google API Verification", "Model Training", "Results & Analysis"], 
                          key='navigation', label_visibility="collapsed")

    # --- DASHBOARD ---
    if page == "Dashboard":
        st.markdown("<h1 class='main-header'>FactChecker Dashboard</h1>", unsafe_allow_html=True)
        
        # Status cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_claims = len(st.session_state.get('filtered_df', pd.DataFrame()))
            if total_claims == 0:
                total_claims = len(st.session_state['scraped_df']) if not st.session_state['scraped_df'].empty else 0
            st.markdown(f'''
            <div style="background: var(--white); padding: 20px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); border: 1px solid var(--light-blue); margin-bottom: 15px;">
                <h3 style="color: var(--accent-blue);">Data Overview</h3>
                <p>Collect and manage training data</p>
                <p style="font-size: 20px; font-weight: bold; color: var(--accent-blue);">
                    {total_claims} claims
                </p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            trained_model_count = sum(1 for m in st.session_state['trained_models'].values() if m is not None)
            st.markdown(f'''
            <div style="background: var(--white); padding: 20px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); border: 1px solid var(--light-blue); margin-bottom: 15px;">
                <h3 style="color: var(--accent-blue);">Model Training</h3>
                <p>Advanced NLP feature extraction and ML training</p>
                <p style="font-size: 20px; font-weight: bold; color: var(--accent-blue);">
                    {trained_model_count} models ready
                </p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col3:
            google_claims_count = len(st.session_state['google_df']) if not st.session_state['google_df'].empty else 0
            st.markdown(f'''
            <div style="background: var(--white); padding: 20px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); border: 1px solid var(--light-blue); margin-bottom: 15px;">
                <h3 style="color: var(--accent-blue);">External Data</h3>
                <p>Access to external fact-checking sources</p>
                <p style="font-size: 20px; font-weight: bold; color: var(--accent-blue);">
                    {google_claims_count} external claims
                </p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col4:
            verification_count = 0
            verification_rate = 0
            # Use filtered verification if available, otherwise use full verification
            verification_df = st.session_state.get('filtered_verification_df', st.session_state.get('verification_df', pd.DataFrame()))
            if not verification_df.empty:
                verification_count = len(verification_df)
                found_verified = verification_df[verification_df['google_verification'] == 'Found']
                found_count = len(found_verified)
                verification_rate = (found_count / verification_count * 100) if verification_count > 0 else 0
            
            st.markdown(f'''
            <div style="background: var(--white); padding: 20px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); border: 1px solid var(--light-blue); margin-bottom: 15px;">
                <h3 style="color: var(--accent-blue);">API Verification</h3>
                <p>Google API verification status</p>
                <p style="font-size: 20px; font-weight: bold; color: var(--accent-blue);">
                    {verification_rate:.1f}% verified
                </p>
                <p style="font-size: 12px; color: #666;">{verification_count} claims checked</p>
            </div>
            ''', unsafe_allow_html=True)

    # --- DATA COLLECTION ---
    elif page == "Data Collection":
        st.markdown("<h1 class='main-header'>Data Collection</h1>", unsafe_allow_html=True)
        
        # Choose data source
        data_source = st.radio("Choose data source:", ["Upload CSV File", "Web Scrape Politifact"], horizontal=True)
        
        if data_source == "Upload CSV File":
            st.write("Upload a CSV file containing Politifact fact-checked claims.")
            
            # CSV format help
            with st.expander("CSV Format Help", expanded=False):
                st.write("Your CSV should have at minimum these columns:")
                st.code("""
                [text_column], [label_column], [date_column (optional)]
                "The Earth is flat","False","2023-01-15"
                "Vaccines are safe","True","2023-02-20"
                "Climate change is real","Mostly True","2023-03-10"
                """, language='text')
                
                st.write("Label column can contain:")
                st.write("- Text labels: 'True', 'False', 'Mostly True', etc.")
                st.write("- Binary labels: '1' for True, '0' for False")
                st.write("- Numeric labels: 1 for True, 0 for False")
                
                if st.button("Download Template CSV", key="template_btn"):
                    template_df = get_politifact_csv_template()
                    csv = template_df.to_csv(index=False)
                    st.download_button(
                        label="Download Template",
                        data=csv,
                        file_name="politifact_template.csv",
                        mime="text/csv",
                        key="download_template"
                    )
            
            uploaded_file = st.file_uploader("Choose a Politifact CSV file", type="csv", key="csv_uploader")
            
            if uploaded_file is not None:
                try:
                    # Read the CSV file
                    scraped_df = pd.read_csv(uploaded_file)
                    
                    # Show column selection interface
                    st.subheader("Select Columns for Analysis")
                    
                    # Get all columns
                    all_columns = scraped_df.columns.tolist()
                    
                    if not all_columns:
                        st.error("CSV file has no columns!")
                    else:
                        # Let user select text, label, and date columns
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            text_column = st.selectbox(
                                "Select column containing claim text:",
                                all_columns,
                                index=0,
                                key="text_column_select"
                            )
                        
                        with col2:
                            label_column = st.selectbox(
                                "Select column containing labels/ratings:",
                                all_columns,
                                index=1 if len(all_columns) > 1 else 0,
                                key="label_column_select"
                            )
                        
                        with col3:
                            # Find date columns
                            date_candidates = [col for col in all_columns if any(date_word in col.lower() for date_word in ['date', 'time', 'year', 'month', 'day'])]
                            if date_candidates:
                                default_date_idx = all_columns.index(date_candidates[0])
                            else:
                                default_date_idx = 2 if len(all_columns) > 2 else (1 if len(all_columns) > 1 else 0)
                            
                            date_column = st.selectbox(
                                "Select date column (optional):",
                                ['None'] + all_columns,
                                index=0,
                                key="date_column_select"
                            )
                            if date_column == 'None':
                                date_column = None
                        
                        # Show preview of selected columns
                        st.write("Preview of selected columns:")
                        preview_cols = [text_column, label_column]
                        if date_column:
                            preview_cols.append(date_column)
                        preview_df = scraped_df[preview_cols].head(10)
                        st.dataframe(preview_df, use_container_width=True)
                        
                        # Load button
                        if st.button("Load Data with Selected Columns", key="load_csv_btn", use_container_width=True, type="primary"):
                            processed_df = scraped_df.copy()
                            
                            # Save selections to session state
                            st.session_state['selected_text_column'] = text_column
                            st.session_state['selected_label_column'] = label_column
                            st.session_state['selected_date_column'] = date_column
                            
                            # Rename columns for internal processing
                            processed_df = processed_df.rename(columns={
                                text_column: 'statement',
                                label_column: 'label'
                            })
                            
                            if date_column:
                                processed_df['date'] = scraped_df[date_column]
                            
                            # Automatically create binary labels (0/1)
                            st.info("Creating binary labels (0/1) from selected label column...")
                            processed_df = create_binary_label_column(processed_df, 'label')
                            
                            # Store in session state
                            st.session_state['scraped_df'] = processed_df
                            st.session_state['csv_columns_selected'] = True
                            st.session_state['filtered_df'] = processed_df.copy()  # Initially no filter
                            
                            # Clear previous verification if any
                            st.session_state['verification_df'] = pd.DataFrame()
                            st.session_state['filtered_verification_df'] = pd.DataFrame()
                            st.session_state['verification_performed'] = False
                            
                            st.success(f"Successfully loaded {len(processed_df)} claims from CSV!")
                            st.info(f"Using '{text_column}' as text column and '{label_column}' as label column.")
                            if date_column:
                                st.info(f"Using '{date_column}' as date column.")
                            
                            # Show data preview
                            with st.expander("Preview Loaded Data", expanded=True):
                                display_cols = ['statement', 'label', 'binary_label', 'label_readable']
                                if 'date' in processed_df.columns:
                                    display_cols.append('date')
                                st.dataframe(processed_df[display_cols].head(15), use_container_width=True)
                                
                                # Label Distribution visualization
                                st.subheader("Label Distribution")
                                
                                # Show binary label distribution
                                if 'binary_label' in processed_df.columns:
                                    binary_counts = processed_df['binary_label'].value_counts()
                                    readable_counts = processed_df['label_readable'].value_counts()
                                    
                                    # Create visualization
                                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                                    
                                    # Bar chart for binary labels
                                    colors_binary = ['#F44336', '#4CAF50', '#9E9E9E']  # Red, Green, Gray
                                    binary_labels = ['False (0)', 'True (1)', 'Unknown']
                                    binary_values = [
                                        binary_counts.get(0, 0),
                                        binary_counts.get(1, 0),
                                        binary_counts.get(-1, 0)
                                    ]
                                    
                                    bars1 = ax1.bar(range(len(binary_values)), binary_values, 
                                                   color=colors_binary[:len(binary_values)], 
                                                   edgecolor='black', linewidth=1)
                                    ax1.set_xlabel('Binary Labels')
                                    ax1.set_ylabel('Count')
                                    ax1.set_title('Binary Label Distribution (0/1)')
                                    ax1.set_xticks(range(len(binary_labels)))
                                    ax1.set_xticklabels(binary_labels, fontsize=9)
                                    
                                    # Add count labels on bars
                                    for bar, count in zip(bars1, binary_values):
                                        height = bar.get_height()
                                        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                                                f'{count}', ha='center', va='bottom', fontsize=9)
                                    
                                    # Pie chart for readable labels
                                    if len(readable_counts) > 0:
                                        colors_pie = plt.cm.Set3(np.linspace(0, 1, len(readable_counts)))
                                        wedges, texts, autotexts = ax2.pie(readable_counts.values, 
                                                                          labels=readable_counts.index, 
                                                                          autopct='%1.1f%%',
                                                                          startangle=90,
                                                                          colors=colors_pie,
                                                                          textprops={'fontsize': 9})
                                        ax2.set_title('Readable Label Distribution')
                                        
                                        # Improve pie chart text
                                        for autotext in autotexts:
                                            autotext.set_color('black')
                                            autotext.set_fontsize(9)
                                            autotext.set_fontweight('bold')
                                    else:
                                        ax2.text(0.5, 0.5, 'No labels found', 
                                                ha='center', va='center')
                                        ax2.set_title('No Label Data')
                                    
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    
                                    # Show statistics
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("True Claims (1)", binary_counts.get(1, 0))
                                    with col2:
                                        st.metric("False Claims (0)", binary_counts.get(0, 0))
                                    with col3:
                                        st.metric("Unknown Claims", binary_counts.get(-1, 0))
                            
                            # Date-wise filtering section
                            if date_column and 'date' in processed_df.columns:
                                st.markdown("---")
                                st.subheader("Date-wise Filtering")
                                
                                # Try to parse dates
                                try:
                                    processed_df['date_parsed'] = pd.to_datetime(processed_df['date'], errors='coerce')
                                    valid_dates = processed_df['date_parsed'].notna().sum()
                                    
                                    if valid_dates > 0:
                                        min_date = processed_df['date_parsed'].min().date()
                                        max_date = processed_df['date_parsed'].max().date()
                                        
                                        date_col1, date_col2 = st.columns(2)
                                        with date_col1:
                                            filter_start_date = st.date_input(
                                                "Filter Start Date",
                                                value=min_date,
                                                min_value=min_date,
                                                max_value=max_date,
                                                key="filter_start_date"
                                            )
                                        
                                        with date_col2:
                                            filter_end_date = st.date_input(
                                                "Filter End Date",
                                                value=max_date,
                                                min_value=min_date,
                                                max_value=max_date,
                                                key="filter_end_date"
                                            )
                                        
                                        if st.button("Apply Date Filter", key="apply_date_filter", use_container_width=True):
                                            if filter_start_date > filter_end_date:
                                                st.error("Start date must be before end date")
                                            else:
                                                filtered_df = filter_by_date_range(
                                                    processed_df,
                                                    'date',
                                                    filter_start_date,
                                                    filter_end_date
                                                )
                                                
                                                if not filtered_df.empty:
                                                    st.session_state['filtered_df'] = filtered_df
                                                    st.success(f"Filtered to {len(filtered_df)} claims from {filter_start_date} to {filter_end_date}")
                                                    
                                                    # Also filter verification results if they exist
                                                    if not st.session_state['verification_df'].empty:
                                                        filtered_verification = apply_date_filter_to_verification(
                                                            st.session_state['verification_df'],
                                                            processed_df,
                                                            'date',
                                                            filter_start_date,
                                                            filter_end_date
                                                        )
                                                        st.session_state['filtered_verification_df'] = filtered_verification
                                                        if not filtered_verification.empty:
                                                            st.info(f"Also filtered verification results: {len(filtered_verification)} claims")
                                                    
                                                    # Show filtered data preview
                                                    with st.expander("Preview Filtered Data", expanded=False):
                                                        display_cols = ['statement', 'label', 'binary_label', 'label_readable', 'date']
                                                        st.dataframe(filtered_df[display_cols].head(10), use_container_width=True)
                                                else:
                                                    st.warning("No claims found in the selected date range.")
                                    else:
                                        st.warning("No valid dates found in the date column.")
                                except Exception as e:
                                    st.warning(f"Could not parse dates: {e}")
                except Exception as e:
                    st.error(f"Error reading CSV file: {e}")
                    st.exception(e)
        
        elif data_source == "Web Scrape Politifact":
            # KEEPING THE ORIGINAL DATE-WISE EXTRACTION CODE
            st.write("Collect political fact-check data from Politifact.com within a specific date range.")
            
            min_date = pd.to_datetime('2007-01-01')
            max_date = pd.to_datetime('today').normalize()
            
            date_col1, date_col2 = st.columns(2)
            with date_col1:
                start_date = st.date_input("Start Date", min_value=min_date, max_value=max_date, 
                                          value=pd.to_datetime('2023-01-01'), key="scrape_start_date")
            
            with date_col2:
                end_date = st.date_input("End Date", min_value=min_date, max_value=max_date, 
                                        value=max_date, key="scrape_end_date")
            
            if st.button("Scrape Politifact Data", key="scrape_btn", use_container_width=True, type="primary"):
                if start_date > end_date:
                    st.error("Start date must be before end date")
                else:
                    with st.spinner("Scraping political claims from Politifact..."):
                        scraped_df = scrape_data_by_date_range(pd.to_datetime(start_date), pd.to_datetime(end_date))
                    
                    if not scraped_df.empty:
                        # Automatically create binary labels for scraped data too
                        scraped_df = create_binary_label_column(scraped_df, 'label')
                        
                        st.session_state['scraped_df'] = scraped_df
                        st.session_state['filtered_df'] = scraped_df.copy()
                        st.session_state['selected_text_column'] = 'statement'
                        st.session_state['selected_label_column'] = 'label'
                        st.session_state['selected_date_column'] = 'date'
                        st.session_state['csv_columns_selected'] = True
                        st.success(f"Successfully scraped {len(scraped_df)} claims!")
                        
                        # Show data preview
                        with st.expander("Preview Scraped Data", expanded=True):
                            display_cols = ['statement', 'label', 'binary_label', 'label_readable', 'date']
                            st.dataframe(scraped_df[display_cols].head(15), use_container_width=True)
                            
                            # Improved Label Distribution visualization
                            st.subheader("Label Distribution")
                            
                            # Show binary label distribution
                            if 'binary_label' in scraped_df.columns:
                                binary_counts = scraped_df['binary_label'].value_counts()
                                readable_counts = scraped_df['label_readable'].value_counts()
                                
                                # Create visualization
                                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                                
                                # Bar chart for binary labels
                                colors_binary = ['#F44336', '#4CAF50', '#9E9E9E']
                                binary_labels = ['False (0)', 'True (1)', 'Unknown']
                                binary_values = [
                                    binary_counts.get(0, 0),
                                    binary_counts.get(1, 0),
                                    binary_counts.get(-1, 0)
                                ]
                                
                                bars1 = ax1.bar(range(len(binary_values)), binary_values, 
                                               color=colors_binary[:len(binary_values)], 
                                               edgecolor='black', linewidth=1)
                                ax1.set_xlabel('Binary Labels')
                                ax1.set_ylabel('Count')
                                ax1.set_title('Binary Label Distribution (0/1)')
                                ax1.set_xticks(range(len(binary_labels)))
                                ax1.set_xticklabels(binary_labels, fontsize=9)
                                
                                # Add count labels on bars
                                for bar, count in zip(bars1, binary_values):
                                    height = bar.get_height()
                                    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                                            f'{count}', ha='center', va='bottom', fontsize=9)
                                
                                # Pie chart for readable labels
                                if len(readable_counts) > 0:
                                    colors_pie = plt.cm.Set3(np.linspace(0, 1, len(readable_counts)))
                                    wedges, texts, autotexts = ax2.pie(readable_counts.values, 
                                                                      labels=readable_counts.index, 
                                                                      autopct='%1.1f%%',
                                                                      startangle=90,
                                                                      colors=colors_pie,
                                                                      textprops={'fontsize': 9})
                                    ax2.set_title('Original Label Distribution')
                                    
                                    for autotext in autotexts:
                                        autotext.set_color('black')
                                        autotext.set_fontsize(9)
                                        autotext.set_fontweight('bold')
                                else:
                                    ax2.text(0.5, 0.5, 'No labels found', 
                                            ha='center', va='center')
                                    ax2.set_title('No Label Data')
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                    else:
                        st.warning("No data found. Try adjusting date range.")
        
        # Show existing data if available
        if not st.session_state['scraped_df'].empty:
            st.markdown("---")
            st.subheader("Current Data Overview")
            
            # Show which columns are being used
            if st.session_state.get('csv_columns_selected', False):
                st.info(f"**Text Column:** '{st.session_state['selected_text_column']}' | **Label Column:** '{st.session_state['selected_label_column']}'")
                if st.session_state.get('selected_date_column'):
                    st.info(f"**Date Column:** '{st.session_state['selected_date_column']}'")
            
            # Show filtered data if available, otherwise show all data
            display_df = st.session_state.get('filtered_df', st.session_state['scraped_df'])
            
            st.dataframe(display_df.head(10), use_container_width=True)
            
            # Statistics
            stats_col1, stats_col2, stats_col3 = st.columns(3)
            with stats_col1:
                st.metric("Total Claims", len(display_df))
            with stats_col2:
                if 'binary_label' in display_df.columns:
                    true_count = len(display_df[display_df['binary_label'] == 1])
                    false_count = len(display_df[display_df['binary_label'] == 0])
                    st.metric("True/False Ratio", f"{true_count}/{false_count}")
                else:
                    unique_labels = display_df['label'].nunique()
                    st.metric("Unique Labels", unique_labels)
            with stats_col3:
                if 'date' in display_df.columns:
                    try:
                        dates = pd.to_datetime(display_df['date'], errors='coerce')
                        valid_dates = dates.dropna()
                        if len(valid_dates) > 0:
                            date_range = valid_dates.agg(['min', 'max'])
                            st.metric("Date Range", f"{date_range['min'].strftime('%Y-%m-%d')} to {date_range['max'].strftime('%Y-%m-%d')}")
                        else:
                            st.metric("Date Info", "Invalid dates")
                    except:
                        st.metric("Date Info", "Not available")

    # --- GOOGLE API VERIFICATION (DEDICATED SECTION) ---
    elif page == "Google API Verification":
        st.markdown("<h1 class='main-header'>Google API Verification</h1>", unsafe_allow_html=True)
        
        # Get Google API key
        api_key = get_google_api_key()
        if not api_key:
            st.stop()
        
        # Check if we have data to verify
        if st.session_state['scraped_df'].empty:
            st.warning("No data available for verification. Please collect data first from the Data Collection page.")
            if st.button("Go to Data Collection", use_container_width=True):
                st.switch_page("Data Collection")
            st.stop()
        
        # Use filtered data if available, otherwise use scraped data
        verification_data = st.session_state.get('filtered_df', st.session_state['scraped_df'])
        
        st.info(f"Ready to verify {len(verification_data)} claims using Google Fact Check API")
        
        # Verification options in tabs
        tab1, tab2 = st.tabs(["Batch Verification", "Single Claim Verification"])
        
        with tab1:
            st.subheader("Batch Verification")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                max_claims = st.slider("Maximum claims to verify", 10, 1000, 30, 5, 
                                      help="Limit the number of claims to avoid rate limiting")
            
            with col2:
                verification_mode = st.selectbox(
                    "Verification Mode",
                    ["All Claims", "Only Unverified Claims", "Only False Claims", "Only True Claims"]
                )
            
            # Apply verification mode filter
            filtered_for_verification = verification_data.copy()
            
            if verification_mode == "Only Unverified Claims":
                # Check if claims have been verified before
                if not st.session_state['verification_df'].empty:
                    verified_statements = set(st.session_state['verification_df']['original_statement'].str.lower())
                    filtered_for_verification = filtered_for_verification[
                        ~filtered_for_verification['statement'].str.lower().isin(verified_statements)
                    ]
                st.info(f"Found {len(filtered_for_verification)} unverified claims")
            
            elif verification_mode == "Only False Claims" and 'binary_label' in filtered_for_verification.columns:
                filtered_for_verification = filtered_for_verification[filtered_for_verification['binary_label'] == 0]
                st.info(f"Found {len(filtered_for_verification)} false claims")
            
            elif verification_mode == "Only True Claims" and 'binary_label' in filtered_for_verification.columns:
                filtered_for_verification = filtered_for_verification[filtered_for_verification['binary_label'] == 1]
                st.info(f"Found {len(filtered_for_verification)} true claims")
            
            # Show sample claims to be verified
            with st.expander("Preview Claims to Verify", expanded=True):
                display_cols = ['statement', 'label']
                if 'binary_label' in filtered_for_verification.columns:
                    display_cols.append('binary_label')
                st.dataframe(filtered_for_verification[display_cols].head(10), use_container_width=True)
            
            if st.button("Start Batch Verification", key="batch_verify_btn", use_container_width=True, type="primary"):
                if len(filtered_for_verification) == 0:
                    st.error("No claims to verify with the selected mode")
                else:
                    with st.spinner(f"Verifying up to {min(max_claims, len(filtered_for_verification))} claims with Google API (this may take a while)..."):
                        verification_df = verify_batch_claims_with_google(
                            filtered_for_verification,
                            api_key,
                            max_verifications=max_claims,
                            text_column='statement'
                        )
                        
                        if not verification_df.empty:
                            # Update session state
                            st.session_state['verification_df'] = verification_df
                            st.session_state['verification_performed'] = True
                            
                            # If we have filtered data, also store filtered verification
                            if 'filtered_df' in st.session_state and not st.session_state['filtered_df'].empty:
                                st.session_state['filtered_verification_df'] = verification_df
                            
                            st.success("Google API verification completed!")
                            
                            # Show verification results
                            show_verification_results(verification_df)
                        else:
                            st.error("Verification failed. Please check your API key and try again.")
        
        with tab2:
            st.subheader("Single Claim Verification")
            
            # Option to enter custom claim or select from dataset
            verify_option = st.radio(
                "Choose claim source:",
                ["Enter Custom Claim", "Select from Dataset"],
                horizontal=True
            )
            
            if verify_option == "Enter Custom Claim":
                custom_claim = st.text_area(
                    "Enter a claim to verify:",
                    placeholder="e.g., The Earth is flat",
                    height=100
                )
                
                if custom_claim:
                    if st.button("Verify Custom Claim", key="custom_verify_btn", use_container_width=True):
                        with st.spinner("Verifying claim with Google API..."):
                            result = verify_single_claim_with_google(custom_claim, api_key)
                            
                            # Display results
                            st.subheader("Verification Results")
                            
                            if result['verification_status'] == 'Found':
                                st.success(f"Claim verified!")
                                
                                # Create a nice display
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Google Rating", result['google_rating'])
                                    st.metric("Match Score", f"{result['match_score']:.2%}")
                                    st.metric("Confidence", result['verification_confidence'])
                                
                                with col2:
                                    st.metric("Publisher", result['google_publisher'])
                                    if result['google_binary_label'] != -1:
                                        label_text = "True" if result['google_binary_label'] == 1 else "False"
                                        st.metric("Binary Label", label_text)
                                
                                # Show details
                                with st.expander("Detailed Information"):
                                    st.write(f"**Original Claim:** {result['claim_text']}")
                                    st.write(f"**Google's Version:** {result['google_claim_text']}")
                                    if result['verification_url']:
                                        st.write(f"**Source:** [View Verification]({result['verification_url']})")
                                    if result['google_claim_date']:
                                        st.write(f"**Claim Date:** {result['google_claim_date']}")
                                    if result['google_review_date']:
                                        st.write(f"**Review Date:** {result['google_review_date']}")
                                
                            elif result['verification_status'] == 'Rate Limited':
                                st.warning("Rate limited. Please try again later.")
                            else:
                                st.warning("Claim not found in Google Fact Check database.")
            
            else:  # Select from dataset
                if not verification_data.empty:
                    claim_options = verification_data['statement'].tolist()[:50]  # Limit to first 50
                    selected_claim = st.selectbox(
                        "Select a claim from your dataset:",
                        claim_options,
                        key="dataset_claim_select"
                    )
                    
                    if selected_claim and st.button("Verify Selected Claim", key="dataset_verify_btn", use_container_width=True):
                        with st.spinner("Verifying claim with Google API..."):
                            result = verify_single_claim_with_google(selected_claim, api_key)
                            
                            # Display results
                            st.subheader("Verification Results")
                            
                            if result['verification_status'] == 'Found':
                                st.success(f"Claim verified!")
                                
                                # Get original label for comparison
                                original_row = verification_data[verification_data['statement'] == selected_claim].iloc[0]
                                
                                # Create comparison
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Original Label", original_row.get('label', 'N/A'))
                                    if 'binary_label' in original_row:
                                        st.metric("Original Binary", "True" if original_row['binary_label'] == 1 else "False" if original_row['binary_label'] == 0 else "Unknown")
                                
                                with col2:
                                    st.metric("Google Rating", result['google_rating'])
                                    if result['google_binary_label'] != -1:
                                        label_text = "True" if result['google_binary_label'] == 1 else "False"
                                        st.metric("Google Binary", label_text)
                                
                                with col3:
                                    st.metric("Match Score", f"{result['match_score']:.2%}")
                                    st.metric("Confidence", result['verification_confidence'])
                                
                                # Show agreement/disagreement
                                if 'binary_label' in original_row and result['google_binary_label'] != -1:
                                    if original_row['binary_label'] == result['google_binary_label']:
                                        st.success("Labels agree!")
                                    elif original_row['binary_label'] == -1:
                                        st.info("Original label was unknown")
                                    else:
                                        st.warning("Labels disagree!")
                                
                                # Show details
                                with st.expander("Detailed Information"):
                                    st.write(f"**Original Claim:** {result['claim_text']}")
                                    st.write(f"**Google's Version:** {result['google_claim_text']}")
                                    if result['verification_url']:
                                        st.write(f"**Source:** [View Verification]({result['verification_url']})")
                                    if result['google_publisher']:
                                        st.write(f"**Publisher:** {result['google_publisher']}")
                            
                            elif result['verification_status'] == 'Rate Limited':
                                st.warning("Rate limited. Please try again later.")
                            else:
                                st.warning("Claim not found in Google Fact Check database.")
                else:
                    st.info("No dataset available. Please load data first.")
        
        # Show existing verification results if available
        if st.session_state.get('verification_performed', False):
            st.markdown("---")
            st.subheader("Existing Verification Results")
            
            verification_df = st.session_state.get('filtered_verification_df', st.session_state.get('verification_df', pd.DataFrame()))
            
            if not verification_df.empty:
                # Summary statistics
                found_count = len(verification_df[verification_df['google_verification'] == 'Found'])
                total_count = len(verification_df)
                verification_rate = (found_count / total_count * 100) if total_count > 0 else 0
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Claims Checked", total_count)
                with col2:
                    st.metric("Claims Verified", found_count)
                with col3:
                    st.metric("Verification Rate", f"{verification_rate:.1f}%")
                
                # Show detailed results
                show_verification_results(verification_df)
                
                # Export option
                if st.button("Export Verification Results", key="export_verify_btn"):
                    csv = verification_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="google_verification_results.csv",
                        mime="text/csv",
                        key="download_verification"
                    )
            else:
                st.info("No verification results available yet.")

    # --- MODEL TRAINING ---
    elif page == "Model Training":
        st.markdown("<h1 class='main-header'>Model Training</h1>", unsafe_allow_html=True)
        
        # Use filtered data if available, otherwise use scraped data
        training_df = st.session_state.get('filtered_df', st.session_state['scraped_df'])
        
        if training_df.empty:
            st.warning("Please collect data first from the Data Collection page!")
            if st.button("Go to Data Collection", use_container_width=True):
                st.switch_page("Data Collection")
        else:
            # Show which columns are being used
            if st.session_state.get('csv_columns_selected', False):
                st.info(f"**Using:** Text from column '{st.session_state['selected_text_column']}' and Labels from column '{st.session_state['selected_label_column']}'")
                if len(training_df) != len(st.session_state['scraped_df']):
                    st.info(f"**Note:** Using filtered data ({len(training_df)} claims) instead of full dataset ({len(st.session_state['scraped_df'])} claims)")
            
            # Show verification status
            if st.session_state.get('verification_performed', False):
                verification_df = st.session_state.get('filtered_verification_df', st.session_state.get('verification_df', pd.DataFrame()))
                if not verification_df.empty:
                    found_count = len(verification_df[verification_df['google_verification'] == 'Found'])
                    st.success(f"Google API Verification available: {found_count} verified claims")
            
            # Training configuration
            st.write("Configure and train machine learning models using different NLP feature extraction methods.")
            
            config_col1, config_col2 = st.columns(2)
            with config_col1:
                phases = ["Lexical & Morphological", "Syntactic", "Semantic", "Discourse", "Pragmatic"]
                selected_phase = st.selectbox("Feature Extraction Method:", phases, key='selected_phase')
            
            with config_col2:
                use_smote = st.checkbox("Apply SMOTE for class balancing", value=True)
                st.session_state['use_smote'] = use_smote
            
            # Add option to use verified data if available
            use_verified_data = False
            if st.session_state.get('verification_performed', False):
                verification_df = st.session_state.get('filtered_verification_df', st.session_state.get('verification_df', pd.DataFrame()))
                verified_count = 0
                if not verification_df.empty:
                    verified_claims = verification_df[verification_df['google_verification'] == 'Found']
                    verified_count = len(verified_claims)
                
                if verified_count > 0:
                    use_verified_data = st.checkbox(f"Use Google-verified claims only ({verified_count} available)", value=False)
                else:
                    st.warning("Google API verification was performed but no verified claims were found.")
            
            # Data preview
            with st.expander("Training Data Preview", expanded=False):
                if use_verified_data and st.session_state.get('verification_performed', False):
                    verification_df = st.session_state.get('filtered_verification_df', st.session_state.get('verification_df', pd.DataFrame()))
                    verified_claims = verification_df[verification_df['google_verification'] == 'Found']
                    st.info(f"Using {len(verified_claims)} Google-verified claims for training")
                    display_cols = ['original_statement', 'original_label', 'google_rating', 'match_score', 'verification_confidence']
                    st.dataframe(verified_claims[display_cols].head(10), use_container_width=True)
                else:
                    display_cols = ['statement', 'label']
                    if 'binary_label' in training_df.columns:
                        display_cols.append('binary_label')
                    if 'label_readable' in training_df.columns:
                        display_cols.append('label_readable')
                    st.dataframe(training_df[display_cols].head(10), use_container_width=True)
                
                # Show clear binary class distribution
                if 'binary_label' in training_df.columns:
                    binary_counts = training_df['binary_label'].value_counts()
                    
                    fig, ax = plt.subplots(figsize=(8, 5))
                    
                    # Simple bar chart for binary labels
                    colors = ['#F44336', '#4CAF50', '#9E9E9E']  # Red (False), Green (True), Gray (Unknown)
                    labels = ['False (0)', 'True (1)', 'Unknown']
                    values = [
                        binary_counts.get(0, 0),
                        binary_counts.get(1, 0),
                        binary_counts.get(-1, 0)
                    ]
                    
                    bars = ax.bar(range(len(values)), values, color=colors[:len(values)], 
                                edgecolor='black', linewidth=1)
                    ax.set_title('Binary Label Distribution for Model Training', fontsize=12)
                    ax.set_ylabel('Count', fontsize=10)
                    ax.set_xlabel('Binary Label', fontsize=10)
                    ax.set_xticks(range(len(labels)))
                    ax.set_xticklabels(labels, fontsize=10)
                    
                    # Add count labels
                    for bar, count in zip(bars, values):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                               f'{count}', ha='center', va='bottom', fontsize=10)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
            
            if st.button("Run Model Analysis", key="analyze_btn", use_container_width=True, type="primary"):
                with st.spinner(f"Training 4 models with {N_SPLITS}-Fold CV..."):
                    # Select appropriate data source
                    if use_verified_data and st.session_state.get('verification_performed', False):
                        # Use verified data
                        verification_df = st.session_state.get('filtered_verification_df', st.session_state.get('verification_df', pd.DataFrame()))
                        verified_df = verification_df[verification_df['google_verification'] == 'Found']
                        
                        # Create a DataFrame compatible with evaluate_models
                        training_data = pd.DataFrame({
                            'statement': verified_df['original_statement'],
                            'label': verified_df['original_label']
                        })
                        
                        # Try to get binary labels from Google verification
                        if 'google_binary_label' in verified_df.columns:
                            training_data['binary_label'] = verified_df['google_binary_label']
                            # Filter out unknown labels
                            training_data = training_data[training_data['binary_label'] != -1]
                        
                        st.info(f"Using {len(training_data)} Google-verified claims for training")
                    else:
                        # Use original/filtered data
                        training_data = training_df.copy()
                    
                    df_results, trained_models, trained_vectorizer = evaluate_models(
                        training_data, 
                        selected_phase, 
                        text_column='statement',
                        label_column='label',
                        use_smote=use_smote
                    )
                    
                    if not df_results.empty:
                        st.session_state['df_results'] = df_results
                        st.session_state['trained_models'] = trained_models
                        st.session_state['trained_vectorizer'] = trained_vectorizer
                        st.session_state['selected_phase_run'] = selected_phase
                        
                        # Count successfully trained models
                        successful_models = sum(1 for m in trained_models.values() if m is not None)
                        st.success(f"Analysis complete! {successful_models} out of 4 models trained and saved to disk.")
                        
                        # Show immediate results
                        st.subheader("Training Results")
                        st.dataframe(df_results, use_container_width=True)
                    else:
                        st.error("Model training failed. Please check your data and try again.")

    # --- RESULTS & ANALYSIS ---
elif page == "Results & Analysis":
    st.markdown("<h1 class='main-header'>Results & Analysis</h1>", unsafe_allow_html=True)
    
    # Check if we have results
    if st.session_state['df_results'].empty:
        st.warning("No model results available. Please train models first.")
        if st.button("Go to Model Training", use_container_width=True):
            st.switch_page("Model Training")
    else:
        st.write("Analyze model performance and make predictions on new claims.")
        
        # Display results in tabs
        tab1, tab2 = st.tabs(["Performance Results", "Make Predictions"])
        
        with tab1:
            st.subheader("Model Performance")
            df_results = st.session_state['df_results']
            selected_phase = st.session_state['selected_phase_run']
            
            # Display results table
            st.dataframe(df_results, use_container_width=True)
            
            # Visualization - MOVED UP
            st.subheader("Performance Visualization")
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Accuracy comparison
            axes[0, 0].bar(df_results['Model'], df_results['Accuracy'], color='#2196F3')
            axes[0, 0].set_title('Model Accuracy (%)')
            axes[0, 0].set_ylabel('Accuracy %')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].grid(True, alpha=0.3)
            
            # F1-Score comparison
            axes[0, 1].bar(df_results['Model'], df_results['F1-Score'], color='#4CAF50')
            axes[0, 1].set_title('F1-Score Comparison')
            axes[0, 1].set_ylabel('F1-Score')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3)
            
            # Precision-Recall comparison
            x = range(len(df_results))
            width = 0.35
            axes[1, 0].bar([i - width/2 for i in x], df_results['Precision'], width, label='Precision', color='#FF9800')
            axes[1, 0].bar([i + width/2 for i in x], df_results['Recall'], width, label='Recall', color='#9C27B0')
            axes[1, 0].set_title('Precision vs Recall')
            axes[1, 0].set_ylabel('Score')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(df_results['Model'], rotation=45)
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Inference latency
            axes[1, 1].bar(df_results['Model'], df_results['Inference Latency (ms)'], color='#F44336')
            axes[1, 1].set_title('Inference Latency (ms)')
            axes[1, 1].set_ylabel('Milliseconds')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Generate humorous critique - MOVED DOWN
            st.subheader("AI Performance Analysis")
            critique = generate_humorous_critique(df_results, selected_phase)
            st.markdown(critique)
            
        with tab2:
            st.subheader("Make Predictions")
            
            # Input for single text prediction
            input_text = st.text_area(
                "Enter a claim to analyze:",
                placeholder="e.g., The moon landing was faked by NASA",
                height=100
            )
            
            if input_text and st.button("Analyze Claim", key="predict_btn", use_container_width=True, type="primary"):
                if not st.session_state['trained_models']:
                    st.error("No trained models available. Please train models first.")
                else:
                    results = predict_single_text(
                        input_text,
                        st.session_state['trained_models'],
                        st.session_state['trained_vectorizer'],
                        st.session_state['selected_phase_run']
                    )
                    
                    # Display results
                    st.subheader("Prediction Results")
                    
                    # Create a nice display
                    cols = st.columns(4)
                    model_names = list(results.keys())
                    
                    for idx, col in enumerate(cols):
                        if idx < len(model_names):
                            model_name = model_names[idx]
                            result = results[model_name]
                            
                            with col:
                                if 'error' in result:
                                    st.error(f"{model_name}")
                                    st.write(result['error'])
                                else:
                                    prediction = result['prediction']
                                    label_text = "True" if prediction == 1 else "False" if prediction == 0 else "Unknown"
                                    color = "green" if prediction == 1 else "red" if prediction == 0 else "gray"
                                    
                                    st.markdown(f"""
                                    <div style="text-align: center; padding: 15px; border-radius: 8px; background-color: #f8f9fa; border: 2px solid {color};">
                                        <h4>{model_name}</h4>
                                        <h2 style="color: {color};">{label_text}</h2>
                                        <p>Prediction: {prediction}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                    
                    # Show consensus
                    st.subheader("Model Consensus")
                    predictions = [r.get('prediction', -1) for r in results.values() if 'error' not in r]
                    if predictions:
                        valid_predictions = [p for p in predictions if p in [0, 1]]
                        if valid_predictions:
                            avg_prediction = sum(valid_predictions) / len(valid_predictions)
                            consensus = "True" if avg_prediction > 0.5 else "False" if avg_prediction < 0.5 else "Tie"
                            confidence = abs(avg_prediction - 0.5) * 2 * 100  # Convert to percentage
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Consensus", consensus)
                            with col2:
                                st.metric("Confidence", f"{confidence:.1f}%")

if __name__ == '__main__':
    app()

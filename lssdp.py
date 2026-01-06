# lssdp.py
# Updated Streamlit app with blue/white theme, larger navigation menu, and optional SMOTE section.
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
# GOOGLE FACT CHECK INTEGRATION
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
                st.error("Invalid API key. Please check your GOOGLE_API_KEY in .streamlit/secrets.toml")
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
# WEB SCRAPING (Politifact)
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
        vectorizer = vectorizer if vectorizer else CountVectorizer(binary=True, ngram_range=(1,2))
        X_features = vectorizer.fit_transform(X_processed)
        return X_features, vectorizer
    elif phase == "Syntactic":
        X_processed = X.apply(syntactic_features)
        vectorizer = vectorizer if vectorizer else TfidfVectorizer(max_features=5000)
        X_features = vectorizer.fit_transform(X_processed)
        return X_features, vectorizer
    elif phase == "Semantic":
        X_features = pd.DataFrame(X.apply(semantic_features).tolist(), columns=["polarity", "subjectivity"])
        return X_features, None
    elif phase == "Discourse":
        X_processed = X.apply(discourse_features)
        vectorizer = vectorizer if vectorizer else CountVectorizer(ngram_range=(1,2), max_features=5000)
        X_features = vectorizer.fit_transform(X_processed)
        return X_features, vectorizer
    elif phase == "Pragmatic":
        X_features = pd.DataFrame(X.apply(pragmatic_features).tolist(), columns=pragmatic_words)
        return X_features, None
    return None, None

# --------------------------
# Save & Load trained models (joblib)
# --------------------------
def save_trained_models(trained_models: dict, vectorizer, selected_phase):
    try:
        MODELS_DIR.mkdir(exist_ok=True)
        # save each model separately using joblib
        for name, model in trained_models.items():
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
        return {}
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
            X_features = pd.DataFrame(X_raw_series.apply(semantic_features).tolist(), columns=["polarity", "subjectivity"]).values
        elif selected_phase == "Pragmatic":
            X_features = pd.DataFrame(X_raw_series.apply(pragmatic_features).tolist(), columns=pragmatic_words).values
        else:
            raise ValueError(f"Unknown selected_phase: {selected_phase}")
    except Exception as e:
        return {"error": f"Feature extraction failed: {e}"}
    results = {}
    for model_name, model in trained_models.items():
        if model is None:
            results[model_name] = {"error": "Model unavailable"}
            continue
        try:
            if model_name == "Naive Bayes":
                X_for_model = np.abs(X_features).astype(float)
            else:
                X_for_model = X_features
            pred = model.predict(X_for_model)
            label = int(pred[0]) if hasattr(pred, "__len__") else int(pred)
            results[model_name] = {"prediction": label}
        except Exception as e:
            # Try conversion to dense
            try:
                X_alt = X_for_model.toarray() if hasattr(X_for_model, "toarray") else X_for_model
                pred = model.predict(X_alt)
                label = int(pred[0]) if hasattr(pred, "__len__") else int(pred)
                results[model_name] = {"prediction": label}
            except Exception as e2:
                results[model_name] = {"error": f"Prediction failed: {e2}"}
    return results

# --------------------------
# Model training & evaluation (K-Fold & optional SMOTE)
# --------------------------
def evaluate_models(df: pd.DataFrame, selected_phase: str, use_smote: bool = True):
    REAL_LABELS = ["True", "No Flip", "Mostly True", "Half Flip", "Half True"]
    FAKE_LABELS = ["False", "Barely True", "Pants On Fire", "Full Flop"]

    def create_binary_target(label):
        if label in REAL_LABELS:
            return 1
        elif label in FAKE_LABELS:
            return 0
        else:
            return np.nan

    df['target_label'] = df['label'].apply(create_binary_target)
    df = df.dropna(subset=['target_label'])
    df = df[df['statement'].astype(str).str.len() > 10]
    X_raw = df['statement'].astype(str)
    y_raw = df['target_label'].astype(int)
    if len(np.unique(y_raw)) < 2:
        st.error("After binary mapping, only one class remains (all Real or all Fake). Cannot train classifier.")
        return pd.DataFrame(), {}, None

    X_features_full, vectorizer = apply_feature_extraction(X_raw, selected_phase)
    if X_features_full is None:
        st.error("Feature extraction failed.")
        return pd.DataFrame(), {}, None
    if isinstance(X_features_full, pd.DataFrame):
        X_features_full = X_features_full.values
    y = y_raw.values
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    models_to_run = {
        "Naive Bayes": MultinomialNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        "Logistic Regression": LogisticRegression(max_iter=1000, solver='liblinear', random_state=42, class_weight='balanced'),
        "SVM": SVC(kernel='linear', C=0.5, random_state=42, class_weight='balanced')
    }
    model_metrics = {name: [] for name in models_to_run.keys()}
    X_raw_list = X_raw.tolist()

    for name, model in models_to_run.items():
        st.caption(f"Training {name} with {N_SPLITS}-Fold CV{' & SMOTE' if use_smote else ''}...")
        fold_metrics = {'accuracy': [], 'f1': [], 'precision': [], 'recall': [], 'train_time': [], 'inference_time': []}
        for fold, (train_index, test_index) in enumerate(skf.split(X_features_full, y)):
            X_train_raw = pd.Series([X_raw_list[i] for i in train_index])
            X_test_raw = pd.Series([X_raw_list[i] for i in test_index])
            y_train = y[train_index]
            y_test = y[test_index]
            if vectorizer is not None:
                if 'Lexical' in selected_phase:
                    X_train = vectorizer.transform(X_train_raw.apply(lexical_features))
                    X_test = vectorizer.transform(X_test_raw.apply(lexical_features))
                elif 'Syntactic' in selected_phase:
                    X_train = vectorizer.transform(X_train_raw.apply(syntactic_features))
                    X_test = vectorizer.transform(X_test_raw.apply(syntactic_features))
                elif 'Discourse' in selected_phase:
                    X_train = vectorizer.transform(X_train_raw.apply(discourse_features))
                    X_test = vectorizer.transform(X_test_raw.apply(discourse_features))
                else:
                    # default transform
                    X_train = vectorizer.transform(X_train_raw.apply(lexical_features if 'Lexical' in selected_phase else lexic))
                    X_test = vectorizer.transform(X_test_raw.apply(lexical_features if 'Lexical' in selected_phase else lexic))
            else:
                X_train, _ = apply_feature_extraction(X_train_raw, selected_phase)
                X_test, _ = apply_feature_extraction(X_test_raw, selected_phase)
            start_time = time.time()
            try:
                if name == "Naive Bayes":
                    X_train_final = np.abs(X_train).astype(float)
                    clf = model
                    clf.fit(X_train_final, y_train)
                else:
                    if use_smote:
                        smote_pipeline = ImbPipeline([('sampler', SMOTE(random_state=42, k_neighbors=3)), ('classifier', model)])
                        smote_pipeline.fit(X_train, y_train)
                        clf = smote_pipeline
                    else:
                        clf = model
                        clf.fit(X_train, y_train)
                train_time = time.time() - start_time
                start_inference = time.time()
                y_pred = clf.predict(X_test)
                inference_time = (time.time() - start_inference) * 1000
                fold_metrics['accuracy'].append(accuracy_score(y_test, y_pred))
                fold_metrics['f1'].append(f1_score(y_test, y_pred, average='weighted', zero_division=0))
                fold_metrics['precision'].append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
                fold_metrics['recall'].append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
                fold_metrics['train_time'].append(train_time)
                fold_metrics['inference_time'].append(inference_time)
            except Exception as e:
                st.warning(f"Fold {fold+1} failed for {name}: {e}")
                for key in fold_metrics:
                    fold_metrics[key].append(0)
                continue
        if fold_metrics['accuracy']:
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
            st.error(f"{name} failed across all folds.")
            model_metrics[name] = {
                "Model": name, "Accuracy": 0, "F1-Score": 0, "Precision": 0, "Recall": 0,
                "Training Time (s)": 0, "Inference Latency (ms)": 9999,
            }

    # Train final models on full dataset and save them
    st.caption("Training final models on complete dataset for benchmarking...")
    trained_models_final = {}
    for name in models_to_run.keys():
        try:
            final_model = get_classifier(name)
            if vectorizer is not None:
                if 'Lexical' in selected_phase:
                    X_final_processed = X_raw.apply(lexical_features)
                elif 'Syntactic' in selected_phase:
                    X_final_processed = X_raw.apply(syntactic_features)
                elif 'Discourse' in selected_phase:
                    X_final_processed = X_raw.apply(discourse_features)
                else:
                    X_final_processed = X_raw
                X_final = vectorizer.transform(X_final_processed)
            else:
                X_final = X_features_full
            if name == "Naive Bayes":
                X_final_train = np.abs(X_final).astype(float)
                final_model.fit(X_final_train, y)
                trained_models_final[name] = final_model
            else:
                if use_smote:
                    smote_pipeline_final = ImbPipeline([('sampler', SMOTE(random_state=42, k_neighbors=3)), ('classifier', final_model)])
                    smote_pipeline_final.fit(X_final, y)
                    trained_models_final[name] = smote_pipeline_final
                else:
                    final_model.fit(X_final, y)
                    trained_models_final[name] = final_model
        except Exception as e:
            st.warning(f"Failed to train final {name} model: {e}")
            trained_models_final[name] = None

    # Save trained models & vectorizer
    try:
        save_trained_models(trained_models_final, vectorizer, selected_phase)
    except Exception as e:
        st.warning(f"Saving after training failed: {e}")

    results_list = list(model_metrics.values())
    return pd.DataFrame(results_list), trained_models_final, vectorizer

# --------------------------
# SMOTE Analysis Section
# --------------------------
def analyze_smote_impact(df: pd.DataFrame, selected_phase: str):
    """
    Analyze and compare performance with and without SMOTE
    """
    st.subheader("⚖️ SMOTE Impact Analysis")
    
    with st.expander("📊 Class Distribution Before/After SMOTE", expanded=True):
        REAL_LABELS = ["True", "No Flip", "Mostly True", "Half Flip", "Half True"]
        FAKE_LABELS = ["False", "Barely True", "Pants On Fire", "Full Flop"]
        
        def create_binary_target(label):
            if label in REAL_LABELS:
                return 1
            elif label in FAKE_LABELS:
                return 0
            else:
                return np.nan
        
        df_clean = df.copy()
        df_clean['target_label'] = df_clean['label'].apply(create_binary_target)
        df_clean = df_clean.dropna(subset=['target_label'])
        df_clean = df_clean[df_clean['statement'].astype(str).str.len() > 10]
        
        X_raw = df_clean['statement'].astype(str)
        y_raw = df_clean['target_label'].astype(int)
        
        if len(np.unique(y_raw)) < 2:
            st.error("Need both classes for SMOTE analysis")
            return
        
        # Get features for a sample
        X_sample, _ = apply_feature_extraction(X_raw.sample(min(100, len(X_raw)), random_state=42), selected_phase)
        if hasattr(X_sample, "toarray"):
            X_sample = X_sample.toarray()
        
        # Show class distribution
        col1, col2 = st.columns(2)
        with col1:
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            class_counts = pd.Series(y_raw).value_counts()
            colors = ['#1e88e5', '#ff7043']
            ax1.bar(['Real (1)', 'Fake (0)'], class_counts.values, color=colors)
            ax1.set_title('Original Class Distribution')
            ax1.set_ylabel('Count')
            ax1.set_xlabel('Class')
            for i, v in enumerate(class_counts.values):
                ax1.text(i, v + 0.5, str(v), ha='center', fontweight='bold')
            st.pyplot(fig1)
        
        # Apply SMOTE to show balanced distribution
        with col2:
            try:
                smote = SMOTE(random_state=42, k_neighbors=3)
                X_res, y_res = smote.fit_resample(X_sample, y_raw[:len(X_sample)])
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                res_counts = pd.Series(y_res).value_counts()
                ax2.bar(['Real (1)', 'Fake (0)'], res_counts.values, color=colors)
                ax2.set_title('After SMOTE Resampling')
                ax2.set_ylabel('Count')
                ax2.set_xlabel('Class')
                for i, v in enumerate(res_counts.values):
                    ax2.text(i, v + 0.5, str(v), ha='center', fontweight='bold')
                st.pyplot(fig2)
            except Exception as e:
                st.warning(f"Could not visualize SMOTE: {e}")
    
    # Run comparison with and without SMOTE
    st.subheader("📈 Performance Comparison: With vs Without SMOTE")
    
    progress_bar = st.progress(0)
    results_comparison = []
    
    for i, use_smote_flag in enumerate([False, True]):
        progress_bar.progress((i + 1) * 50)
        
        # Run evaluation
        try:
            df_results, _, _ = evaluate_models(df, selected_phase, use_smote=use_smote_flag)
            for _, row in df_results.iterrows():
                results_comparison.append({
                    'Model': row['Model'],
                    'SMOTE': 'With SMOTE' if use_smote_flag else 'Without SMOTE',
                    'Accuracy': row['Accuracy'],
                    'F1-Score': row['F1-Score'],
                    'Precision': row['Precision'],
                    'Recall': row['Recall'],
                    'Training Time (s)': row['Training Time (s)']
                })
        except Exception as e:
            st.warning(f"Failed to evaluate with SMOTE={use_smote_flag}: {e}")
    
    progress_bar.empty()
    
    if results_comparison:
        comparison_df = pd.DataFrame(results_comparison)
        
        # Display comparison table
        st.dataframe(comparison_df, use_container_width=True)
        
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # F1-Score comparison
        pivot_f1 = comparison_df.pivot(index='Model', columns='SMOTE', values='F1-Score')
        pivot_f1.plot(kind='bar', ax=axes[0], color=['#1e88e5', '#ff7043'])
        axes[0].set_title('F1-Score: With vs Without SMOTE')
        axes[0].set_ylabel('F1-Score')
        axes[0].legend(title='SMOTE Status')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Accuracy comparison
        pivot_acc = comparison_df.pivot(index='Model', columns='SMOTE', values='Accuracy')
        pivot_acc.plot(kind='bar', ax=axes[1], color=['#1e88e5', '#ff7043'])
        axes[1].set_title('Accuracy: With vs Without SMOTE')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].legend(title='SMOTE Status')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # SMOTE Recommendations
        st.subheader("💡 SMOTE Recommendations")
        with st.container():
            st.markdown("""
            **When to use SMOTE:**
            ✅ Class imbalance is severe (>70/30 split)  
            ✅ You have enough minority class samples (at least 50)  
            ✅ Your models are consistently biased toward majority class
            
            **When to avoid SMOTE:**
            ❌ Classes are already balanced  
            ❌ Very small dataset (<100 samples)  
            ❌ Minority class has insufficient genuine samples
            """)
            
            # Auto-recommendation
            class_counts = pd.Series(y_raw).value_counts()
            imbalance_ratio = min(class_counts) / max(class_counts)
            
            if imbalance_ratio < 0.3:
                st.success(f"**Recommendation: Use SMOTE** (Imbalance ratio: {imbalance_ratio:.2f})")
            else:
                st.info(f"**Recommendation: SMOTE optional** (Imbalance ratio: {imbalance_ratio:.2f})")

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
# STREAMLIT APP
# --------------------------
def app():
    st.set_page_config(page_title='FactChecker: AI Fact-Checking Platform', layout='wide', initial_sidebar_state='expanded')

    # BLUE AND WHITE THEME CSS
    st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --primary-blue: #1e88e5;
        --secondary-blue: #1565c0;
        --light-blue: #bbdefb;
        --accent-blue: #0d47a1;
        --white: #ffffff;
        --light-gray: #f5f5f5;
        --dark-text: #212121;
    }
    
    /* Overall page styling */
    .stApp {
        background-color: #f8fbff;
        color: var(--dark-text);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #0d47a1;
        background: linear-gradient(180deg, #0d47a1 0%, #1e88e5 100%);
    }
    
    /* Navigation menu - LARGER FONT */
    section[data-testid="stSidebar"] .stRadio label {
        font-size: 18px !important;
        font-weight: 600 !important;
        color: white !important;
        padding: 12px 8px !important;
        margin: 4px 0 !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
    }
    
    section[data-testid="stSidebar"] .stRadio label:hover {
        background-color: rgba(255, 255, 255, 0.15) !important;
        transform: translateX(5px) !important;
    }
    
    section[data-testid="stSidebar"] .stRadio label div {
        font-size: 18px !important;
        font-weight: 600 !important;
    }
    
    /* Sidebar title */
    section[data-testid="stSidebar"] h1 {
        color: white !important;
        font-size: 28px !important;
        text-align: center !important;
        margin-bottom: 30px !important;
    }
    
    /* Cards and containers */
    .card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(30, 136, 229, 0.15);
        border-left: 5px solid var(--primary-blue);
        margin-bottom: 15px;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: var(--accent-blue) !important;
        border-bottom: 2px solid var(--light-blue);
        padding-bottom: 8px;
    }
    
    /* Buttons */
    .stButton button {
        background-color: var(--primary-blue) !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton button:hover {
        background-color: var(--secondary-blue) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(30, 136, 229, 0.3) !important;
    }
    
    /* Dataframes and tables */
    .stDataFrame {
        border: 1px solid var(--light-blue) !important;
        border-radius: 8px !important;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: var(--accent-blue) !important;
        font-size: 24px !important;
        font-weight: bold !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 8px 8px 0 0;
        border: 1px solid var(--light-blue);
        padding: 12px 24px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--primary-blue) !important;
        color: white !important;
    }
    
    /* Input fields */
    .stTextInput input, .stSelectbox select, .stDateInput input {
        border: 2px solid var(--light-blue) !important;
        border-radius: 8px !important;
        padding: 10px !important;
    }
    
    .stTextInput input:focus, .stSelectbox select:focus, .stDateInput input:focus {
        border-color: var(--primary-blue) !important;
        box-shadow: 0 0 0 3px rgba(30, 136, 229, 0.1) !important;
    }
    
    /* Success/Error messages */
    .stAlert {
        border-radius: 8px !important;
        border-left: 5px solid !important;
    }
    
    /* Progress bars */
    .stProgress > div > div > div {
        background-color: var(--primary-blue) !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: var(--light-blue) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
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
            st.success("Loaded saved models from disk. Single-text checks are available.")
        st.session_state['models_loaded_attempted'] = True

    # Sidebar and navigation with larger font
    st.sidebar.markdown("<h1 style='color:white; text-align:center;'>🔍 FactChecker</h1>", unsafe_allow_html=True)
    page = st.sidebar.radio("", ["Dashboard", "Data Collection", "Model Training", "SMOTE Analysis", "Benchmark Testing", "Results & Analysis"], 
                          key='navigation', label_visibility="collapsed")

    # --- DASHBOARD ---
    if page == "Dashboard":
        st.markdown("<h1 style='color:#0d47a1;'>📊 FactChecker Dashboard</h1>", unsafe_allow_html=True)
        
        # Status cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('''
            <div class="card">
                <h3 style="color:#0d47a1;">📁 Data Overview</h3>
                <p>Collect and manage training data from Politifact archives</p>
                <p><strong>Current Data:</strong> {} claims</p>
            </div>
            '''.format(len(st.session_state['scraped_df']) if not st.session_state['scraped_df'].empty else 0), 
            unsafe_allow_html=True)
        
        with col2:
            st.markdown('''
            <div class="card">
                <h3 style="color:#0d47a1;">🤖 Model Training</h3>
                <p>Advanced NLP feature extraction and ML training</p>
                <p><strong>Models Ready:</strong> {}</p>
            </div>
            '''.format(len(st.session_state['trained_models']) if st.session_state['trained_models'] else 0), 
            unsafe_allow_html=True)
        
        with col3:
            st.markdown('''
            <div class="card">
                <h3 style="color:#0d47a1;">⚖️ Benchmark Testing</h3>
                <p>Validate models with real-world data</p>
                <p><strong>Last Test:</strong> {} claims</p>
            </div>
            '''.format(len(st.session_state['google_df']) if not st.session_state['google_df'].empty else 0), 
            unsafe_allow_html=True)
        
        # Quick actions
        st.markdown("---")
        st.subheader("🚀 Quick Actions")
        
        quick_col1, quick_col2, quick_col3 = st.columns(3)
        with quick_col1:
            if st.button("Collect New Data", use_container_width=True):
                st.switch_page("Data Collection")
        
        with quick_col2:
            if st.button("Train Models", use_container_width=True):
                st.switch_page("Model Training")
        
        with quick_col3:
            if st.button("Run Benchmark", use_container_width=True):
                st.switch_page("Benchmark Testing")

    # --- DATA COLLECTION ---
    elif page == "Data Collection":
        st.markdown("<h1 style='color:#0d47a1;'>📁 Data Collection</h1>", unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("Collect political fact-check data from Politifact.com within a specific date range.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        min_date = pd.to_datetime('2007-01-01')
        max_date = pd.to_datetime('today').normalize()
        
        date_col1, date_col2 = st.columns(2)
        with date_col1:
            start_date = st.date_input("📅 Start Date", min_value=min_date, max_value=max_date, 
                                      value=pd.to_datetime('2023-01-01'), help="Select the start date for data collection")
        
        with date_col2:
            end_date = st.date_input("📅 End Date", min_value=min_date, max_value=max_date, 
                                    value=max_date, help="Select the end date for data collection")
        
        if st.button("🚀 Scrape Politifact Data", key="scrape_btn", use_container_width=True, type="primary"):
            if start_date > end_date:
                st.error("❌ Start date must be before end date")
            else:
                with st.spinner("🌐 Scraping political claims from Politifact..."):
                    scraped_df = scrape_data_by_date_range(pd.to_datetime(start_date), pd.to_datetime(end_date))
                
                if not scraped_df.empty:
                    st.session_state['scraped_df'] = scraped_df
                    st.success(f"✅ Successfully scraped {len(scraped_df)} claims!")
                    
                    # Show data preview
                    with st.expander("📋 Preview Scraped Data", expanded=True):
                        st.dataframe(st.session_state['scraped_df'].head(15), use_container_width=True)
                        
                        # Show label distribution
                        st.subheader("🏷️ Label Distribution")
                        label_counts = scraped_df['label'].value_counts()
                        col1, col2 = st.columns(2)
                        with col1:
                            st.dataframe(label_counts, use_container_width=True)
                        with col2:
                            fig, ax = plt.subplots(figsize=(8, 6))
                            colors = ['#1e88e5', '#ff7043', '#43a047', '#fb8c00', '#8e24aa']
                            ax.pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%', 
                                  colors=colors[:len(label_counts)], startangle=90)
                            ax.set_title('Claim Label Distribution')
                            st.pyplot(fig)
                else:
                    st.warning("⚠️ No data found. Try adjusting date range.")
        
        # Show existing data if available
        if not st.session_state['scraped_df'].empty:
            st.markdown("---")
            st.subheader("📊 Current Data Overview")
            st.dataframe(st.session_state['scraped_df'].head(10), use_container_width=True)
            
            # Statistics
            stats_col1, stats_col2, stats_col3 = st.columns(3)
            with stats_col1:
                st.metric("Total Claims", len(st.session_state['scraped_df']))
            with stats_col2:
                unique_labels = st.session_state['scraped_df']['label'].nunique()
                st.metric("Unique Labels", unique_labels)
            with stats_col3:
                date_range = pd.to_datetime(st.session_state['scraped_df']['date']).agg(['min', 'max'])
                st.metric("Date Range", f"{date_range['min'].strftime('%Y-%m-%d')} to {date_range['max'].strftime('%Y-%m-%d')}")

    # --- MODEL TRAINING ---
    elif page == "Model Training":
        st.markdown("<h1 style='color:#0d47a1;'>🤖 Model Training</h1>", unsafe_allow_html=True)
        
        if st.session_state['scraped_df'].empty:
            st.warning("⚠️ Please collect data first from the Data Collection page!")
            if st.button("Go to Data Collection", use_container_width=True):
                st.switch_page("Data Collection")
        else:
            # Training configuration
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.write("Configure and train machine learning models using different NLP feature extraction methods.")
            st.markdown('</div>', unsafe_allow_html=True)
            
            config_col1, config_col2 = st.columns(2)
            with config_col1:
                phases = ["Lexical & Morphological", "Syntactic", "Semantic", "Discourse", "Pragmatic"]
                selected_phase = st.selectbox("🔧 Feature Extraction Method:", phases, key='selected_phase',
                                             help="Choose the NLP feature extraction method to use")
            
            with config_col2:
                use_smote = st.checkbox("⚖️ Apply SMOTE for class balancing", value=True,
                                       help="Use SMOTE to handle class imbalance in training data")
                st.session_state['use_smote'] = use_smote
            
            # Data preview
            with st.expander("📊 Training Data Preview", expanded=False):
                st.dataframe(st.session_state['scraped_df'][['statement', 'label']].head(10), use_container_width=True)
                
                # Show class distribution for binary mapping
                REAL_LABELS = ["True", "No Flip", "Mostly True", "Half Flip", "Half True"]
                FAKE_LABELS = ["False", "Barely True", "Pants On Fire", "Full Flop"]
                
                def create_binary_target(label):
                    if label in REAL_LABELS:
                        return "Real"
                    elif label in FAKE_LABELS:
                        return "Fake"
                    else:
                        return "Other"
                
                df_preview = st.session_state['scraped_df'].copy()
                df_preview['binary_label'] = df_preview['label'].apply(create_binary_target)
                binary_counts = df_preview['binary_label'].value_counts()
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                
                # Original labels
                original_counts = st.session_state['scraped_df']['label'].value_counts()
                ax1.bar(range(len(original_counts)), original_counts.values, color='#1e88e5')
                ax1.set_title('Original Label Distribution')
                ax1.set_ylabel('Count')
                ax1.set_xticks(range(len(original_counts)))
                ax1.set_xticklabels(original_counts.index, rotation=45, ha='right')
                
                # Binary labels
                colors = ['#43a047', '#ff7043', '#9e9e9e']
                ax2.bar(binary_counts.index, binary_counts.values, color=colors[:len(binary_counts)])
                ax2.set_title('Binary Label Distribution')
                ax2.set_ylabel('Count')
                
                plt.tight_layout()
                st.pyplot(fig)
            
            if st.button("🚀 Run Model Analysis", key="analyze_btn", use_container_width=True, type="primary"):
                with st.spinner(f"🔄 Training 4 models with {N_SPLITS}-Fold CV{' & SMOTE' if use_smote else ''}..."):
                    df_results, trained_models, trained_vectorizer = evaluate_models(
                        st.session_state['scraped_df'], selected_phase, use_smote=use_smote
                    )
                    
                    if not df_results.empty:
                        st.session_state['df_results'] = df_results
                        st.session_state['trained_models'] = trained_models
                        st.session_state['trained_vectorizer'] = trained_vectorizer
                        st.session_state['selected_phase_run'] = selected_phase
                        
                        st.success("✅ Analysis complete! Models trained and saved to disk.")
                        
                        # Show immediate results
                        st.subheader("📈 Training Results")
                        st.dataframe(df_results, use_container_width=True)
                    else:
                        st.error("❌ Model training failed. Please check your data and try again.")
    
    # --- SMOTE ANALYSIS ---
    elif page == "SMOTE Analysis":
        st.markdown("<h1 style='color:#0d47a1;'>⚖️ SMOTE Analysis</h1>", unsafe_allow_html=True)
        
        if st.session_state['scraped_df'].empty:
            st.warning("⚠️ Please collect data first from the Data Collection page!")
            if st.button("Go to Data Collection", use_container_width=True):
                st.switch_page("Data Collection")
        else:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.write("""
            **SMOTE (Synthetic Minority Over-sampling Technique)** helps handle class imbalance by creating 
            synthetic samples for the minority class. This section analyzes the impact of SMOTE on model performance.
            """)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Select phase for analysis
            phases = ["Lexical & Morphological", "Syntactic", "Semantic", "Discourse", "Pragmatic"]
            selected_phase_smote = st.selectbox("🔧 Select Feature Extraction Method for Analysis:", phases, 
                                               key='selected_phase_smote')
            
            if st.button("📊 Run SMOTE Impact Analysis", key="smote_analyze_btn", use_container_width=True, type="primary"):
                analyze_smote_impact(st.session_state['scraped_df'], selected_phase_smote)
    
    # --- BENCHMARK TESTING ---
    elif page == "Benchmark Testing":
        st.markdown("<h1 style='color:#0d47a1;'>⚖️ Benchmark Testing</h1>", unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("Test trained models on external data and perform single-text fact-checking.")
        st.markdown('</div>', unsafe_allow_html=True)

        # read query param if present
        try:
            query_params = st.experimental_get_query_params()
            external_text = ""
            if 'text' in query_params and query_params['text']:
                external_text = query_params['text'][0]
                if len(external_text) > 2000:
                    external_text = external_text[:2000]
        except Exception:
            external_text = ""

        # Single-text prediction section
        st.subheader("🔍 Single-Text Fact Checking")
        
        custom_text = st.text_area("📝 Enter text to fact-check:", 
                                  value=external_text, 
                                  height=150,
                                  placeholder="Paste or type the claim you want to fact-check here...",
                                  key="custom_text_input",
                                  help="Enter any text claim to see model predictions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔎 Run Single-Text Check", key="single_check_btn", use_container_width=True, type="primary"):
                if not st.session_state.get('trained_models'):
                    st.error("❌ Please train models first in Model Training page before running single-text check.")
                elif not custom_text.strip():
                    st.warning("⚠️ Please enter some text to check.")
                else:
                    with st.spinner("🔍 Analyzing text with trained models..."):
                        trained_models = st.session_state['trained_models']
                        trained_vectorizer = st.session_state.get('trained_vectorizer')
                        selected_phase_run = st.session_state.get('selected_phase_run')
                        results = predict_single_text(custom_text, trained_models, trained_vectorizer, selected_phase_run)
                        
                        st.markdown("### 📊 Model Predictions")
                        
                        if 'error' in results:
                            st.error(f"❌ {results['error']}")
                        else:
                            # Display results in a nice format
                            pred_cols = st.columns(4)
                            model_names = list(results.keys())
                            
                            for idx, (mname, res) in enumerate(results.items()):
                                col_idx = idx % 4
                                with pred_cols[col_idx]:
                                    if 'prediction' in res:
                                        if res['prediction'] == 1:
                                            st.markdown(f'''
                                            <div style="background-color:#e8f5e9; padding:15px; border-radius:8px; border-left:5px solid #43a047;">
                                                <h4 style="margin:0; color:#2e7d32;">{mname}</h4>
                                                <p style="font-size:24px; font-weight:bold; color:#2e7d32; margin:10px 0;">✓ REAL</p>
                                                <p style="color:#666; font-size:12px; margin:0;">Prediction: {res['prediction']}</p>
                                            </div>
                                            ''', unsafe_allow_html=True)
                                        else:
                                            st.markdown(f'''
                                            <div style="background-color:#ffebee; padding:15px; border-radius:8px; border-left:5px solid #e53935;">
                                                <h4 style="margin:0; color:#c62828;">{mname}</h4>
                                                <p style="font-size:24px; font-weight:bold; color:#c62828; margin:10px 0;">✗ FAKE</p>
                                                <p style="color:#666; font-size:12px; margin:0;">Prediction: {res['prediction']}</p>
                                            </div>
                                            ''', unsafe_allow_html=True)
                                    else:
                                        st.markdown(f'''
                                        <div style="background-color:#fff3e0; padding:15px; border-radius:8px; border-left:5px solid #fb8c00;">
                                            <h4 style="margin:0; color:#ef6c00;">{mname}</h4>
                                            <p style="color:#ef6c00; margin:10px 0;">Error</p>
                                            <p style="color:#666; font-size:10px; margin:0;">{res.get('error', 'Unknown error')}</p>
                                        </div>
                                        ''', unsafe_allow_html=True)
        
        # Auto-run when external text present
        if external_text and st.session_state.get('trained_models') and custom_text == external_text:
            with st.spinner("🤖 Auto-running models on external text..."):
                trained_models = st.session_state['trained_models']
                trained_vectorizer = st.session_state.get('trained_vectorizer')
                selected_phase_run = st.session_state.get('selected_phase_run')
                auto_results = predict_single_text(external_text, trained_models, trained_vectorizer, selected_phase_run)
                
                st.markdown("### 🌐 Auto-run predictions for URL text")
                
                if 'error' in auto_results:
                    st.error(f"❌ {auto_results['error']}")
                else:
                    # Count predictions
                    real_count = sum(1 for res in auto_results.values() if res.get('prediction') == 1)
                    fake_count = sum(1 for res in auto_results.values() if res.get('prediction') == 0)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Real Predictions", real_count)
                    with col2:
                        st.metric("Fake Predictions", fake_count)
        
        st.markdown("---")
        st.subheader("📊 Benchmark with External Data")
        
        benchmark_col1, benchmark_col2 = st.columns(2)
        with benchmark_col1:
            use_demo = st.checkbox("Use Demo Data (Google Fact Check API)", value=True,
                                 help="Use pre-loaded demo data instead of making API calls")
        
        with benchmark_col2:
            num_claims = st.slider("Number of test claims:", min_value=5, max_value=50, value=15, step=5, 
                                 key='num_claims', help="Number of claims to use for benchmarking")
        
        if st.button("🚀 Run Benchmark Test", key="benchmark_btn", use_container_width=True, type="primary"):
            if not st.session_state.get('trained_models'):
                st.error("❌ Please train models first in the Model Training page!")
            else:
                with st.spinner('📥 Loading fact-check data...'):
                    if use_demo:
                        api_results = get_demo_google_claims()
                        st.success("✅ Demo Google Fact Check loaded successfully!")
                    else:
                        if 'GOOGLE_API_KEY' not in st.secrets:
                            st.error("🔑 API Key not found in secrets.toml")
                            api_results = []
                        else:
                            api_key = st.secrets["GOOGLE_API_KEY"]
                            api_results = fetch_google_claims(api_key, num_claims)
                            if api_results:
                                st.success(f"✅ Fetched {len(api_results)} claims from Google API!")
                    
                    google_df = process_and_map_google_claims(api_results)
                    
                    if not google_df.empty:
                        trained_models = st.session_state['trained_models']
                        trained_vectorizer = st.session_state['trained_vectorizer']
                        selected_phase_run = st.session_state['selected_phase_run']
                        benchmark_results_df = run_google_benchmark(google_df, trained_models, trained_vectorizer, selected_phase_run)
                        
                        st.session_state['google_benchmark_results'] = benchmark_results_df
                        st.session_state['google_df'] = google_df
                        
                        st.success(f"✅ Benchmark complete! Tested on {len(google_df)} claims.")
                        
                        # Show results immediately
                        st.subheader("📈 Benchmark Results")
                        st.dataframe(benchmark_results_df, use_container_width=True)
                    else:
                        st.warning("⚠️ No claims were processed. Try adjusting parameters.")

        # Display saved benchmark results
        if not st.session_state['google_benchmark_results'].empty:
            st.markdown("---")
            st.subheader("📋 Previous Benchmark Results")
            st.dataframe(st.session_state['google_benchmark_results'], use_container_width=True)

    # --- RESULTS & ANALYSIS ---
    elif page == "Results & Analysis":
        st.markdown("<h1 style='color:#0d47a1;'>📊 Results & Analysis</h1>", unsafe_allow_html=True)
        
        if st.session_state['df_results'].empty:
            st.warning("⚠️ No results available. Please train models first in the Model Training page!")
            if st.button("Go to Model Training", use_container_width=True):
                st.switch_page("Model Training")
        else:
            # Model performance summary
            st.header("📈 Model Performance Results")
            
            df_results = st.session_state['df_results']
            
            # Top metrics cards
            st.markdown('<div class="card">', unsafe_allow_html=True)
            results_col1, results_col2, results_col3, results_col4 = st.columns(4)
            
            # Find best model by F1-Score
            best_model_row = df_results.loc[df_results['F1-Score'].idxmax()]
            
            for i, (_, row) in enumerate(df_results.iterrows()):
                col = [results_col1, results_col2, results_col3, results_col4][i]
                with col:
                    is_best = row['Model'] == best_model_row['Model']
                    border_color = "#43a047" if is_best else "#1e88e5"
                    star = "⭐ " if is_best else ""
                    
                    st.markdown(f'''
                    <div style="background:white; padding:15px; border-radius:8px; border-left:5px solid {border_color}; box-shadow:0 2px 8px rgba(0,0,0,0.1);">
                        <h4 style="margin:0; color:{border_color};">{star}{row['Model']}</h4>
                        <p style="font-size:24px; font-weight:bold; color:#333; margin:10px 0;">{row['Accuracy']:.1f}%</p>
                        <div style="display:flex; justify-content:space-between; font-size:12px; color:#666;">
                            <span>F1: {row['F1-Score']:.3f}</span>
                            <span>{row['Inference Latency (ms)']}ms</span>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Visualization section
            st.markdown("---")
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                st.subheader("📊 Performance Metrics Comparison")
                chart_metric = st.selectbox("Select metric to visualize:", 
                                           ['Accuracy', 'F1-Score', 'Precision', 'Recall', 'Training Time (s)', 'Inference Latency (ms)'], 
                                           key='chart_metric')
                
                fig, ax = plt.subplots(figsize=(10, 6))
                models = df_results['Model']
                values = df_results[chart_metric]
                
                colors = ['#1e88e5' if m != best_model_row['Model'] else '#43a047' for m in models]
                
                bars = ax.bar(models, values, color=colors, edgecolor='white', linewidth=2)
                ax.set_ylabel(chart_metric)
                ax.set_title(f'{chart_metric} by Model')
                ax.grid(axis='y', alpha=0.3, linestyle='--')
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
                
                st.pyplot(fig)
            
            with viz_col2:
                st.subheader("⚡ Speed vs Accuracy Trade-off")
                fig, ax = plt.subplots(figsize=(10, 6))
                
                colors = ['#1e88e5', '#ff7043', '#43a047', '#8e24aa']
                for i, (_, row) in enumerate(df_results.iterrows()):
                    size = 200 if row['Model'] == best_model_row['Model'] else 150
                    alpha = 1.0 if row['Model'] == best_model_row['Model'] else 0.7
                    
                    ax.scatter(row['Inference Latency (ms)'], row['Accuracy'], 
                              s=size, alpha=alpha, color=colors[i], label=row['Model'], 
                              edgecolors='white', linewidth=2)
                    
                    # Add model name annotation
                    offset_x = 5 if row['Model'] != best_model_row['Model'] else 10
                    offset_y = 0.1 if row['Model'] != best_model_row['Model'] else 0.2
                    ax.annotate(row['Model'], 
                               (row['Inference Latency (ms)'] + offset_x, row['Accuracy'] + offset_y),
                               fontsize=10, fontweight='bold' if row['Model'] == best_model_row['Model'] else 'normal')
                
                ax.set_xlabel('Inference Latency (ms)')
                ax.set_ylabel('Accuracy (%)')
                ax.set_title('Model Performance: Speed vs Accuracy')
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.legend(loc='upper right')
                
                st.pyplot(fig)
            
            # Benchmark comparison if available
            if not st.session_state['google_benchmark_results'].empty:
                st.markdown("---")
                st.header("🔬 Fact Check Benchmark Results")
                
                google_results = st.session_state['google_benchmark_results']
                politifacts_results = st.session_state['df_results']
                
                st.subheader("📊 Performance Comparison: Training vs Benchmark")
                
                comp_cols = st.columns(4)
                for idx, (_, row) in enumerate(google_results.iterrows()):
                    model_name = row['Model']
                    google_accuracy = row['Accuracy']
                    
                    politifacts_row = politifacts_results[politifacts_results['Model'] == model_name]
                    if not politifacts_row.empty:
                        politifacts_accuracy = politifacts_row['Accuracy'].values[0]
                        delta = google_accuracy - politifacts_accuracy
                        
                        with comp_cols[idx]:
                            st.metric(
                                label=f"{model_name}",
                                value=f"{google_accuracy:.1f}%",
                                delta=f"{delta:+.1f}%",
                                delta_color="normal" if delta >= 0 else "inverse"
                            )
                
                # Benchmark visualization
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                
                # Training results
                models = politifacts_results['Model']
                train_acc = politifacts_results['Accuracy']
                benchmark_acc = [google_results[google_results['Model'] == m]['Accuracy'].values[0] 
                               if m in google_results['Model'].values else 0 for m in models]
                
                x = np.arange(len(models))
                width = 0.35
                
                ax1.bar(x - width/2, train_acc, width, label='Training Data', color='#1e88e5')
                ax1.bar(x + width/2, benchmark_acc, width, label='Benchmark Data', color='#ff7043')
                ax1.set_xlabel('Model')
                ax1.set_ylabel('Accuracy (%)')
                ax1.set_title('Accuracy Comparison: Training vs Benchmark')
                ax1.set_xticks(x)
                ax1.set_xticklabels(models, rotation=45)
                ax1.legend()
                ax1.grid(axis='y', alpha=0.3)
                
                # F1-Score comparison
                train_f1 = politifacts_results['F1-Score']
                benchmark_f1 = [google_results[google_results['Model'] == m]['F1-Score'].values[0] 
                              if m in google_results['Model'].values else 0 for m in models]
                
                ax2.bar(x - width/2, train_f1, width, label='Training Data', color='#1e88e5')
                ax2.bar(x + width/2, benchmark_f1, width, label='Benchmark Data', color='#ff7043')
                ax2.set_xlabel('Model')
                ax2.set_ylabel('F1-Score')
                ax2.set_title('F1-Score Comparison: Training vs Benchmark')
                ax2.set_xticks(x)
                ax2.set_xticklabels(models, rotation=45)
                ax2.legend()
                ax2.grid(axis='y', alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
            
            # Humorous critique section
            st.markdown("---")
            st.header("🎭 AI Performance Review")
            
            critique_col1, critique_col2 = st.columns([2, 1])
            
            with critique_col1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                critique_text = generate_humorous_critique(st.session_state['df_results'], 
                                                         st.session_state['selected_phase_run'])
                st.markdown(critique_text)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with critique_col2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("🏆 Winner's Circle")
                
                if not st.session_state['df_results'].empty:
                    best_model = st.session_state['df_results'].loc[st.session_state['df_results']['F1-Score'].idxmax()]
                    
                    st.markdown(f"""
                    <div style="text-align:center; padding:20px;">
                        <h3 style="color:#ff9800; margin-bottom:20px;">🏆 Champion Model</h3>
                        <div style="background:linear-gradient(135deg,#ff9800,#ff5722); color:white; padding:20px; border-radius:12px; margin-bottom:20px;">
                            <h2 style="margin:0;">{best_model['Model']}</h2>
                        </div>
                        
                        <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; margin:20px 0;">
                            <div style="background:#e3f2fd; padding:10px; border-radius:8px;">
                                <p style="margin:0; font-size:12px; color:#1565c0;">Accuracy</p>
                                <p style="margin:0; font-size:20px; font-weight:bold; color:#0d47a1;">{best_model['Accuracy']:.1f}%</p>
                            </div>
                            <div style="background:#f3e5f5; padding:10px; border-radius:8px;">
                                <p style="margin:0; font-size:12px; color:#7b1fa2;">F1-Score</p>
                                <p style="margin:0; font-size:20px; font-weight:bold; color:#4a148c;">{best_model['F1-Score']:.3f}</p>
                            </div>
                        </div>
                        
                        <div style="background:#e8f5e9; padding:15px; border-radius:8px; margin-top:20px;">
                            <p style="margin:0; font-size:14px; color:#2e7d32;">
                                <strong>Feature Set:</strong><br>
                                {st.session_state['selected_phase_run']}
                            </p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)

if __name__ == '__main__':
    app()

# lssdp.py
# Updated Streamlit app with single-text URL handling, save/load trained models, and single-text prediction.
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
# Model training & evaluation (K-Fold & SMOTE)
# --------------------------
def evaluate_models(df: pd.DataFrame, selected_phase: str):
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
        st.caption(f"Training {name} with {N_SPLITS}-Fold CV & SMOTE...")
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
                    smote_pipeline = ImbPipeline([('sampler', SMOTE(random_state=42, k_neighbors=3)), ('classifier', model)])
                    smote_pipeline.fit(X_train, y_train)
                    clf = smote_pipeline
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
                smote_pipeline_final = ImbPipeline([('sampler', SMOTE(random_state=42, k_neighbors=3)), ('classifier', final_model)])
                smote_pipeline_final.fit(X_final, y)
                trained_models_final[name] = smote_pipeline_final
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
# Humor & critique functions (unchanged)
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

    # CSS (as before) - kept minimal to reduce length
    st.markdown("""
    <style>
    body { background:#0f171e; color:#e6e6e6; }
    .card { background:#1a242f; padding:12px; border-radius:8px; }
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

    # Attempt to load previously saved models on startup (only once)
    if 'models_loaded_attempted' not in st.session_state:
        expected_names = ["Naive_Bayes", "Decision_Tree", "Logistic_Regression", "SVM"]
        # Try load both with spaces and underscores
        trained_models_loaded, vec_loaded, selected_phase_loaded = load_trained_models(expected_model_names=["Naive Bayes","Decision Tree","Logistic Regression","SVM"])
        # If any model found (not None), populate session state
        any_model_loaded = any(v is not None for v in trained_models_loaded.values()) if trained_models_loaded else False
        if any_model_loaded:
            # normalize model keys to display names
            st.session_state['trained_models'] = trained_models_loaded
            st.session_state['trained_vectorizer'] = vec_loaded
            if selected_phase_loaded:
                st.session_state['selected_phase_run'] = selected_phase_loaded
            st.success("Loaded saved models from disk. Single-text checks are available.")
        st.session_state['models_loaded_attempted'] = True

    # Sidebar and navigation
    st.sidebar.title("FactChecker")
    page = st.sidebar.radio("Navigation", ["Dashboard", "Data Collection", "Model Training", "Benchmark Testing", "Results & Analysis"], key='navigation')

    # --- DASHBOARD ---
    if page == "Dashboard":
        st.markdown("<h1 style='color:#00c8ff;'>FactChecker Dashboard</h1>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="card"><h3>Data Overview</h3><p>Collect and manage training data from Politifact archives</p></div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="card"><h3>Model Training</h3><p>Advanced NLP feature extraction and ML training</p></div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="card"><h3>Benchmark Testing</h3><p>Validate models with real-world data</p></div>', unsafe_allow_html=True)

    # --- DATA COLLECTION ---
    elif page == "Data Collection":
        st.markdown("<h1 style='color:#00c8ff;'>Data Collection</h1>", unsafe_allow_html=True)
        min_date = pd.to_datetime('2007-01-01')
        max_date = pd.to_datetime('today').normalize()
        date_col1, date_col2 = st.columns(2)
        with date_col1:
            start_date = st.date_input("Start Date", min_value=min_date, max_value=max_date, value=pd.to_datetime('2023-01-01'))
        with date_col2:
            end_date = st.date_input("End Date", min_value=min_date, max_value=max_date, value=max_date)
        if st.button("Scrape Politifact Data", key="scrape_btn", use_container_width=True):
            if start_date > end_date:
                st.error("Start date must be before end date")
            else:
                with st.spinner("Scraping political claims..."):
                    scraped_df = scrape_data_by_date_range(pd.to_datetime(start_date), pd.to_datetime(end_date))
                if not scraped_df.empty:
                    st.session_state['scraped_df'] = scraped_df
                    st.success(f"Successfully scraped {len(scraped_df)} claims!")
                else:
                    st.warning("No data found. Try adjusting date range.")
        if not st.session_state['scraped_df'].empty:
            st.dataframe(st.session_state['scraped_df'].head(10), use_container_width=True)

    # --- MODEL TRAINING ---
    elif page == "Model Training":
        st.markdown("<h1 style='color:#00c8ff;'>Model Training</h1>", unsafe_allow_html=True)
        if st.session_state['scraped_df'].empty:
            st.warning("Please collect data first from the Data Collection page!")
        else:
            phases = ["Lexical & Morphological", "Syntactic", "Semantic", "Discourse", "Pragmatic"]
            selected_phase = st.selectbox("Feature Extraction Method:", phases, key='selected_phase')
            st.caption(f"*Selected: {selected_phase}*")
            if st.button("Run Model Analysis", key="analyze_btn", use_container_width=True):
                with st.spinner(f"Training 4 models with {N_SPLITS}-Fold CV..."):
                    df_results, trained_models, trained_vectorizer = evaluate_models(st.session_state['scraped_df'], selected_phase)
                    st.session_state['df_results'] = df_results
                    st.session_state['trained_models'] = trained_models
                    st.session_state['trained_vectorizer'] = trained_vectorizer
                    st.session_state['selected_phase_run'] = selected_phase
                    st.success("Analysis complete! Models trained and saved to disk.")

    # --- BENCHMARK TESTING ---
    elif page == "Benchmark Testing":
        st.markdown("<h1 style='color:#00c8ff;'>Benchmark Testing</h1>", unsafe_allow_html=True)

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

        st.subheader("Fact Check Benchmark")
        mode_col1, mode_col2 = st.columns(2)
        with mode_col1:
            use_demo = st.checkbox("Google Fact Check API (Demo mode)", value=True)
        with mode_col2:
            pass

        num_claims = st.slider("Number of test claims:", min_value=5, max_value=50, value=10, step=5, key='num_claims')

        st.markdown("---")
        st.subheader("Check custom text")
        custom_text = st.text_input("Text to check (prefilled from extension if provided):", value=external_text, key="custom_text_input")
        if st.button("Run Single-Text Check", key="single_check_btn", use_container_width=True):
            if not st.session_state.get('trained_models'):
                st.error("Please train models first in Model Training page before running single-text check.")
            else:
                with st.spinner("Running models on the provided text..."):
                    trained_models = st.session_state['trained_models']
                    trained_vectorizer = st.session_state.get('trained_vectorizer')
                    selected_phase_run = st.session_state.get('selected_phase_run')
                    results = predict_single_text(custom_text, trained_models, trained_vectorizer, selected_phase_run)
                    st.markdown("### Single-text Model Predictions")
                    if 'error' in results:
                        st.error(results['error'])
                    else:
                        for mname, res in results.items():
                            if 'prediction' in res:
                                pretty = "REAL/TRUE (1)" if res['prediction'] == 1 else "FAKE/ FALSE (0)"
                                st.write(f"**{mname}** → {pretty}")
                            else:
                                st.write(f"**{mname}** → Error: {res.get('error')}")

        # auto-run when external text present and models loaded
        if external_text and st.session_state.get('trained_models') and custom_text == external_text:
            with st.spinner("Auto-running models on external text..."):
                trained_models = st.session_state['trained_models']
                trained_vectorizer = st.session_state.get('trained_vectorizer')
                selected_phase_run = st.session_state.get('selected_phase_run')
                auto_results = predict_single_text(external_text, trained_models, trained_vectorizer, selected_phase_run)
                st.markdown("### Auto-run predictions for URL text")
                if 'error' in auto_results:
                    st.error(auto_results['error'])
                else:
                    for mname, res in auto_results.items():
                        if 'prediction' in res:
                            pretty = "REAL/TRUE (1)" if res['prediction'] == 1 else "FAKE/ FALSE (0)"
                            st.write(f"**{mname}** → {pretty}")
                        else:
                            st.write(f"**{mname}** → Error: {res.get('error')}")

        st.markdown("---")
        # Benchmark using demo or Google API
        if st.button("Run Benchmark Test", key="benchmark_btn", use_container_width=True):
            if not st.session_state.get('trained_models'):
                st.error("Please train models first in the Model Training page!")
            else:
                with st.spinner('Loading fact-check data...'):
                    if use_demo:
                        api_results = get_demo_google_claims()
                        st.success("✅ Demo Google Fact Check loaded successfully!")
                    else:
                        if 'GOOGLE_API_KEY' not in st.secrets:
                            st.error("API Key not found in secrets.toml")
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
                    else:
                        st.warning("No claims were processed. Try adjusting parameters.")

        if not st.session_state['google_benchmark_results'].empty:
            st.subheader("Benchmark Results")
            st.dataframe(st.session_state['google_benchmark_results'], use_container_width=True)

    # --- RESULTS & ANALYSIS ---
    elif page == "Results & Analysis":
        st.markdown("<h1 style='color:#00c8ff;'>Results & Analysis</h1>", unsafe_allow_html=True)
        if st.session_state['df_results'].empty:
            st.warning("No results available. Please train models first in the Model Training page!")
        else:
            st.header("Model Performance Results")
            df_results = st.session_state['df_results']
            results_col1, results_col2, results_col3, results_col4 = st.columns(4)
            metrics_data = []
            for _, row in df_results.iterrows():
                metrics_data.append({'model': row['Model'], 'accuracy': row['Accuracy'], 'f1': row['F1-Score'], 'training_time': row['Training Time (s)']})
            for i, metric in enumerate(metrics_data):
                col = [results_col1, results_col2, results_col3, results_col4][i]
                with col:
                    st.markdown(f"<div class='card'><h4>{metric['model']}</h4><p>{metric['accuracy']:.1f}% | F1 {metric['f1']:.3f} | {metric['training_time']}s</p></div>", unsafe_allow_html=True)
            st.markdown("---")
            viz_col1, viz_col2 = st.columns(2)
            with viz_col1:
                st.subheader("Performance Metrics")
                chart_metric = st.selectbox("Select metric to visualize:", ['Accuracy', 'F1-Score', 'Precision', 'Recall', 'Training Time (s)', 'Inference Latency (ms)'], key='chart_metric')
                chart_data = df_results[['Model', chart_metric]].set_index('Model')
                st.bar_chart(chart_data)
            with viz_col2:
                st.subheader("Speed vs Accuracy Trade-off")
                fig, ax = plt.subplots(figsize=(8, 6))
                colors = ['#00a8e1', '#00c8ff', '#1a8cd8', '#2d9cdb']
                for i, (_, row) in enumerate(df_results.iterrows()):
                    ax.scatter(row['Inference Latency (ms)'], row['Accuracy'], s=200, alpha=0.7, color=colors[i], label=row['Model'])
                    ax.annotate(row['Model'], (row['Inference Latency (ms)'] + 5, row['Accuracy']), fontsize=9, alpha=0.8)
                ax.set_xlabel('Inference Latency (ms)')
                ax.set_ylabel('Accuracy (%)')
                ax.set_title('Model Performance: Speed vs Accuracy')
                ax.grid(True, alpha=0.3)
                ax.legend()
                st.pyplot(fig)
            if not st.session_state['google_benchmark_results'].empty:
                st.markdown("---")
                st.header("Fact Check Benchmark Results")
                google_results = st.session_state['google_benchmark_results']
                politifacts_results = st.session_state['df_results']
                st.subheader("Performance Comparison")
                comp_col1, comp_col2, comp_col3, comp_col4 = st.columns(4)
                for idx, (_, row) in enumerate(google_results.iterrows()):
                    model_name = row['Model']
                    google_accuracy = row['Accuracy']
                    politifacts_row = politifacts_results[politifacts_results['Model'] == model_name]
                    if not politifacts_row.empty:
                        politifacts_accuracy = politifacts_row['Accuracy'].values[0]
                        delta = google_accuracy - politifacts_accuracy
                        delta_color = "normal" if delta >= 0 else "inverse"
                    else:
                        delta = None
                        delta_color = "off"
                    col = [comp_col1, comp_col2, comp_col3, comp_col4][idx]
                    with col:
                        if delta is not None:
                            st.metric(label=f"{model_name}", value=f"{google_accuracy:.1f}%", delta=f"{delta:+.1f}%", delta_color=delta_color)
                        else:
                            st.metric(label=f"{model_name}", value=f"{google_accuracy:.1f}%")
            st.markdown("---")
            st.header("AI Performance Review")
            critique_col1, critique_col2 = st.columns([2, 1])
            with critique_col1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                critique_text = generate_humorous_critique(st.session_state['df_results'], st.session_state['selected_phase_run'])
                st.markdown(critique_text)
                st.markdown('</div>', unsafe_allow_html=True)
            with critique_col2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("Winner's Circle")
                if not st.session_state['df_results'].empty:
                    best_model = st.session_state['df_results'].loc[st.session_state['df_results']['F1-Score'].idxmax()]
                    st.markdown(f"""
                    **Champion Model:**  
                    **{best_model['Model']}**
                    
                    **Performance:**  
                    {best_model['Accuracy']:.1f}% Accuracy  
                    {best_model['F1-Score']:.3f} F1-Score  
                    {best_model['Inference Latency (ms)']}ms Inference
                    
                    **Feature Set:**  
                    {st.session_state['selected_phase_run']}
                    """)
                st.markdown('</div>', unsafe_allow_html=True)

if __name__ == '__main__':
    app()

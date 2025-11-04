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

# --- NEW DEP: Imbalanced-learn ---
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline 

# --- NLP & ML Imports ---
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

# --- Configuration ---
SCRAPED_DATA_PATH = 'politifact_data.csv'
N_SPLITS = 5

# Google Fact Check API rating mappings (for binary classification)
GOOGLE_TRUE_RATINGS = ["True", "Mostly True", "Accurate", "Correct"]
GOOGLE_FALSE_RATINGS = ["False", "Mostly False", "Pants on Fire", "Pants on Fire!", "Fake", "Incorrect", "Baseless", "Misleading"] 

# --- SpaCy Loading Function (Robust for Streamlit Cloud) ---
@st.cache_resource
def load_spacy_model():
    """Attempts to load SpaCy model, relying on the model being in requirements.txt."""
    try:
        nlp = spacy.load("en_core_web_sm")
        return nlp
    except OSError as e:
        st.error(f"SpaCy model 'en_core_web_sm' not found. Please ensure the direct GitHub link for the model is correctly listed in your 'requirements.txt' file.")
        st.code("""
        # Example of the line needed in requirements.txt:
        https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
        imbalanced-learn # Required for SMOTE
        """, language='text')
        raise e

# Load resources outside main app flow
try:
    NLP_MODEL = load_spacy_model()
except Exception:
    st.stop() 

stop_words = STOP_WORDS
pragmatic_words = ["must", "should", "might", "could", "will", "?", "!"]

# ============================
# GOOGLE FACT CHECK API INTEGRATION
# ============================

def fetch_google_claims(api_key, num_claims=100):
    """
    Fetches claims from Google Fact Check API with pagination handling.

    TODO: User must add GOOGLE_API_KEY to .streamlit/secrets.toml
    Get your API key from: https://console.cloud.google.com/apis/credentials
    Enable the "Fact Check Tools API" in Google Cloud Console first.
    """
    base_url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    collected_claims = []
    page_token = None
    placeholder = st.empty()

    try:
        while len(collected_claims) < num_claims:
            # Build request parameters
            params = {
                'key': api_key,
                'languageCode': 'en',
                'pageSize': min(100, num_claims - len(collected_claims))
            }

            if page_token:
                params['pageToken'] = page_token

            # Update progress
            placeholder.text(f"Fetching Google claims... {len(collected_claims)} collected so far")

            # Make API request
            response = requests.get(base_url, params=params, timeout=15)

            # Check for HTTP errors
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

            # Check if response has claims
            if 'claims' not in data or not data['claims']:
                placeholder.success(f"Fetched {len(collected_claims)} claims (no more available)")
                break

            # Process each claim
            for claim_obj in data['claims']:
                if len(collected_claims) >= num_claims:
                    break

                # Extract claim text
                claim_text = claim_obj.get('text', '')

                # Extract rating from first claimReview
                claim_reviews = claim_obj.get('claimReview', [])
                if not claim_reviews or len(claim_reviews) == 0:
                    continue  # Skip claims without reviews

                textual_rating = claim_reviews[0].get('textualRating', '')

                # Skip if missing required fields
                if not claim_text or not textual_rating:
                    continue

                collected_claims.append({
                    'claim_text': claim_text,
                    'rating': textual_rating
                })

            # Check for next page
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
    """
    Converts Google's granular ratings into binary format (1=True, 0=False) and creates DataFrame.
    Discards ambiguous ratings like 'Half True', 'Mixed', etc.
    """
    if not api_results:
        return pd.DataFrame(columns=['claim_text', 'ground_truth'])

    processed_claims = []
    true_count = 0
    false_count = 0
    discarded_count = 0

    for claim_data in api_results:
        claim_text = claim_data.get('claim_text', '').strip()
        rating = claim_data.get('rating', '').strip()

        # Data quality checks
        if not claim_text or len(claim_text) < 10:
            discarded_count += 1
            continue

        if not rating:
            discarded_count += 1
            continue

        # Normalize rating for comparison (remove punctuation, lowercase)
        rating_normalized = rating.lower().strip().rstrip('!').rstrip('?')

        # Map to binary
        is_true = any(rating_normalized == r.lower() for r in GOOGLE_TRUE_RATINGS)
        is_false = any(rating_normalized == r.lower() for r in GOOGLE_FALSE_RATINGS)

        if is_true:
            processed_claims.append({
                'claim_text': claim_text,
                'ground_truth': 1
            })
            true_count += 1
        elif is_false:
            processed_claims.append({
                'claim_text': claim_text,
                'ground_truth': 0
            })
            false_count += 1
        else:
            # Ambiguous rating - discard
            discarded_count += 1

    # Create DataFrame
    google_df = pd.DataFrame(processed_claims)

    if not google_df.empty:
        # Remove duplicates (keep first occurrence)
        google_df = google_df.drop_duplicates(subset=['claim_text'], keep='first')

    # Display statistics
    total_processed = len(api_results)
    st.info(f"Processed {total_processed} claims: {true_count} True, {false_count} False, {discarded_count} ambiguous (discarded)")

    # Warn if only one class
    if not google_df.empty and len(google_df['ground_truth'].unique()) < 2:
        st.warning("Only one class found in processed claims. Results may not be meaningful.")

    return google_df


def run_google_benchmark(google_df, trained_models, vectorizer, selected_phase):
    """
    Tests trained models on Google claims and calculates performance metrics.
    """
    if google_df.empty:
        st.error("No Google claims available for benchmarking.")
        return pd.DataFrame()

    # Extract claim texts and ground truth labels
    X_raw = google_df['claim_text']
    y_true = google_df['ground_truth'].values

    # Apply same feature extraction as training
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
            # Dense features - no vectorizer needed
            X_features = pd.DataFrame(X_raw.apply(semantic_features).tolist(), columns=["polarity", "subjectivity"]).values

        elif selected_phase == "Pragmatic":
            # Dense features - no vectorizer needed
            X_features = pd.DataFrame(X_raw.apply(pragmatic_features).tolist(), columns=pragmatic_words).values

        else:
            st.error(f"Unknown feature phase: {selected_phase}")
            return pd.DataFrame()

    except Exception as e:
        st.error(f"Feature extraction failed for Google claims: {e}")
        return pd.DataFrame()

    # Test each trained model
    results_list = []

    for model_name, model in trained_models.items():
        try:
            # Handle Naive Bayes with negative values (same as training)
            if model_name == "Naive Bayes":
                X_features_model = np.abs(X_features).astype(float)
            else:
                X_features_model = X_features

            # Measure inference time
            start_inference = time.time()
            y_pred = model.predict(X_features_model)
            inference_time = (time.time() - start_inference) * 1000  # Convert to ms

            # Calculate metrics
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

# ============================
# 1. WEB SCRAPING FUNCTION (Remains identical to previous successful version)
# ============================

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

# ============================
# 2. FEATURE EXTRACTION (SPA/TEXTBLOB)
# ============================

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

# ============================
# 3. MODEL TRAINING AND EVALUATION (K-FOLD & SMOTE)
# ============================

def get_classifier(name):
    """Initializes a classifier instance with hyperparameter tuning for imbalance."""
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
    """Applies the chosen feature extraction technique and optimization (e.g., N-Grams)."""
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


def evaluate_models(df: pd.DataFrame, selected_phase: str):
    """Trains and evaluates models using Stratified K-Fold Cross-Validation and SMOTE."""
    
    # 1. FEATURE ENGINEERING: BINARY TARGET MAPPING
    
    # Define mapping groups
    REAL_LABELS = ["True", "No Flip", "Mostly True", "Half Flip", "Half True"]
    FAKE_LABELS = ["False", "Barely True", "Pants On Fire", "Full Flop"]
    
    # Create the new binary target column
    def create_binary_target(label):
        if label in REAL_LABELS:
            return 1 # Real/True
        elif label in FAKE_LABELS:
            return 0 # Fake/False
        else:
            return np.nan # Mark unmappable/error labels

    df['target_label'] = df['label'].apply(create_binary_target)
    
    # 2. DATA CLEANING AND FILTERING
    
    # Drop rows where mapping failed
    df = df.dropna(subset=['target_label'])
    
    # Remove rows with short statements (noise/lack of context)
    df = df[df['statement'].astype(str).str.len() > 10]
    
    X_raw = df['statement'].astype(str)
    y_raw = df['target_label'].astype(int) # Target is now explicitly 0 or 1
    
    if len(np.unique(y_raw)) < 2:
        st.error("After binary mapping, only one class remains (all Real or all Fake). Cannot train classifier.")
        return pd.DataFrame() 
    
    # 3. Feature Extraction (Apply to all data once per phase)
    X_features_full, vectorizer = apply_feature_extraction(X_raw, selected_phase)
    
    if X_features_full is None:
        st.error("Feature extraction failed.")
        return pd.DataFrame()
        
    # Prepare data for K-Fold
    if isinstance(X_features_full, pd.DataFrame):
        X_features_full = X_features_full.values
    
    y = y_raw.values
    
    # 4. K-Fold Setup
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
        st.caption(f"🚀 Training {name} with {N_SPLITS}-Fold CV & SMOTE...")
        
        fold_metrics = {
            'accuracy': [], 'f1': [], 'precision': [], 'recall': [], 'train_time': [], 'inference_time': []
        }
        
        for fold, (train_index, test_index) in enumerate(skf.split(X_features_full, y)):
            
            # 4a. Get data indices for this fold
            X_train_raw = pd.Series([X_raw_list[i] for i in train_index])
            X_test_raw = pd.Series([X_raw_list[i] for i in test_index])
            y_train = y[train_index]
            y_test = y[test_index]
            
            # 4b. Transform the features using the fitted vectorizer (if applicable)
            if vectorizer is not None:
                # Need to run the phase's preprocessing (lexical_features or syntactic_features) on the raw text first
                X_train = vectorizer.transform(X_train_raw.apply(lexical_features if 'Lexical' in selected_phase else syntactic_features))
                X_test = vectorizer.transform(X_test_raw.apply(lexical_features if 'Lexical' in selected_phase else syntactic_features))
            else:
                # Dense feature sets (Semantic/Pragmatic)
                X_train, _ = apply_feature_extraction(X_train_raw, selected_phase)
                X_test, _ = apply_feature_extraction(X_test_raw, selected_phase)
            
            
            start_time = time.time()
            try:
                # --- SMOTE PIPELINE & Naive Bayes Fix ---
                if name == "Naive Bayes":
                    # FIX: Use np.abs on sparse matrix to get positive counts, then convert to int/float as needed.
                    X_train_final = np.abs(X_train).astype(float) 
                    clf = model
                    model.fit(X_train_final, y_train)
                else:
                    # Apply SMOTE to training data for other models
                    smote_pipeline = ImbPipeline([
                        ('sampler', SMOTE(random_state=42, k_neighbors=3)),
                        ('classifier', model)
                    ])
                    smote_pipeline.fit(X_train, y_train)
                    clf = smote_pipeline
                
                train_time = time.time() - start_time
                
                start_inference = time.time()
                y_pred = clf.predict(X_test)
                inference_time = (time.time() - start_inference) * 1000 
                
                # Metrics
                fold_metrics['accuracy'].append(accuracy_score(y_test, y_pred))
                fold_metrics['f1'].append(f1_score(y_test, y_pred, average='weighted', zero_division=0))
                fold_metrics['precision'].append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
                fold_metrics['recall'].append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
                fold_metrics['train_time'].append(train_time)
                fold_metrics['inference_time'].append(inference_time)

            except Exception as e:
                st.warning(f"Fold {fold+1} failed for {name}: {e}")
                for key in fold_metrics: fold_metrics[key].append(0)
                continue

        # Calculate means across all folds
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
             st.error(f"⚠️ {name} failed across all folds.")
             model_metrics[name] = {
                "Model": name, "Accuracy": 0, "F1-Score": 0, "Precision": 0, "Recall": 0,
                "Training Time (s)": 0, "Inference Latency (ms)": 9999,
            }

    # 5. TRAIN FINAL MODELS ON FULL DATASET (for Google benchmark)
    st.caption("Training final models on complete dataset for benchmarking...")
    trained_models_final = {}

    for name in models_to_run.keys():
        try:
            # Get fresh model instance
            final_model = get_classifier(name)

            # Prepare features for final training
            if vectorizer is not None:
                # Transform using the fitted vectorizer
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
                # Dense features (Semantic/Pragmatic)
                X_final = X_features_full

            # Apply SMOTE and train (same pattern as K-Fold)
            if name == "Naive Bayes":
                X_final_train = np.abs(X_final).astype(float)
                final_model.fit(X_final_train, y)
                trained_models_final[name] = final_model
            else:
                # Apply SMOTE to full dataset for other models
                smote_pipeline_final = ImbPipeline([
                    ('sampler', SMOTE(random_state=42, k_neighbors=3)),
                    ('classifier', final_model)
                ])
                smote_pipeline_final.fit(X_final, y)
                trained_models_final[name] = smote_pipeline_final

        except Exception as e:
            st.warning(f"Failed to train final {name} model: {e}")
            trained_models_final[name] = None

    results_list = list(model_metrics.values())
    return pd.DataFrame(results_list), trained_models_final, vectorizer

# ============================
# 4. HUMOR & CRITIQUE FUNCTIONS (REMAINS UNCHANGED)
# ============================

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
    
    headline = f"👑 The Golden Snitch Award goes to the {best_model}!"
    
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

# ============================
# 5. STREAMLIT APP FUNCTION
# ============================

def app():
    # --- Modern Theme Configuration ---
    st.set_page_config(
        page_title='TruthGuard: AI Fact-Checking Platform',
        layout='wide',
        initial_sidebar_state='expanded'
    )

    # Custom CSS for modern theme
    st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --primary: #2563eb;
        --secondary: #7c3aed;
        --accent: #06d6a0;
        --background: #0f172a;
        --surface: #1e293b;
        --text: #f8fafc;
        --text-secondary: #cbd5e1;
    }
    
    .main-header {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    
    .main-header h1 {
        color: white;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    
    .main-header h3 {
        color: rgba(255,255,255,0.9);
        font-size: 1.3rem;
        text-align: center;
        font-weight: 300;
    }
    
    .card {
        background: var(--surface);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid var(--accent);
        box-shadow: 0 4px 16px rgba(0,0,0,0.2);
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, var(--surface) 0%, #2d3748 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #374151;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        color: white;
        border: none;
        padding: 0.7rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.4);
    }
    
    .feature-pill {
        background: var(--primary);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 0.2rem;
        display: inline-block;
    }
    
    .success-box {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #d97706 0%, #f59e0b 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .info-box {
        background: linear-gradient(135deg, #0369a1 0%, #0ea5e9 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # --- Modern Header ---
    st.markdown("""
    <div class="main-header">
        <h1>🔍 TruthGuard</h1>
        <h3>AI-Powered Fact-Checking & Misanalysis Detection</h3>
    </div>
    """, unsafe_allow_html=True)

    # --- State Management ---
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

    # ============================
    # MAIN LAYOUT - MODERN CARDS
    # ============================
    
    # Row 1: Data Collection Cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 Data Collection")
        st.markdown("**Politifact Archive Scraper**")
        
        min_date = pd.to_datetime('2007-01-01')
        max_date = pd.to_datetime('today').normalize()

        date_col1, date_col2 = st.columns(2)
        with date_col1:
            start_date = st.date_input("Start Date", min_value=min_date, max_value=max_date, value=pd.to_datetime('2023-01-01'))
        with date_col2:
            end_date = st.date_input("End Date", min_value=min_date, max_value=max_date, value=max_date)

        if st.button("🚀 Scrape Politifact Data", key="scrape_btn"):
            if start_date > end_date:
                st.error("❌ Start date must be before end date")
            else:
                with st.spinner("🕸️ Scraping political claims..."):
                    scraped_df = scrape_data_by_date_range(pd.to_datetime(start_date), pd.to_datetime(end_date))
                
                if not scraped_df.empty:
                    st.session_state['scraped_df'] = scraped_df
                    st.markdown(f'<div class="success-box">✅ Successfully scraped {len(scraped_df)} claims!</div>', unsafe_allow_html=True)
                else:
                    st.warning("⚠️ No data found. Try adjusting date range.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("⚙️ Analysis Setup")
        st.markdown("**NLP Feature Configuration**")
        
        phases = [
            "Lexical & Morphological",
            "Syntactic", 
            "Semantic",
            "Discourse",
            "Pragmatic"
        ]
        selected_phase = st.selectbox("Feature Extraction Method:", phases, key='selected_phase')
        
        # Feature descriptions
        feature_descriptions = {
            "Lexical & Morphological": "Word-level analysis: lemmatization, stopword removal, n-grams",
            "Syntactic": "Grammar structure: part-of-speech tags, sentence patterns", 
            "Semantic": "Meaning analysis: sentiment polarity, subjectivity scoring",
            "Discourse": "Text structure: sentence count, discourse markers",
            "Pragmatic": "Intent analysis: modal verbs, question marks, emphasis markers"
        }
        
        st.caption(f"*{feature_descriptions[selected_phase]}*")
        
        if st.button("🔬 Run Model Analysis", key="analyze_btn"):
            if st.session_state['scraped_df'].empty:
                st.error("❌ Please scrape data first!")
            else:
                with st.spinner(f"🤖 Training 4 models with {N_SPLITS}-Fold CV..."):
                    df_results, trained_models, trained_vectorizer = evaluate_models(st.session_state['scraped_df'], selected_phase)
                    st.session_state['df_results'] = df_results
                    st.session_state['trained_models'] = trained_models
                    st.session_state['trained_vectorizer'] = trained_vectorizer
                    st.session_state['selected_phase_run'] = selected_phase
                    st.markdown('<div class="success-box">✅ Analysis complete! Results ready below.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Row 2: Real-time Benchmarking
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🌐 Real-Time Benchmark")
    st.markdown("**Google Fact Check API Integration**")
    
    bench_col1, bench_col2, bench_col3 = st.columns([2,2,1])
    
    with bench_col1:
        num_google_claims = st.slider(
            "Number of test claims:",
            min_value=10,
            max_value=500,
            value=100,
            step=10,
            key='num_google_claims'
        )
    
    with bench_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📊 Run Google Benchmark", key="benchmark_btn"):
            # Pre-flight checks
            if not st.session_state.get('trained_models'):
                st.error("❌ Please train models first!")
            elif 'GOOGLE_API_KEY' not in st.secrets:
                st.error("🔑 Google API key not found in secrets.toml")
            else:
                with st.spinner('🌍 Fetching live fact-check data...'):
                    api_key = st.secrets["GOOGLE_API_KEY"]
                    api_results = fetch_google_claims(api_key, num_google_claims)
                    google_df = process_and_map_google_claims(api_results)

                    if not google_df.empty:
                        trained_models = st.session_state['trained_models']
                        trained_vectorizer = st.session_state['trained_vectorizer']
                        selected_phase_run = st.session_state['selected_phase_run']
                        benchmark_results_df = run_google_benchmark(google_df, trained_models, trained_vectorizer, selected_phase_run)
                        st.session_state['google_benchmark_results'] = benchmark_results_df
                        st.session_state['google_df'] = google_df
                        st.markdown(f'<div class="success-box">✅ Benchmark complete! Tested on {len(google_df)} Google claims.</div>', unsafe_allow_html=True)
    
    with bench_col3:
        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("Tests trained models against live fact-check data")
    st.markdown('</div>', unsafe_allow_html=True)

    # ============================
    # RESULTS SECTION
    # ============================
    
    if not st.session_state['df_results'].empty:
        st.markdown("---")
        st.header("📈 Performance Results")
        
        # Model Metrics in Cards
        results_col1, results_col2, results_col3, results_col4 = st.columns(4)
        df_results = st.session_state['df_results']
        
        metrics_data = []
        for _, row in df_results.iterrows():
            metrics_data.append({
                'model': row['Model'],
                'accuracy': row['Accuracy'],
                'f1': row['F1-Score'],
                'training_time': row['Training Time (s)']
            })
        
        for i, metric in enumerate(metrics_data):
            col = [results_col1, results_col2, results_col3, results_col4][i]
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{metric['model']}</h3>
                    <h2>🏆 {metric['accuracy']:.1f}%</h2>
                    <p>F1: {metric['f1']:.3f} | Time: {metric['training_time']}s</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Detailed Results and Visualizations
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            st.subheader("📊 Performance Metrics")
            chart_metric = st.selectbox(
                "Select metric to visualize:",
                ['Accuracy', 'F1-Score', 'Precision', 'Recall', 'Training Time (s)', 'Inference Latency (ms)'],
                key='chart_metric'
            )
            
            chart_data = df_results[['Model', chart_metric]].set_index('Model')
            st.bar_chart(chart_data)
        
        with viz_col2:
            st.subheader("⚡ Speed vs Accuracy Trade-off")
            
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = ['#2563eb', '#7c3aed', '#06d6a0', '#f59e0b']
            
            for i, (_, row) in enumerate(df_results.iterrows()):
                ax.scatter(row['Inference Latency (ms)'], row['Accuracy'], 
                          s=200, alpha=0.7, color=colors[i], label=row['Model'])
                ax.annotate(row['Model'], 
                           (row['Inference Latency (ms)'] + 5, row['Accuracy']), 
                           fontsize=9, alpha=0.8)
            
            ax.set_xlabel('Inference Latency (ms)')
            ax.set_ylabel('Accuracy (%)')
            ax.set_title('Model Performance: Speed vs Accuracy')
            ax.grid(True, alpha=0.3)
            ax.legend()
            st.pyplot(fig)

        # Google Benchmark Results
        if not st.session_state['google_benchmark_results'].empty:
            st.markdown("---")
            st.header("🌐 Google Fact Check Benchmark")
            
            google_results = st.session_state['google_benchmark_results']
            politifacts_results = st.session_state['df_results']
            
            # Comparison metrics
            st.subheader("Performance Comparison")
            comp_col1, comp_col2, comp_col3, comp_col4 = st.columns(4)
            
            for idx, (_, row) in enumerate(google_results.iterrows()):
                model_name = row['Model']
                google_accuracy = row['Accuracy']
                
                # Find corresponding Politifacts accuracy
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
                        st.metric(
                            label=f"{model_name}",
                            value=f"{google_accuracy:.1f}%",
                            delta=f"{delta:+.1f}%",
                            delta_color=delta_color
                        )
                    else:
                        st.metric(
                            label=f"{model_name}",
                            value=f"{google_accuracy:.1f}%"
                        )

    # ============================
    # HUMOROUS CRITIQUE SECTION
    # ============================
    
    if not st.session_state['df_results'].empty:
        st.markdown("---")
        st.header("🎭 AI Performance Review")
        
        critique_col1, critique_col2 = st.columns([2, 1])
        
        with critique_col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            critique_text = generate_humorous_critique(
                st.session_state['df_results'], 
                st.session_state['selected_phase_run']
            )
            st.markdown(critique_text)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with critique_col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("🏆 Winner's Circle")
            if not st.session_state['df_results'].empty:
                best_model = st.session_state['df_results'].loc[st.session_state['df_results']['F1-Score'].idxmax()]
                st.markdown(f"""
                **Champion Model:**  
                🥇 **{best_model['Model']}**
                
                **Performance:**  
                📊 {best_model['Accuracy']:.1f}% Accuracy  
                🎯 {best_model['F1-Score']:.3f} F1-Score  
                ⚡ {best_model['Inference Latency (ms)']}ms Inference
                
                **Feature Set:**  
                {st.session_state['selected_phase_run']}
                """)
            st.markdown('</div>', unsafe_allow_html=True)

    # ============================
    # EMPTY STATE MESSAGES
    # ============================
    
    if st.session_state['scraped_df'].empty and st.session_state['df_results'].empty:
        st.markdown("---")
        st.header("🚀 Getting Started")
        
        guide_col1, guide_col2, guide_col3 = st.columns(3)
        
        with guide_col1:
            st.markdown("""
            <div class="card">
            <h3>📥 Step 1: Collect Data</h3>
            <p>Use the Politifact scraper to gather training data. Select a date range and click "Scrape Politifact Data".</p>
            <ul>
                <li>Choose recent dates for current trends</li>
                <li>Allow 1-2 minutes for scraping</li>
                <li>Data is cached for repeated use</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with guide_col2:
            st.markdown("""
            <div class="card">
            <h3>🔧 Step 2: Configure Analysis</h3>
            <p>Select an NLP feature extraction method and run the model analysis:</p>
            <div class="feature-pill">Lexical</div>
            <div class="feature-pill">Syntactic</div>
            <div class="feature-pill">Semantic</div>
            <div class="feature-pill">Discourse</div>
            <div class="feature-pill">Pragmatic</div>
            <p>Each method extracts different linguistic features for classification.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with guide_col3:
            st.markdown("""
            <div class="card">
            <h3>🌐 Step 3: Real-World Test</h3>
            <p>Benchmark your models against live Google Fact Check data:</p>
            <ul>
                <li>Tests generalization to unseen data</li>
                <li>Uses real-time fact checks</li>
                <li>Compares performance deltas</li>
            </ul>
            <p><em>Requires Google API key in secrets.toml</em></p>
            </div>
            """, unsafe_allow_html=True)

# --- Run App ---
if __name__ == '__main__':
    app()

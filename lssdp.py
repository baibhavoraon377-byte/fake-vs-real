# ============================================
# 📌 NLP Phases for Fake vs Real Detection
# Using Naive Bayes at each step
# With Robust Preprocessing (LOCAL CSV VERSION)
# ============================================

# Install dependencies (run once if needed)
# pip install scikit-learn pandas nltk spacy textblob

import pandas as pd
import numpy as np
import nltk, string, spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from textblob import TextBlob

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# ============================
# Download NLTK resources
# ============================
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# ============================
# Load spaCy model
# ============================
nlp = spacy.load("en_core_web_sm")

# ============================
# Step 1: Load Dataset
# ============================
# 👉 Put your CSV file name here
df = pd.read_csv("your_dataset.csv")   # Must have 'Statement' & 'BinaryTarget'

print("Dataset Shape:", df.shape)
print("Columns:", df.columns)
print(df.head())

X = df['Statement']
y = df['BinaryTarget']

# ============================
# Step 2: Robust Preprocessing
# ============================
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def robust_preprocess(text):
    if pd.isnull(text):
        return ""
    text = str(text).lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha()]  # keep only alphabetic
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return " ".join(tokens)

# Apply preprocessing globally
X_processed = X.astype(str).apply(robust_preprocess)

# Drop rows that became empty after preprocessing
mask_global = X_processed.str.strip().astype(bool)
X_processed, y_filtered = X_processed[mask_global], y[mask_global]


# ============================
# Helper: Train Naive Bayes
# ============================
def train_nb(X_features, y, name):
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=0.2, random_state=42
    )
    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n🔹 {name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    return acc


# ============================
# Phase 1: Lexical & Morphological Analysis
# ============================
X_lexical = X_processed.copy()
mask_lex = X_lexical.str.strip().astype(bool)
X_lexical, y_lex = X_lexical[mask_lex], y_filtered[mask_lex]

if X_lexical.empty:
    print("\n🔹 Lexical & Morphological Analysis: No documents left after filtering.")
    acc1 = None
else:
    vec_lexical = CountVectorizer().fit_transform(X_lexical)
    acc1 = train_nb(vec_lexical, y_lex, "Lexical & Morphological Analysis")


# ============================
# Phase 2: Syntactic Analysis
# ============================
def syntactic_features(text):
    doc = nlp(text)
    pos_tags = " ".join([token.pos_ for token in doc])
    return pos_tags

X_syntax = X_processed.apply(syntactic_features)
mask_syn = X_syntax.str.strip().astype(bool)
X_syntax, y_syn = X_syntax[mask_syn], y_filtered[mask_syn]

if X_syntax.empty:
    print("\n🔹 Syntactic Analysis: No documents left after filtering.")
    acc2 = None
else:
    vec_syntax = CountVectorizer().fit_transform(X_syntax)
    acc2 = train_nb(vec_syntax, y_syn, "Syntactic Analysis")


# ============================
# Phase 3: Semantic Analysis
# ============================
def semantic_features(text):
    blob = TextBlob(text)
    return f"{blob.sentiment.polarity} {blob.sentiment.subjectivity}"

X_semantic = X_processed.apply(semantic_features)
mask_sem = X_semantic.str.strip().astype(bool)
X_semantic, y_sem = X_semantic[mask_sem], y_filtered[mask_sem]

if X_semantic.empty:
    print("\n🔹 Semantic Analysis: No documents left after filtering.")
    acc3 = None
else:
    vec_semantic = TfidfVectorizer().fit_transform(X_semantic)
    acc3 = train_nb(vec_semantic, y_sem, "Semantic Analysis")


# ============================
# Phase 4: Discourse Integration
# ============================
def discourse_features(text):
    sentences = sent_tokenize(text)
    if len(sentences) == 0:
        return "0"
    return f"{len(sentences)} {' '.join([s.split()[0] for s in sentences if len(s.split())>0])}"

X_discourse = X_processed.apply(discourse_features)
mask_disc = X_discourse.str.strip().astype(bool)
X_discourse, y_disc = X_discourse[mask_disc], y_filtered[mask_disc]

if X_discourse.empty:
    print("\n🔹 Discourse Integration: No documents left after filtering.")
    acc4 = None
else:
    vec_discourse = CountVectorizer().fit_transform(X_discourse)
    acc4 = train_nb(vec_discourse, y_disc, "Discourse Integration")


# ============================
# Phase 5: Pragmatic Analysis (NUMERIC MATRIX)
# ============================
pragmatic_words = ["must", "should", "might", "could", "will", "?", "!"]

def pragmatic_features(text):
    features = []
    text_lower = text.lower()
    for w in pragmatic_words:
        features.append(str(text_lower.count(w)))
    return " ".join(features)

X_pragmatic = X_processed.apply(pragmatic_features)
mask_prag = X_pragmatic.str.strip().astype(bool)
X_pragmatic, y_prag = X_pragmatic[mask_prag], y_filtered[mask_prag]

if X_pragmatic.empty:
    print("\n🔹 Pragmatic Analysis: No documents left after filtering.")
    acc5 = None
else:
    # Convert the count strings to a numeric matrix
    X_pragmatic_matrix = np.array([list(map(int, s.split())) for s in X_pragmatic])
    acc5 = train_nb(X_pragmatic_matrix, y_prag, "Pragmatic Analysis")


# ============================
# Final Results
# ============================
print("\n📊 Phase-wise Naive Bayes Accuracies:")
print(f"1. Lexical & Morphological: {acc1 if acc1 is not None else 'N/A'}")
print(f"2. Syntactic: {acc2 if acc2 is not None else 'N/A'}")
print(f"3. Semantic: {acc3 if acc3 is not None else 'N/A'}")
print(f"4. Discourse: {acc4 if acc4 is not None else 'N/A'}")
print(f"5. Pragmatic: {acc5 if acc5 is not None else 'N/A'}")

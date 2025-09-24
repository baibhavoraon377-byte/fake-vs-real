# --------------------------------------------------------------
# Fake vs Real – Lexical / Syntactic / Semantic / Sentiment / Pragmatic
# Streamlit App (auto-detects column names)
# --------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Download NLTK data (only needs to run once per environment)
nltk.download("punkt")
nltk.download("stopwords")

st.title("Fake vs Real Detection – LSSDP Pipeline (Naive Bayes)")

# --------------------------------------------------------------
# 1. Upload CSV
# --------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload CSV (columns can be 'Statement'/'BinaryTarget' OR 'text'/'label')",
    type="csv",
)
if uploaded_file is None:
    st.stop()

df = pd.read_csv(uploaded_file)
st.write("### Dataset Preview", df.head())

# --------------------------------------------------------------
# 2. Detect correct column names
# --------------------------------------------------------------
possible_text_cols = ["Statement", "text"]
possible_label_cols = ["BinaryTarget", "label"]

text_col = next((c for c in possible_text_cols if c in df.columns), None)
label_col = next((c for c in possible_label_cols if c in df.columns), None)

if text_col is None or label_col is None:
    st.error("CSV must have either ('Statement','BinaryTarget') or ('text','label') columns.")
    st.write("Found columns:", list(df.columns))
    st.stop()

# --------------------------------------------------------------
# 3. Text cleaning
# --------------------------------------------------------------
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[0-9]+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    return " ".join(tokens)

df["clean"] = df[text_col].astype(str).apply(clean_text)

# --------------------------------------------------------------
# 4. Helper: Train & evaluate Naive Bayes
# --------------------------------------------------------------
def train_nb(X, y, name):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
    nb = MultinomialNB()
    nb.fit(Xtr, ytr)
    pred = nb.predict(Xte)
    acc = accuracy_score(yte, pred)
    st.write(f"**{name} Accuracy:** {acc*100:.2f}%")
    return acc

# --------------------------------------------------------------
# 5. Lexical Features – Bag of Words
# --------------------------------------------------------------
st.subheader("Lexical Analysis")
vec_lexical = CountVectorizer()
X_lexical = vec_lexical.fit_transform(df["clean"])
train_nb(X_lexical, df[label_col], "Lexical")

# --------------------------------------------------------------
# 6. Syntactic Features – Bigrams
# --------------------------------------------------------------
st.subheader("Syntactic Analysis")
vec_syntactic = CountVectorizer(ngram_range=(2, 2))
X_syntactic = vec_syntactic.fit_transform(df["clean"])
train_nb(X_syntactic, df[label_col], "Syntactic")

# --------------------------------------------------------------
# 7. Semantic Features – TF-IDF
# --------------------------------------------------------------
st.subheader("Semantic Analysis")
vec_semantic = TfidfVectorizer()
X_semantic = vec_semantic.fit_transform(df["clean"])
train_nb(X_semantic, df[label_col], "Semantic")

# --------------------------------------------------------------
# 8. Sentiment Features – toy polarity count
# --------------------------------------------------------------
st.subheader("Sentiment Analysis")
positive_words = {"good", "great", "excellent", "positive", "fortunate", "correct", "superior"}
negative_words = {"bad", "terrible", "poor", "negative", "wrong", "inferior"}

def sentiment_vector(text):
    tokens = text.split()
    pos = sum(t in positive_words for t in tokens)
    neg = sum(t in negative_words for t in tokens)
    return f"{pos} {neg}"

sent_feats = df["clean"].apply(sentiment_vector)
vec_sentiment = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
X_sentiment = vec_sentiment.fit_transform(sent_feats)
train_nb(X_sentiment, df[label_col], "Sentiment")

# --------------------------------------------------------------
# 9. Pragmatic Features – simple numeric counts
# --------------------------------------------------------------
st.subheader("Pragmatic Analysis")

def pragmatic_vector(text):
    words = text.split()
    word_count = len(words)
    avg_word_len = np.mean([len(w) for w in words]) if words else 0
    punctuation_ratio = sum(ch in string.punctuation for ch in text) / (len(text) + 1)
    return f"{word_count} {avg_word_len:.2f} {punctuation_ratio:.2f}"

prag_feats = df[text_col].astype(str).apply(pragmatic_vector)
X_pragmatic = np.array([list(map(float, s.split())) for s in prag_feats])

# MultinomialNB requires non-negative integers → scale
X_pragmatic_int = np.round(X_pragmatic * 100).astype(int)
train_nb(X_pragmatic_int, df[label_col], "Pragmatic")

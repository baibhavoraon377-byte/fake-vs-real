# --------------------------------------------------------------
# Fake vs Real – Lexical / Syntactic / Semantic / Sentiment / Pragmatic
# Streamlit App (LOCAL CSV with Statement & BinaryTarget)
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

nltk.download('punkt')
nltk.download('stopwords')

st.title("Fake vs Real Detection – LSSDP Pipeline (Naive Bayes)")

# --------------------------------------------------------------
# 1. Upload CSV
# --------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload CSV (must contain 'Statement' & 'BinaryTarget' columns)", type="csv"
)
if uploaded_file is None:
    st.stop()

df = pd.read_csv(uploaded_file)
st.write("Dataset Preview", df.head())

# Ensure required columns exist
if not {"Statement", "BinaryTarget"}.issubset(df.columns):
    st.error("CSV must have 'Statement' and 'BinaryTarget' columns.")
    st.stop()

# --------------------------------------------------------------
# 2. Helper: clean text
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

df["clean"] = df["Statement"].astype(str).apply(clean_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df["clean"], df["BinaryTarget"], test_size=0.25, random_state=42
)

# --------------------------------------------------------------
# 3. Train function
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
# 4. Lexical Features – bag of words
# --------------------------------------------------------------
st.subheader("Lexical Analysis")
vec_lexical = CountVectorizer()
X_lexical = vec_lexical.fit_transform(df["clean"])
train_nb(X_lexical, df["BinaryTarget"], "Lexical")

# --------------------------------------------------------------
# 5. Syntactic Features – n-grams (bigrams)
# --------------------------------------------------------------
st.subheader("Syntactic Analysis")
vec_syntactic = CountVectorizer(ngram_range=(2, 2))
X_syntactic = vec_syntactic.fit_transform(df["clean"])
train_nb(X_syntactic, df["BinaryTarget"], "Syntactic")

# --------------------------------------------------------------
# 6. Semantic Features – TF-IDF
# --------------------------------------------------------------
st.subheader("Semantic Analysis")
vec_semantic = TfidfVectorizer()
X_semantic = vec_semantic.fit_transform(df["clean"])
train_nb(X_semantic, df["BinaryTarget"], "Semantic")

# --------------------------------------------------------------
# 7. Sentiment Features – toy polarity count
# --------------------------------------------------------------
st.subheader("Sentiment Analysis")
positive_words = {"good","great","excellent","positive","fortunate","correct","superior"}
negative_words = {"bad","terrible","poor","negative","wrong","inferior"}

def sentiment_vector(text):
    tokens = text.split()
    pos = sum(t in positive_words for t in tokens)
    neg = sum(t in negative_words for t in tokens)
    return f"{pos} {neg}"

sent_feats = df["clean"].apply(sentiment_vector)
vec_sentiment = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
X_sentiment = vec_sentiment.fit_transform(sent_feats)
train_nb(X_sentiment, df["BinaryTarget"], "Sentiment")

# --------------------------------------------------------------
# 8. Pragmatic Features – numeric counts
# --------------------------------------------------------------
st.subheader("Pragmatic Analysis")

def pragmatic_vector(text):
    words = text.split()
    word_count = len(words)
    avg_word_len = np.mean([len(w) for w in words]) if words else 0
    punctuation_ratio = sum(ch in string.punctuation for ch in text) / (len(text)+1)
    return f"{word_count} {avg_word_len:.2f} {punctuation_ratio:.2f}"

prag_feats = df["Statement"].astype(str).apply(pragmatic_vector)
X_pragmatic = np.array([list(map(float, s.split())) for s in prag_feats])

# MultinomialNB needs integers → scale
X_pragmatic_int = np.round(X_pragmatic * 100).astype(int)
train_nb(X_pragmatic_int, df["BinaryTarget"], "Pragmatic")

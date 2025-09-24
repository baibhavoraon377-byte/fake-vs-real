# app.py
# ============================================
# 📌 NLP Phases for Fake vs Real Detection
# Using Naive Bayes + Streamlit UI
# ============================================

import streamlit as st
import pandas as pd
import nltk, string, spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# ============================
# Setup
# ============================
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")
nlp = spacy.load("en_core_web_sm")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# ============================
# Helpers
# ============================
def train_nb(X_features, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=0.2, random_state=42
    )
    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return acc, report

def lexical_preprocess(text):
    tokens = nltk.word_tokenize(str(text).lower())
    tokens = [
        lemmatizer.lemmatize(w)
        for w in tokens
        if w not in stop_words and w not in string.punctuation
    ]
    return " ".join(tokens)

def syntactic_features(text):
    doc = nlp(str(text))
    pos_tags = " ".join([token.pos_ for token in doc])
    return pos_tags

def semantic_features(text):
    blob = TextBlob(str(text))
    return f"{blob.sentiment.polarity} {blob.sentiment.subjectivity}"

def discourse_features(text):
    sentences = nltk.sent_tokenize(str(text))
    return f"{len(sentences)} {' '.join([s.split()[0] for s in sentences if len(s.split())>0])}"
# Streamlit UI
# ============================
st.set_page_config(page_title="Fake vs Real Detection (Naive Bayes)", layout="wide")
st.title("📰 Fake vs Real News Detection")
st.write("Naive Bayes classifier across different NLP phases")

uploaded_file = st.file_uploader("politifact_full.csv", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file).head(5000)  # limit for speed
    st.write("Dataset Preview:", df.head())

    # Make sure BinaryTarget exists
    if "BinaryTarget" not in df.columns:
        df["BinaryTarget"] = df["Rating"].apply(
            lambda x: 1 if x in ["true", "mostly-true", "half-true"] else 0
        )

    X = df["statement"]
    y = df["BinaryTarget"]

    phase = st.selectbox(
        "Choose NLP Phase",
        ["Lexical & Morphological", "Syntactic", "Semantic", "Discourse"],
    )

    if st.button("Run Training"):
        if phase == "Lexical & Morphological":
            X_proc = X.apply(lexical_preprocess)
            vec = CountVectorizer().fit_transform(X_proc)
        elif phase == "Syntactic":
            X_proc = X.apply(syntactic_features)
            vec = CountVectorizer().fit_transform(X_proc)
        elif phase == "Semantic":
            X_proc = X.apply(semantic_features)
            vec = TfidfVectorizer().fit_transform(X_proc)
        elif phase == "Discourse":
            X_proc = X.apply(discourse_features)
            vec = CountVectorizer().fit_transform(X_proc)

        acc, report = train_nb(vec, y)
        st.success(f"✅ {phase} Accuracy: {acc:.4f}")
        st.json(report)

    st.subheader("🔍 Test on Custom Input")
    user_text = st.text_area("Enter a statement to classify:")
    if st.button("Classify Statement") and user_text.strip():
        if phase == "Lexical & Morphological":
            processed = lexical_preprocess(user_text)
            vec = CountVectorizer().fit(X.apply(lexical_preprocess))
        elif phase == "Syntactic":
            processed = syntactic_features(user_text)
            vec = CountVectorizer().fit(X.apply(syntactic_features))
        elif phase == "Semantic":
            processed = semantic_features(user_text)
            vec = TfidfVectorizer().fit(X.apply(semantic_features))
        elif phase == "Discourse":
            processed = discourse_features(user_text)
            vec = CountVectorizer().fit(X.apply(discourse_features))

        X_features = vec.transform([processed])
        model = MultinomialNB().fit(vec.transform(X.apply(
            lexical_preprocess if phase=="Lexical & Morphological" 
            else syntactic_features if phase=="Syntactic"
            else semantic_features if phase=="Semantic"
            else discourse_features
        )), y)

        pred = model.predict(X_features)[0]
        st.write("Prediction:", "✅ True" if pred == 1 else "❌ Fake")

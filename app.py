import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt
import seaborn as sns

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(page_title="Intelligent NLP Chatbot", layout="wide")

# ======================
# HEADER (CHATGPT STYLE)
# ======================
st.markdown("""
    <style>
    .main-title {
        font-size:40px;
        font-weight:700;
        color:#4F8BF9;
    }
    .sub-title {
        font-size:18px;
        color:gray;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🤖 Intelligent FAQ Chatbot System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">NLP-based Smart Response + Evaluation Dashboard</div>', unsafe_allow_html=True)

# ======================
# LOAD DATA
# ======================
df = pd.read_csv("faq.csv")
df.columns = df.columns.str.strip().str.lower()

# ======================
# TRAIN / TEST SPLIT (REAL ML)
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    df["question"],
    df["answer"],
    test_size=0.3,
    random_state=42
)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ======================
# SIDEBAR MENU
# ======================
menu = st.sidebar.radio(
    "📌 Navigation",
    ["🏠 Home", "💬 Chatbot", "📊 Performance", "📂 Dataset"]
)

# ======================
# HOME
# ======================
if menu == "🏠 Home":
    st.subheader("📌 Project Overview")

    st.markdown("""
### 🎯 Problem Statement
Users struggle to manually search FAQ information efficiently.

### 🎯 Objective
Build an NLP-based chatbot that automatically answers FAQ questions.

### ⚙️ Methodology
- TF-IDF Vectorization
- Cosine Similarity
- Machine Learning Evaluation

### 🎯 Expected Output
- Smart FAQ chatbot
- Performance metrics (Accuracy, F1, etc.)
- Interactive dashboard
""")

# ======================
# CHATBOT
# ======================
elif menu == "💬 Chatbot":
    st.subheader("💬 Ask Your Question")

    user_input = st.selectbox("Select a question:", df["question"].tolist())

    if st.button("Get Answer"):
        user_vec = vectorizer.transform([user_input])
        similarity = cosine_similarity(user_vec, X_train_vec)

        index = np.argmax(similarity)
        confidence = np.max(similarity) * 100

        st.success("Answer: " + list(y_train)[index])

        st.info(f"Confidence Score: {confidence:.2f}%")

        st.progress(int(confidence))

# ======================
# PERFORMANCE (REAL METRICS)
# ======================
elif menu == "📊 Performance":
    st.subheader("📊 Model Evaluation Metrics")

    y_pred = []

    for q in X_test:
        q_vec = vectorizer.transform([q])
        sim = cosine_similarity(q_vec, X_train_vec)
        idx = np.argmax(sim)
        y_pred.append(list(y_train)[idx])

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=1)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", f"{acc:.2f}")
    col2.metric("Precision", f"{prec:.2f}")
    col3.metric("Recall", f"{rec:.2f}")
    col4.metric("F1 Score", f"{f1:.2f}")

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred, labels=y_test.unique())

    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)

# ======================
# DATASET
# ======================
elif menu == "📂 Dataset":
    st.subheader("📂 FAQ Dataset")
    st.dataframe(df)

    st.write("Total FAQs:", len(df))

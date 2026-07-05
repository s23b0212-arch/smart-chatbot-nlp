
import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Smart FAQ Chatbot",
    page_icon="🤖",
    layout="wide"
)

# ----------------------------
# LOAD DATASET
# ----------------------------
df = pd.read_csv("faq.csv")

questions = df["question"].tolist()
answers = df["answer"].tolist()

# ----------------------------
# TF-IDF MODEL
# ----------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

# ----------------------------
# SESSION STATE (CHAT HISTORY)
# ----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ----------------------------
# SIDEBAR
# ----------------------------
st.sidebar.title("🤖 Smart Chatbot")
page = st.sidebar.radio("Navigation", ["Home", "Chatbot", "About"])

st.sidebar.markdown("---")
st.sidebar.info("Built using NLP (TF-IDF + Cosine Similarity)")

# ----------------------------
# HOME PAGE
# ----------------------------
if page == "Home":
    st.title("🤖 Smart FAQ Chatbot System")

    st.markdown("""
    ### 📌 Project Overview
    This chatbot answers frequently asked questions using Natural Language Processing.

    ### 🧠 How it works
    - User enters a question
    - System compares it with stored FAQ dataset
    - Finds most similar question using TF-IDF
    - Returns best matching answer

    ### ⚙️ Technologies Used
    - Python
    - Streamlit
    - TF-IDF (Scikit-learn)
    - NLP (Text Processing)

    ### 🎯 Objective
    To build an intelligent FAQ chatbot that responds automatically to user queries.
    """)

# ----------------------------
# CHATBOT PAGE
# ----------------------------
elif page == "Chatbot":
    st.title("💬 Chat with AI Bot")

    st.write("Select a sample question or type your own:")

    sample_question = st.selectbox(
        "Quick Questions",
        ["", *questions]
    )

    user_input = st.text_input("Enter your question:", sample_question)

    def get_response(query):
        query_vec = vectorizer.transform([query])
        similarity = cosine_similarity(query_vec, X)
        index = np.argmax(similarity)
        score = similarity[0][index]

        if score < 0.2:
            return "Sorry, I don't understand that question. Try asking differently."
        return answers[index]

    if st.button("Ask"):
        if user_input.strip() != "":
            response = get_response(user_input)

            # store chat history
            st.session_state.chat_history.append((user_input, response))

            st.success("Answer:")
            st.write(response)

    # ----------------------------
    # CHAT HISTORY (ChatGPT STYLE)
    # ----------------------------
    st.markdown("---")
    st.subheader("🧾 Chat History")

    for q, a in reversed(st.session_state.chat_history):
        st.markdown(f"""
        <div style="background-color:#1e1e1e;padding:10px;border-radius:10px;margin-bottom:10px">
        <b>🧑 You:</b> {q}<br>
        <b>🤖 Bot:</b> {a}
        </div>
        """, unsafe_allow_html=True)

# ----------------------------
# ABOUT PAGE
# ----------------------------
elif page == "About":
    st.title("ℹ️ About Project")

    st.markdown("""
    ### 🤖 Smart FAQ Chatbot

    This project is built for Natural Language Processing coursework.

    ### 🎯 Features:
    - FAQ-based chatbot
    - TF-IDF similarity matching
    - Streamlit interactive UI
    - Chat history system
    - Dropdown suggestion input

    ### 🧠 Model:
    TF-IDF + Cosine Similarity

    ### 📊 Output:
    User asks questions → system finds closest match → returns answer
    """)

# ----------------------------
# FOOTER
# ----------------------------
st.markdown("---")
st.markdown("💡 Built using Streamlit + NLP | Smart FAQ Chatbot")


import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
import os

# Ensure NLTK stopwords are downloaded (if not already)
try:
    nltk.download('stopwords', quiet=True)
except LookupError:
    pass
arabic_stopwords = set(stopwords.words('arabic'))

# --- Preprocessing functions ---
def normalize_arabic(text):
    text = re.sub(r'[ً-ْ]', '', text)
    text = re.sub(r'[أإآ]', 'ا', text)
    text = re.sub(r'ؤ', 'و', text)
    text = re.sub(r'[ئي]', 'ي', text)
    text = re.sub(r'ـ', '', text)
    text = re.sub(r'[^؀-ۿ\s]', '', text) # Fixed regex here
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_arabic_stopwords(text):
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word not in arabic_stopwords]
    return ' '.join(filtered_tokens)

# --- Load the saved model and vectorizer ---
model_filename_tuned = 'logistic_regression_tuned_model.joblib'
vectorizer_filename = 'tfidf_vectorizer.joblib'

logistic_model_tuned = None
tfidf_vectorizer = None

if os.path.exists(model_filename_tuned) and os.path.exists(vectorizer_filename):
    try:
        logistic_model_tuned = joblib.load(model_filename_tuned)
        tfidf_vectorizer = joblib.load(vectorizer_filename)
        st.sidebar.success("Model and Vectorizer loaded successfully.")
    except Exception as e:
        st.sidebar.error(f"Error loading model or vectorizer: {e}")
        st.stop()
else:
    st.sidebar.error(f"Model files '{model_filename_tuned}' or '{vectorizer_filename}' not found. Please ensure they are in the same directory as the app.")
    st.stop()


# --- Streamlit App Layout ---
st.title("Arabic Fake News Classification")
st.write("Enter an Arabic news article below to classify it as 'real' or 'fake'.")

# Text input
article_input = st.text_area("News Article (Arabic)", height=200)

if st.button("Classify Article"):
    if article_input:
        with st.spinner('Classifying...'):
            # Preprocess the input article
            normalized_text = normalize_arabic(article_input)
            cleaned_text = remove_arabic_stopwords(normalized_text)

            # Transform the text using the loaded TF-IDF vectorizer
            article_tfidf = tfidf_vectorizer.transform([cleaned_text])

            # Make a prediction
            prediction = logistic_model_tuned.predict(article_tfidf)
            prediction_proba = logistic_model_tuned.predict_proba(article_tfidf)

            predicted_label = prediction[0]
            # Find the probability of the predicted label
            if predicted_label == 'real':
                confidence = prediction_proba[0][list(logistic_model_tuned.classes_).index('real')]
            else:
                confidence = prediction_proba[0][list(logistic_model_tuned.classes_).index('fake')]

            st.write("
---")
            st.subheader("Prediction Result:")
            st.metric(label="Predicted Label", value=predicted_label.capitalize())
            st.metric(label="Confidence", value=f"{confidence:.2%}")
            st.write("
---")
    else:
        st.warning("Please enter an article to classify.")

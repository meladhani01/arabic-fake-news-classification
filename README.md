Arabic Fake News Classification Project

​1. Project Overview

​This project aims to detect and classify Arabic news articles as either "real" or "fake" using Natural Language Processing (NLP) and Machine Learning techniques. The pipeline encompasses data ingestion, exploratory data analysis (EDA), Arabic-specific text preprocessing, feature extraction using TF-IDF, and classification using a hyperparameter-tuned Logistic Regression model. Furthermore, the project includes deployment implementations using both a Flask REST API and a Streamlit web application.

​2. Dataset Information

​Source File: fake_news_data.csv

​Total Records: 5,332 rows

​Features: 

​Label: Target variable indicating the authenticity of the news (real or fake).

​Topic: The category/domain of the news article (e.g., politics, sport).

​Article_content: The raw Arabic text of the news article.

​Class Distribution: The dataset is relatively balanced but leans slightly towards 'real' news: 

​Real: ~2,918 samples

​Fake: ~2,414 samples

​3. Data Processing Pipeline

​3.1. Text Preprocessing

​Given the complexities of the Arabic language, custom preprocessing functions were built using the re and nltk libraries to clean and normalize the text.

​Normalization (normalize_arabic): 

​Removes diacritics (Tashkeel).

​Normalizes variations of Hamza (إ، أ، آ to ا), Waw (ؤ to و), and Yeh (ئ، ى to ي).

​Removes character elongation (Tatweel).

​Strips non-alphabetic characters, numbers, and excess whitespace.

​Stopword Removal (remove_arabic_stopwords): 

​Filters out common Arabic stopwords using the NLTK library to reduce noise in the data.

​3.2. Feature Extraction (TF-IDF)

​The cleaned text was converted into numerical vectors using Scikit-learn's TfidfVectorizer.

​Configuration: 

​max_features: 10,000 (limits the vocabulary to the top 10,000 most frequent terms).

​ngram_range: (1, 2) (captures both single words/unigrams and two-word phrases/bigrams).

​min_df: 5 (ignores terms appearing in fewer than 5 documents).

​max_df: 0.7 (ignores terms appearing in more than 70% of the documents).

​sublinear_tf: True (applies logarithmic scaling to term frequencies).

​4. Modeling and Evaluation

​4.1. Model Training & Tuning

​Algorithm: Logistic Regression (LogisticRegression from sklearn.linear_model).

​Data Split: 80% Training (4,265 samples) / 20% Testing (1,067 samples) with stratification to maintain class distribution.

​Hyperparameter Tuning: Conducted using GridSearchCV (3-fold cross-validation) optimizing for the f1_macro score.

​Best Parameters Discovered: {'C': 10, 'penalty': 'l2', 'solver': 'liblinear'}.

​4.2. Model Performance

​The tuned Logistic Regression model demonstrated exceptional performance on the unseen test set:

​Accuracy: 98%

​F1-Score: 0.98 (Macro and Weighted Average)

​Precision/Recall: 

​Fake News: Precision 0.96, Recall 1.00

​Real News: Precision 1.00, Recall 0.97

​ROC AUC Score: 0.9970 (indicating near perfect separability between the classes).

​4.3. Serialization

​The trained model and the TF-IDF vectorizer were saved to disk for future deployment:

​Model file: logistic_regression_tuned_model.joblib

​Vectorizer file: tfidf_vectorizer.joblib

​5. Deployment Options

​The project includes two distinct methods for serving the model in a production or testing environment.

​5.1. REST API (Flask)

​A Flask web service was created to provide programmatic access to the model.

​Endpoint: /predict (POST)

​Input format (JSON): {"article": "عينة من النص العربي هنا..."}

​Output format (JSON): Returns the original article, the predicted_label ('real' or 'fake'), and a confidence score (probability).

​(Note: The project attempted to expose this API publicly via Google Colab using flask-ngrok, though network environment constraints may require running this locally or on a dedicated server).

​5.2. Web Application (Streamlit)

​An interactive web application was generated (streamlit_app.py) for end-users to test the model without writing code.

​Features: Provides a text area for users to paste an Arabic article.

​Functionality: Upon clicking "Classify Article", the app applies the exact same preprocessing and TF-IDF transformations, runs inference, and displays whether the news is Fake or Real along with a confidence percentage.

​Execution: Can be run locally using the command streamlit run streamlit_app.py.

​6. Dependencies

​To run this project, the following primary Python libraries are required:

​pandas

​matplotlib & seaborn

​nltk (with arabic stopwords downloaded)

​scikit-learn

​joblib

​flask (for API)

​streamlit (for Web App)

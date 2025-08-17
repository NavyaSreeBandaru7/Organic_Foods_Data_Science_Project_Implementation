import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from config import MODEL_CONFIG

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    """Clean and preprocess text data"""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Tokenize
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

def extract_features(df, text_column='description'):
    """Create TF-IDF features from text data"""
    df['cleaned_text'] = df[text_column].apply(preprocess_text)
    
    vectorizer = TfidfVectorizer(**MODEL_CONFIG['vectorizer_params'])
    tfidf_matrix = vectorizer.fit_transform(df['cleaned_text'])
    
    feature_names = vectorizer.get_feature_names_out()
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
    
    return pd.concat([df.reset_index(drop=True), tfidf_df], axis=1), vectorizer

def analyze_sentiment(text):
    """Simple sentiment analysis using rule-based approach"""
    positive_words = ['organic', 'natural', 'healthy', 'sustainable', 'fresh']
    negative_words = ['chemical', 'pesticide', 'artificial', 'processed']
    
    score = 0
    for word in positive_words:
        score += text.lower().count(word)
    for word in negative_words:
        score -= text.lower().count(word)
    
    return max(-1, min(1, score/10))  # Normalize to [-1, 1]

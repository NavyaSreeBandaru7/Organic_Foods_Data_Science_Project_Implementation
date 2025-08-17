import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from config import MODEL_CONFIG
from nlp.text_processing import extract_features

def train_organic_classifier(df):
    """Train ML model to predict organic status"""
    # Feature engineering
    df = create_features(df)
    
    # Text feature extraction
    df, vectorizer = extract_features(df)
    
    # Prepare data
    X = df.drop(columns=['is_organic', 'description', 'cleaned_text'])
    y = df['is_organic']
    
    # Handle categorical data
    X = pd.get_dummies(X, columns=['category', 'certification', 'origin_country'])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=MODEL_CONFIG['test_size'],
        random_state=MODEL_CONFIG['random_state']
    )
    
    # Train model
    model = XGBClassifier(
        n_estimators=MODEL_CONFIG['n_estimators'],
        max_depth=MODEL_CONFIG['max_depth'],
        random_state=MODEL_CONFIG['random_state']
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    # Save artifacts
    joblib.dump(model, 'models/organic_classifier.pkl')
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
    
    return model, accuracy, report

def create_features(df):
    """Create additional features"""
    # Price per unit weight
    df['price_per_kg'] = df['price'] / df['weight']
    
    # Nutritional ratios
    df['protein_per_calorie'] = df['protein'] / df['calories']
    df['carb_fat_ratio'] = df['carbs'] / (df['fat'] + 0.001)
    
    # Sentiment analysis
    df['sentiment'] = df['description'].apply(analyze_sentiment)
    
    return df

# Import from text_processing to avoid circular dependency
from nlp.text_processing import analyze_sentiment

import re
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
from textblob import TextBlob

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load stopwords once to improve performance
stop_words = set(stopwords.words('english'))

def preprocess_text(text, n=2):
    # Ensure text is a string
    if not isinstance(text, str):
        raise ValueError("Text input must be a string.")
    
    # Remove HTML tags using an enhanced regex
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Convert text to lowercase
    text = text.lower()
    
    # Tokenization
    words = word_tokenize(text)
    
    # Lemmatization (stopwords are no longer removed)
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    # Generate n-grams (bigrams in this case, but you can set n=3 for trigrams, etc.)
    if n > 1:
        n_grams = list(ngrams(words, n))
        words = ['_'.join(gram) for gram in n_grams]  # Join n-gram tuples into a single string
    else:
        words = words  # If n=1, it's just the unigrams (individual words)
    
    return ' '.join(words)

def detect_anomalies(texts, contamination=0.03):
    # Convert texts to TF-IDF features
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(texts)
    
    # Use Isolation Forest to detect anomalies
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    anomalies = iso_forest.fit_predict(X)
    
    return anomalies == -1  # True for anomalies, False for normal data points

def sentiment_consistency_check(reviews, sentiments, threshold=0.5):
    inconsistent = []
    for review, sentiment in zip(reviews, sentiments):
        blob = TextBlob(review)
        polarity = blob.sentiment.polarity
        
        if (sentiment == 'positive' and polarity < -threshold) or \
           (sentiment == 'negative' and polarity > threshold):
            inconsistent.append(True)
        else:
            inconsistent.append(False)
    
    return inconsistent

def main():
    dataset_path = 'data/raw/IMDB_Dataset.csv'
    imdb_data = pd.read_csv(dataset_path)
    
    tqdm.pandas(desc="Processing reviews")
    imdb_data['cleaned_review'] = imdb_data['review'].progress_apply(lambda x: preprocess_text(x, n=2))
    
    # Detect anomalies
    print("Detecting anomalies...")
    anomalies = detect_anomalies(imdb_data['cleaned_review'])
    
    # Check sentiment consistency
    print("Checking sentiment consistency...")
    inconsistent = sentiment_consistency_check(imdb_data['review'], imdb_data['sentiment'])
    
    # Combine filters
    suspicious = np.logical_or(anomalies, inconsistent)
    
    # Remove suspicious data points
    clean_data = imdb_data[~suspicious].reset_index(drop=True)
    
    print(f"Removed {sum(suspicious)} suspicious data points out of {len(imdb_data)}.")
    
    clean_data.to_csv('data/processed/IMDB_Dataset_Cleaned_Filtered.csv', index=False)
    print("Data preprocessing and filtering complete. Saved to processed directory.")

if __name__ == "__main__":
    main()

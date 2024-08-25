import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load stopwords once to improve performance
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Ensure text is a string
    if not isinstance(text, str):
        raise ValueError("Text input must be a string.")
    
    # Remove HTML tags using an enhanced regex
    text = re.sub(r'<[^>]+>', ' ', text)  # This regex matches all content within angle brackets.
    
    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Adjusted to remove numbers if not needed
    
    # Convert text to lowercase
    text = text.lower()
    
    # Tokenization
    words = word_tokenize(text)
    
    # Remove stopwords
    words = [word for word in words if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    return ' '.join(words)

def main():
    dataset_path = 'data/raw/IMDB_Dataset.csv'
    imdb_data = pd.read_csv(dataset_path)
    
    tqdm.pandas(desc="Processing reviews")
    imdb_data['cleaned_review'] = imdb_data['review'].progress_apply(preprocess_text)
    
    imdb_data.to_csv('data/processed/IMDB_Dataset_Cleaned.csv', index=False)
    print("Data preprocessing complete and saved to processed directory.")

if __name__ == "__main__":
    main()

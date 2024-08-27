import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from tqdm import tqdm

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

def main():
    dataset_path = 'data/raw/IMDB_Dataset.csv'
    imdb_data = pd.read_csv(dataset_path)
    
    tqdm.pandas(desc="Processing reviews")
    imdb_data['cleaned_review'] = imdb_data['review'].progress_apply(lambda x: preprocess_text(x, n=2))  # Set n to 2 for bigrams
    
    imdb_data.to_csv('data/processed/IMDB_Dataset_Cleaned.csv', index=False)
    print("Data preprocessing complete and saved to processed directory.")

if __name__ == "__main__":
    main()

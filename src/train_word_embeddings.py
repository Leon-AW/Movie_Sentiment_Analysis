import pandas as pd
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
from tqdm import tqdm  # Import the tqdm library

nltk.download('punkt')

def load_data(file_path):
    """ Load the cleaned dataset """
    return pd.read_csv(file_path)

class MyCorpus:
    """ Custom iterator for gensim Word2Vec model with a progress bar """
    def __init__(self, tokenized_reviews):
        self.tokenized_reviews = tokenized_reviews

    def __iter__(self):
        for review in tqdm(self.tokenized_reviews, desc="Training Word2Vec Embeddings"):
            yield review

def train_embeddings(data):
    """ Train Word2Vec model on the tokenized sentences with a progress bar """
    # The data['cleaned_review'] column already contains the n-grams as single tokens.
    tokenized_reviews = [review.split() for review in data['cleaned_review']]

    # Creating the custom corpus with tqdm progress bar
    corpus = MyCorpus(tokenized_reviews)
    
    # Training the Word2Vec model with tuned hyperparameters
    model = Word2Vec(
        sentences=corpus, 
        vector_size=200,        
        window=5,               
        min_count=5,
        sg=1,                   
        workers=4,              
        epochs=10
    )
    
    # Check the vocabulary size
    vocab_size = len(model.wv)
    print(f"Vocabulary size: {vocab_size}")
    
    return model

def main():
    # Path to the cleaned dataset
    dataset_path = 'data/processed/IMDB_Dataset_Cleaned_Filtered.csv'
    
    # Load the cleaned text data
    imdb_data = load_data(dataset_path)
    
    # Train Word2Vec embeddings with progress bar
    w2v_model = train_embeddings(imdb_data)
    
    # Save only the word vectors
    w2v_model.wv.save_word2vec_format("models/word2vec_imdb_v2.bin", binary=True)
    print("Word2Vec vectors saved successfully!")

if __name__ == "__main__":
    main()

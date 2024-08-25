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
    # Tokenizing the sentences
    tokenized_reviews = [word_tokenize(review) for review in data['cleaned_review']]

    # Creating the custom corpus with tqdm progress bar
    corpus = MyCorpus(tokenized_reviews)
    
    # Training the Word2Vec model with tuned hyperparameters
    model = Word2Vec(
        sentences=corpus, 
        vector_size=200,        # Increase vector size for more detailed embeddings
        window=5,               # Context window size
        min_count=1,            # Include more rare words
        sg=1,                   # Use Skip-Gram model; try sg=0 for CBOW
        workers=4,              # Number of worker threads
        epochs=10               # Increase number of epochs
    )
    
    return model

def main():
    # Path to the cleaned dataset
    dataset_path = 'data/processed/IMDB_Dataset_Cleaned.csv'
    
    # Load the cleaned text data
    imdb_data = load_data(dataset_path)
    
    # Train Word2Vec embeddings with progress bar
    w2v_model = train_embeddings(imdb_data)
    
    # Save the model
    w2v_model.save("models/word2vec_imdb.model")
    print("Word2Vec model trained and saved successfully!")

if __name__ == "__main__":
    main()

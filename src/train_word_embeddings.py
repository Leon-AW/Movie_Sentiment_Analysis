import pandas as pd
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

def load_data(file_path):
    """ Load the cleaned dataset """
    return pd.read_csv(file_path)

def train_embeddings(data):
    """ Train Word2Vec model on the tokenized sentences """
    # Tokenizing the sentences
    tokenized_reviews = [word_tokenize(review) for review in data['cleaned_review']]
    
    # Training the Word2Vec model
    model = Word2Vec(sentences=tokenized_reviews, vector_size=100, window=5, min_count=2, workers=4)
    
    return model

def main():
    # Path to the cleaned dataset
    dataset_path = 'data/processed/IMDB_Dataset_Cleaned.csv'
    
    # Load the cleaned text data
    imdb_data = load_data(dataset_path)
    
    # Train Word2Vec embeddings
    w2v_model = train_embeddings(imdb_data)
    
    # Save the model
    w2v_model.save("models/word2vec_imdb.model")
    print("Word2Vec model trained and saved successfully!")

if __name__ == "__main__":
    main()

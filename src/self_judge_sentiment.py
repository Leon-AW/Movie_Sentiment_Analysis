import torch
import torch.nn as nn
from train_sentiment_model import FastTextClassifier, w2v_model, embedding_dim
import numpy as np
import re
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')

# Function to clean and preprocess the review
def preprocess_review(review):
    # Lowercase the review
    review = review.lower()

    # Remove special characters and digits
    review = re.sub(r'\d+', '', review)
    review = re.sub(r'\W+', ' ', review)

    # Tokenize the review using word_tokenize (identical to training preprocessing)
    words = word_tokenize(review)

    print(f"Preprocessed Words: {words}")  # Display the words after preprocessing

    # Convert words to vectors using Word2Vec
    vectors = [w2v_model.wv[word] for word in words if word in w2v_model.wv]

    # Average the vectors to get a single vector for the review
    if len(vectors) > 0:
        review_vector = np.mean(vectors, axis=0)
    else:
        review_vector = np.zeros(embedding_dim)  # if no words in review match the word2vec model

    return torch.tensor(review_vector, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

# Function to predict the sentiment of a review
def predict_sentiment(model, review):
    model.eval()
    with torch.no_grad():
        review_vector = preprocess_review(review)
        output = model(review_vector)
        _, predicted = torch.max(output, 1)
        sentiment = 'positive' if predicted.item() == 1 else 'negative'
    return sentiment

def main():
    # Load the trained model
    model = FastTextClassifier(embedding_dim)
    model.load_state_dict(torch.load("models/fasttext_classifier.pth"))

    while True:
        # Input review
        review = input("\nEnter a movie review (or type 'exit' to quit): ")
        
        if review.lower() == 'exit':
            print("Exiting the program.")
            break

        # Predict sentiment
        sentiment = predict_sentiment(model, review)
        print(f'The predicted sentiment is: {sentiment}')

if __name__ == "__main__":
    main()

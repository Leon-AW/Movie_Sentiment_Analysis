import gensim

def load_model(model_path):
    """ Load the pre-trained Word2Vec model in KeyedVectors format """
    return gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)

def find_most_frequent_ngram(model, word):
    """ Find the most frequent n-gram containing the word """
    matching_ngrams = [(ngram, model.get_vecattr(ngram, 'count')) for ngram in model.key_to_index if word in ngram]
    
    if matching_ngrams:
        # Sort by frequency (most frequent first)
        matching_ngrams.sort(key=lambda x: x[1], reverse=True)
        return matching_ngrams[0][0]  # Return the most frequent n-gram
    else:
        return None

def find_similar_words(model, word, top_n=5):
    """ Find and return the top N most similar words or n-grams """
    if word in model:
        similar_words = model.most_similar(word, topn=top_n)
        return similar_words
    else:
        # Find the most frequent n-gram containing the word
        most_frequent_ngram = find_most_frequent_ngram(model, word)
        
        if most_frequent_ngram:
            similar_words = model.most_similar(most_frequent_ngram, topn=top_n)
            return similar_words
        else:
            return None

def main():
    model_path = "models/word2vec_imdb_v2.bin"
    model = load_model(model_path)
    
    print("Word2Vec model loaded successfully!")
    print("Type a word or n-gram to find the 5 most similar words, or type 'exit' to quit.")
    
    while True:
        word = input("Enter a word or n-gram: ").strip()
        
        if word.lower() == 'exit':
            print("Exiting the program.")
            break
        
        similar_words = find_similar_words(model, word)
        
        if similar_words:
            print(f"Top {len(similar_words)} words or n-grams similar to '{word}':")
            for similar_word, similarity in similar_words:
                print(f"  {similar_word}: {similarity:.4f}")
        else:
            print(f"'{word}' not found in the model's vocabulary or in any n-grams.")

if __name__ == "__main__":
    main()

import gensim

def load_model(model_path):
    """ Load the pre-trained Word2Vec model """
    return gensim.models.Word2Vec.load(model_path)

def find_similar_words(model, word, top_n=5):
    """ Find and return the top N most similar words """
    if word in model.wv:
        similar_words = model.wv.most_similar(word, topn=top_n)
        return similar_words
    else:
        return None

def main():
    model_path = "models/word2vec_imdb.model"
    model = load_model(model_path)
    
    print("Word2Vec model loaded successfully!")
    print("Type a word to find the 5 most similar words, or type 'exit' to quit.")
    
    while True:
        word = input("Enter a word: ").strip()
        
        if word.lower() == 'exit':
            print("Exiting the program.")
            break
        
        similar_words = find_similar_words(model, word)
        
        if similar_words:
            print(f"Top {len(similar_words)} words similar to '{word}':")
            for similar_word, similarity in similar_words:
                print(f"  {similar_word}: {similarity:.4f}")
        else:
            print(f"'{word}' not found in the model's vocabulary.")

if __name__ == "__main__":
    main()

# Movie Sentiment Analysis with Word2Vec and FastText Classifier

This project implements a sentiment analysis pipeline for movie reviews using Word2Vec embeddings and a FastText-based neural network classifier. The goal is to classify movie reviews as either positive or negative based on the text content.

## Data

Movie Review data downloaded at Kaggle from: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

## Best Model Performance

My best training of the model achieved the following scores:

- Test Accuracy: 88.53%
- Precision: 0.8839
- Recall: 0.8872
- F1 Score: 0.8855

## Project Structure

- `data_preprocessing.py`: Preprocesses the raw IMDB movie reviews dataset by cleaning the text, generating n-grams, and saving the processed data.
- `train_word_embeddings.py`: Trains Word2Vec embeddings on the cleaned movie reviews, creating vector representations for words and n-grams.
- `train_sentiment_model.py`: Trains a FastText-based neural network model on the vectorized movie reviews to classify sentiments.
- `evaluate_model.py`: Evaluates the trained sentiment model on a test dataset and outputs performance metrics.
- `self_judge_embedding.py`: Allows users to interact with the trained Word2Vec model to find similar words or n-grams.
- `self_judge_sentiment.py`: Allows users to input their own movie reviews and predict the sentiment using the trained FastText model.

## Installation

To set up the environment, install the required Python packages:

```bash
pip install -r requirements.txt
```

Additionally, ensure that NLTK datasets are downloaded:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

## Usage

### 1. Preprocess the Data

Run the data preprocessing script:

```bash
python data_preprocessing.py
```

This will create a cleaned dataset in the `data/processed/` directory.

### 2. Train Word2Vec Embeddings

Train the Word2Vec embeddings on the cleaned dataset:

```bash
python train_word_embeddings.py
```

The trained embeddings will be saved in the `models/` directory.

### 3. Train the Sentiment Classifier

Train the sentiment analysis model:

```bash
python train_sentiment_model.py
```

This script will train the model, apply early stopping based on validation loss, and save the best model to `models/fasttext_classifier_best.pth`. Training and validation loss and accuracy plots will be saved in the `results/plots/` directory.

### 4. Evaluate the Model

Evaluate the trained model on the test set:

```bash
python evaluate_model.py
```

This script calculates and prints the test accuracy, precision, recall, and F1 score. The metrics are also saved to `results/metrics/evaluation_metrics.txt`.

### 5. Interact with Word2Vec Embeddings

Explore the embeddings using the `self_judge_embedding.py` script:

```bash
python self_judge_embedding.py
```

You can input a word or n-gram to find the most similar terms based on the trained Word2Vec model.

### 6. Predict Sentiment on Custom Reviews

Use the `self_judge_sentiment.py` script to input your own movie reviews and get sentiment predictions:

```bash
python self_judge_sentiment.py
```

This script preprocesses the input review, vectorizes it using the Word2Vec embeddings, and predicts whether the sentiment is positive or negative.

## Summary of Scripts

- **data_preprocessing.py**
  - Purpose: Preprocess the raw IMDB dataset by cleaning text, tokenizing, lemmatizing, and generating n-grams.
  - Output: A cleaned and processed CSV file saved in `data/processed/`.

- **train_word_embeddings.py**
  - Purpose: Train Word2Vec embeddings on the processed text data.
  - Output: A binary file containing the trained word vectors saved in `models/word2vec_imdb_v2.bin`.

- **train_sentiment_model.py**
  - Purpose: Train a FastText-based classifier using the vectorized n-grams to classify sentiments as positive or negative.
  - Output: The best model is saved to `models/fasttext_classifier_best.pth`, and the final model is saved to `models/fasttext_classifier_final.pth`.

- **evaluate_model.py**
  - Purpose: Evaluate the trained sentiment classifier on the test dataset and print/save performance metrics.
  - Output: Metrics saved in `results/metrics/evaluation_metrics.txt`.

- **self_judge_embedding.py**
  - Purpose: Allow users to explore the Word2Vec embeddings by finding similar words or n-grams.
  - Output: Prints the most similar terms to a given input.

- **self_judge_sentiment.py**
  - Purpose: Allow users to input custom movie reviews and predict their sentiment using the trained model.
  - Output: Prints the predicted sentiment for the input review.

## License

This project is open-source and available under the MIT License.

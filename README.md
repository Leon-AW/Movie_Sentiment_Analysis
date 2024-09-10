# Movie Sentiment Analysis with Word2Vec and FastText Classifier

I know you usually take the FastText Classifeier to train Word Embedding and Classifier simultaneously. But I wanted to do It separately to strengthen my understanding. This project implements a sentiment analysis pipeline for movie reviews using Word2Vec embeddings and then a FastText-based neural network classifier is trained on it. The goal is to classify movie reviews as either positive or negative based on the text content.

## Data

Movie Review data downloaded at Kaggle from: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

## Best Model Performance

After implementing Noise Robust Learning technique called "Detecting and filtering errors" (since about 3% of the data is noisy), the model performance improved slightly:

- Test Accuracy: 88.72% (↑0.19%)
- Precision: 0.8760 (↓0.0079)
- Recall: 0.9021 (↑0.0149)
- F1 Score: 0.8889 (↑0.0034)

Previous best performance:

- Test Accuracy: 88.53%
- Precision: 0.8839
- Recall: 0.8872
- F1 Score: 0.8855

The improvements in accuracy, recall, and F1 score demonstrate the effectiveness of the Noise Robust Learning techniques in enhancing the model's performance.

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

## Noise Robust Learning Features

This project implements several techniques for Noise Robust Learning to improve the model's performance on potentially noisy datasets:

### 1. Anomaly Detection

- **Isolation Forest**: Used to detect unusual reviews based on their TF-IDF representations.
- **Implementation**: `detect_anomalies()` function in `data_preprocessing.py`.
- **Purpose**: Identifies reviews that are statistically different from the majority, which could indicate mislabeling or outliers.

### 2. Sentiment Consistency Check

- **TextBlob Polarity**: Compares the assigned sentiment label with the calculated polarity of the review text.
- **Implementation**: `sentiment_consistency_check()` function in `data_preprocessing.py`.
- **Purpose**: Flags reviews where the assigned label contradicts the apparent sentiment of the text.

### 3. Data Cleaning

- Combines results from anomaly detection and sentiment consistency check.
- Removes suspicious data points from the training set.
- Improves the quality of the training data by eliminating potential labeling errors.

These techniques help in creating a more robust model by:

- Reducing the impact of mislabeled data
- Improving the overall quality of the training dataset
- Potentially increasing the model's generalization capabilities

## License

This project is open-source and available under the MIT License.

# Movie Sentiment Analysis

## Project Description

This project aims to perform sentiment analysis on movie reviews using the IMDB dataset. The project includes two main components:

1. **Word Embedding Training:** We use the Skip-gram architecture to train word embeddings directly on the IMDB dataset. This allows us to capture the semantic relationships between words in the context of movie reviews.

2. **Sentiment Analysis Using FastText:** After obtaining the word embeddings, we use the FastText classifier to predict the sentiment of the reviews (positive, negative, or neutral). The integration of word embeddings into the classifier aims to enhance the model's performance by providing richer word representations.

This project demonstrates key concepts in Natural Language Processing (NLP) including text preprocessing, word embedding generation, and sentiment classification. It is designed to be both educational and practical, making it applicable to real-world sentiment analysis tasks.

## Installation Instructions

To run this project, you need to have Python installed on your system. Follow the instructions below to set up the environment:

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/your-username/Movie_Sentiment_Analysis.git
    cd Movie_Sentiment_Analysis
    ```

2. **Create a Virtual Environment:**

    ```bash
    python -m venv nlpenv
    source nlpenv/bin/activate  # On Windows use: nlpenv\Scripts\activate
    ```

3. **Install the Required Dependencies:**

    Install the necessary Python packages listed in the `requirements.txt` file:

    ```bash
    pip install -r requirements.txt
    ```

4. **Download the IMDB Dataset:**

    You can download the IMDB dataset directly from [here](https://ai.stanford.edu/~amaas/data/sentiment/). After downloading, place the dataset in the `data/raw/` directory.

## Usage Examples

Once the environment is set up and the dataset is in place, you can run the project components as follows:

1. **Preprocess the Data:**

    Run the data preprocessing script to clean and tokenize the text data:

    ```bash
    python src/data_preprocessing.py
    ```

2. **Train Word Embeddings:**

    Use the Skip-gram model to train word embeddings on the preprocessed IMDB dataset:

    ```bash
    python src/train_word_embeddings.py
    ```

3. **Train the Sentiment Classifier:**

    Train the FastText classifier using the generated word embeddings:

    ```bash
    python src/train_sentiment_model.py
    ```

4. **Evaluate the Model:**

    Evaluate the model’s performance on the test set and generate metrics:

    ```bash
    python src/evaluate_model.py
    ```

## Directory Structure Explanation

The project directory is organized as follows:

```plaintext
Movie_Sentiment_Analysis/
├── data/
│   ├── raw/                # Contains the raw IMDB dataset
│   └── processed/          # Contains processed datasets (e.g., tokenized text)
├── src/
│   ├── data_preprocessing.py # Script for data preprocessing
│   ├── train_word_embeddings.py # Script for training word embeddings
│   ├── train_sentiment_model.py # Script for training the sentiment analysis model
│   └── evaluate_model.py       # Script for evaluating the model
├── models/
│   ├── word_embeddings/    # Stores trained word embeddings
│   └── sentiment_model/    # Stores the trained sentiment analysis model
├── logs/
│   ├── training.log        # Logs for training sessions
│   └── evaluation.log      # Logs for model evaluation
├── results/
│   ├── plots/              # Contains plots and visualizations
│   └── metrics/            # Stores evaluation metrics (e.g., F1-score, accuracy)
├── requirements.txt        # List of dependencies
├── .gitignore              # Specifies files and directories to ignore in Git
└── README.md               # Project description and instructions (this file)

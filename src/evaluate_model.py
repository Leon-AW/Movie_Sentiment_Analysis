import torch
import random
import pandas as pd
from tqdm import tqdm
from train_sentiment_model import FastTextClassifier, load_data, w2v_model, embedding_dim
from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_model(model, test_loader, test_data):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    # Lists to store correct and incorrect predictions
    correct_predictions = []
    incorrect_predictions = []

    with torch.no_grad():
        for batch_idx, (review_vector, label) in enumerate(tqdm(test_loader, desc="Evaluating", unit="batch")):
            output = model(review_vector)
            _, predicted = torch.max(output, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

            # Extend labels and predictions for evaluation
            all_labels.extend(label.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            # Store correct and incorrect predictions along with review text
            for i in range(len(label)):
                review_idx = batch_idx * test_loader.batch_size + i
                original_review = test_data.iloc[review_idx]['review']
                cleaned_review = test_data.iloc[review_idx]['cleaned_review']

                if predicted[i] == label[i]:
                    correct_predictions.append((original_review, cleaned_review, label[i].item(), predicted[i].item()))
                else:
                    incorrect_predictions.append((original_review, cleaned_review, label[i].item(), predicted[i].item()))

    test_accuracy = 100 * correct / total
    print(f'\nTest Accuracy: {test_accuracy:.2f}%')

    # Calculate Precision, Recall, and F1 Score
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)

    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

    # Show random 3 correct and 3 incorrect predictions
    print("\nSample Correct Predictions:")
    for sample in random.sample(correct_predictions, min(3, len(correct_predictions))):
        print(f"Review: {sample[0]}\nCleaned Review: {sample[1]}\nTrue Label: {sample[2]}, Predicted Label: {sample[3]}\n")

    print("\nSample Incorrect Predictions:")
    for sample in random.sample(incorrect_predictions, min(3, len(incorrect_predictions))):
        print(f"Review: {sample[0]}\nCleaned Review: {sample[1]}\nTrue Label: {sample[2]}, Predicted Label: {sample[3]}\n")

def main():
    data_path = 'data/processed/IMDB_Dataset_Cleaned.csv'

    # Load the dataset
    data = pd.read_csv(data_path)

    # Load the data and model
    _, _, test_loader = load_data(data_path)
    model = FastTextClassifier(embedding_dim)
    model.load_state_dict(torch.load("models/fasttext_classifier.pth"))

    # Evaluate the model
    evaluate_model(model, test_loader, data)

if __name__ == "__main__":
    main()

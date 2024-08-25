import torch
from train_sentiment_model import FastTextClassifier, load_data, w2v_model, embedding_dim
from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for review_vector, label in test_loader:
            output = model(review_vector)
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

            all_labels.extend(label.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    test_accuracy = 100 * correct / total
    print(f'Test Accuracy: {test_accuracy:.2f}%')

    # Calculate Precision, Recall, and F1 Score
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)

    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

def main():
    data_path = 'data/processed/IMDB_Dataset_Cleaned.csv'
    _, _, test_loader = load_data(data_path)

    model = FastTextClassifier(embedding_dim)
    model.load_state_dict(torch.load("models/fasttext_classifier.pth"))

    evaluate_model(model, test_loader)

if __name__ == "__main__":
    main()

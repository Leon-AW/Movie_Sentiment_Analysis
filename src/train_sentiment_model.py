import torch
import torch.nn as nn
import torch.optim as optim
from gensim.models import KeyedVectors
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import os

# Load the pre-trained Word2Vec model (KeyedVectors from the .bin file)
w2v_model = KeyedVectors.load_word2vec_format("models/word2vec_imdb_v2.bin", binary=True)

# Get the embedding dimension
embedding_dim = w2v_model.vector_size

class IMDBDataset(torch.utils.data.Dataset):
    def __init__(self, data, word2vec_model, embedding_dim):
        self.data = data
        self.word2vec_model = word2vec_model
        self.embedding_dim = embedding_dim

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        review = self.data.iloc[idx]['cleaned_review'].split()
        sentiment = self.data.iloc[idx]['sentiment']

        # Convert words to vectors
        vectors = [self.word2vec_model[word] for word in review if word in self.word2vec_model]

        # Average the vectors to get a single vector for the review
        if len(vectors) > 0:
            review_vector = np.mean(vectors, axis=0)
        else:
            review_vector = np.zeros(self.embedding_dim)  # if no words in review match the word2vec model

        # Convert sentiment to a binary label
        label = 1 if sentiment == 'positive' else 0

        return torch.tensor(review_vector, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def load_data(data_path):
    # Load the dataset and split into train, validation, and test sets
    data = pd.read_csv(data_path)
    train_val_data, test_data = train_test_split(data, test_size=0.15, random_state=42, stratify=data['sentiment'])
    train_data, val_data = train_test_split(train_val_data, test_size=0.1765, random_state=42, stratify=train_val_data['sentiment'])

    # Create datasets
    train_dataset = IMDBDataset(train_data, w2v_model, embedding_dim)
    val_dataset = IMDBDataset(val_data, w2v_model, embedding_dim)
    test_dataset = IMDBDataset(test_data, w2v_model, embedding_dim)

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, val_loader, test_loader

class FastTextClassifier(nn.Module):
    def __init__(self, embedding_dim):
        super(FastTextClassifier, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, 128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 2)  # Output layer for binary classification

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_model(model, train_loader, val_loader, num_epochs=10, patience=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    # Track the losses and accuracy
    train_losses, val_losses = [], []
    val_accuracies = []

    # Early stopping variables
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):  # Number of epochs
        model.train()
        running_loss = 0.0

        for batch_idx, (review_vector, label) in enumerate(train_loader):
            optimizer.zero_grad()  # Zero the gradients
            output = model(review_vector)  # Forward pass
            loss = criterion(output, label)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update parameters

            running_loss += loss.item()

            # Print batch progress
            if batch_idx % 10 == 9:  # Print every 10 batches
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        # Calculate average training loss
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}')

        # Validation step
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for review_vector, label in val_loader:
                output = model(review_vector)
                loss = criterion(output, label)
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        val_accuracy = 100 * correct / total
        val_accuracies.append(val_accuracy)
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

        # Step the scheduler based on validation loss
        scheduler.step(val_loss)

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save the best model
            torch.save(model.state_dict(), "models/fasttext_classifier_best.pth")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    # Save the final model
    torch.save(model.state_dict(), "models/fasttext_classifier_final.pth")
    print("Final model saved successfully!")

    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('results/plots/loss_plot.png')
    plt.show()

    # Plot validation accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.legend()
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.savefig('results/plots/accuracy_plot.png')
    plt.show()

def main():
    # Ensure the results/plots directory exists
    os.makedirs('results/plots', exist_ok=True)
    
    data_path = 'data/processed/IMDB_Dataset_Cleaned.csv'
    train_loader, val_loader, test_loader = load_data(data_path)

    model = FastTextClassifier(embedding_dim)

    train_model(model, train_loader, val_loader, num_epochs=50, patience=5)

if __name__ == "__main__":
    main()

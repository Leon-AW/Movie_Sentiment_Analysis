import torch
import torch.nn as nn
import torch.optim as optim
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Load the pre-trained Word2Vec model
w2v_model = Word2Vec.load("models/word2vec_imdb.model")

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
        vectors = [self.word2vec_model.wv[word] for word in review if word in self.word2vec_model.wv]

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

def train_model(model, train_loader, val_loader, num_epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

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

        # Print average loss per epoch
        epoch_loss = running_loss / len(train_loader)
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
        val_accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

        # Step the scheduler based on validation loss
        scheduler.step(val_loss)

def main():
    data_path = 'data/processed/IMDB_Dataset_Cleaned.csv'
    train_loader, val_loader, test_loader = load_data(data_path)

    model = FastTextClassifier(embedding_dim)

    train_model(model, train_loader, val_loader, num_epochs=15)

    # Save the trained model
    torch.save(model.state_dict(), "models/fasttext_classifier.pth")
    print("Model saved successfully!")

if __name__ == "__main__":
    main()

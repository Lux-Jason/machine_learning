import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Define the tokenizer as a regular function
def tokenizer_function(x, max_seq_length=256):
    return [min(ord(c), 255) for c in x[:max_seq_length]]

class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = torch.tensor(self.tokenizer(self.texts[idx]), dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return text, label

def collate_fn(batch):
    texts, labels = zip(*batch)
    lengths = torch.tensor([len(text) for text in texts])
    texts = pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.float32)
    return texts, lengths, labels

class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(SentimentRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text, text_lengths):
        embedded = self.embedding(text)
        # In train_model and evaluate_model
        packed_embedded = pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False)

        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        dense_outputs = self.fc(hidden[-1])
        return self.sigmoid(dense_outputs)

def train_model(model, train_loader, optimizer, criterion, num_epochs):
    train_losses = []
    train_accuracies = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        for texts, lengths, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", ncols=100):
            texts, lengths, labels = texts.to(device), lengths.to(device), labels.to(device)
            optimizer.zero_grad()
            predictions = model(texts, lengths.cpu()).squeeze()  # Ensure lengths are on CPU
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            predicted = (predictions >= 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        avg_loss = epoch_loss / len(train_loader)
        accuracy = correct / total
        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    return train_losses, train_accuracies

def evaluate_model(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for texts, lengths, labels in tqdm(val_loader, desc="Evaluating", ncols=100):
            texts, lengths, labels = texts.to(device), lengths.to(device), labels.to(device)
            predictions = model(texts, lengths.cpu()).squeeze()  # Ensure lengths are on CPU
            loss = criterion(predictions, labels)
            val_loss += loss.item()
            predicted = (predictions >= 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    avg_loss = val_loss / len(val_loader)
    accuracy = correct / total
    return avg_loss, accuracy


if __name__ == '__main__':  # Add this line to protect the entry point

    # Load datasets
    train_df = pd.read_csv('./kaggle/input/12123334/train.tsv', sep='\t')
    test_df = pd.read_csv('./kaggle/input/12123334/test.tsv', sep='\t')

    train_texts = train_df['text_a'].tolist()
    train_labels = train_df['label'].tolist()

    val_texts = test_df['text_a'].tolist()
    val_labels = test_df['label'].tolist()

    # Tokenizer and other configurations
    max_seq_length = 256
    # Pass the tokenizer function as an argument
    tokenizer = tokenizer_function

    vocab_size = 256
    embedding_dim = 32
    hidden_dim = 64
    output_dim = 1
    num_epochs = 10
    batch_size = 16
    learning_rate = 0.001

    # Prepare datasets and dataloaders
    train_dataset = CustomDataset(train_texts, train_labels, tokenizer)
    val_dataset = CustomDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model, criterion, and optimizer
    model = SentimentRNN(vocab_size, embedding_dim, hidden_dim, output_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_losses, train_accuracies = train_model(model, train_loader, optimizer, criterion, num_epochs)

    # Evaluate the model
    val_loss, val_accuracy = evaluate_model(model, val_loader, criterion)

    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # Plot the results
    plt.figure(figsize=(12, 5))

    # Plot Training Loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Trend")
    plt.legend()

    # Plot Training Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_accuracies, label="Training Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy Trend")
    plt.legend()

    plt.show()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import matplotlib.pyplot as plt

# 自定义数据集
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

# 自定义 collate_fn 函数，用于对齐序列长度
def collate_fn(batch):
    texts, labels = zip(*batch)
    lengths = torch.tensor([len(text) for text in texts])  # 记录序列长度
    texts = pad_sequence(texts, batch_first=True, padding_value=0)  # 填充序列
    labels = torch.tensor(labels, dtype=torch.float32)
    return texts, lengths, labels

# RNN 模型定义
class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(SentimentRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text, text_lengths):
        # 将 text_lengths 转换为 int64 类型，并确保它在 CPU 上
        text_lengths = text_lengths.to(dtype=torch.int64, device='cpu')  # 转到 CPU

        embedded = self.embedding(text)
        packed_embedded = pack_padded_sequence(embedded, text_lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        dense_outputs = self.fc(hidden[-1])  # 使用最后的隐藏状态
        return self.sigmoid(dense_outputs)


# 训练函数
def train_model(model, train_loader, optimizer, criterion, num_epochs):
    train_losses = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for texts, lengths, labels in train_loader:
            texts, lengths, labels = texts.to(device), lengths.to(device), labels.to(device)  # 确保数据在GPU上
            optimizer.zero_grad()
            predictions = model(texts, lengths).squeeze()
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    return train_losses

# 验证函数
def evaluate_model(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for texts, lengths, labels in val_loader:
            texts, lengths, labels = texts.to(device), lengths.to(device), labels.to(device)  # 确保数据在GPU上
            predictions = model(texts, lengths).squeeze()
            loss = criterion(predictions, labels)
            val_loss += loss.item()
            predicted = (predictions >= 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    avg_loss = val_loss / len(val_loader)
    accuracy = correct / total
    return avg_loss, accuracy

# 数据预处理和超参数
texts = ["This is great", "Horrible experience", "Amazing product", "Not good"] * 100  # 示例文本数据
labels = [1, 0, 1, 0] * 100  # 示例标签

tokenizer = lambda x: [ord(c) for c in x]  # 示例分词器，将字符转为 ASCII 编码（需替换为实际分词器）
vocab_size = 128  # 示例词汇表大小，需根据实际数据调整
embedding_dim = 64
hidden_dim = 128
output_dim = 1
num_epochs = 5
batch_size = 32
learning_rate = 0.001

train_texts, val_texts = texts[:300], texts[300:]
train_labels, val_labels = labels[:300], labels[300:]

train_dataset = CustomDataset(train_texts, train_labels, tokenizer)
val_dataset = CustomDataset(val_texts, val_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# 模型、优化器和损失函数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentimentRNN(vocab_size, embedding_dim, hidden_dim, output_dim).to(device)  # 确保模型在GPU上
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练和验证
train_losses = train_model(model, train_loader, optimizer, criterion, num_epochs)
val_loss, val_accuracy = evaluate_model(model, val_loader, criterion)

print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

# 绘制训练损失趋势
plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Trend")
plt.legend()
plt.show()

import pandas as pd
import numpy as np
import re
import nltk
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.corpus import stopwords
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

# LSTM Model
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size, num_layers=1, dropout=0.5):
        super(SentimentLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden[-1])
        return self.sigmoid(out)

# Load Dataset
try:
    amazon_df = pd.read_csv('amazon_cells_labelled.txt', delimiter='\t', header=None, names=['sentence', 'label'])
    imdb_df = pd.read_csv('imdb_labelled.txt', delimiter='\t', header=None, names=['sentence', 'label'])
    yelp_df = pd.read_csv('yelp_labelled.txt', delimiter='\t', header=None, names=['sentence', 'label'])
except FileNotFoundError:
    print("Dataset files not found. Please ensure the dataset files are in the current directory.")
    exit()

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

df = pd.concat([amazon_df, imdb_df, yelp_df], ignore_index=True)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return words

df['tokens'] = df['sentence'].apply(preprocess_text)

all_tokens = [token for tokens in df['tokens'] for token in tokens]
vocab_size = 1000
most_common_tokens = Counter(all_tokens).most_common(vocab_size)
word_to_idx = {word: idx + 1 for idx, (word, _) in enumerate(most_common_tokens)}  # Start from 1 for padding

def vectorize_tokens(tokens, word_to_idx, max_len=20):
    vector = [word_to_idx.get(token, 0) for token in tokens]
    if len(vector) < max_len:
        vector += [0] * (max_len - len(vector))
    else:
        vector = vector[:max_len]
    return vector

max_len = 20
df['vector'] = df['tokens'].apply(lambda x: vectorize_tokens(x, word_to_idx, max_len))

X = np.stack(df['vector'].values)
y = df['label'].values

split_ratio = 0.8
split_index = int(len(df) * split_ratio)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# DataLoader
batch_size = 32
train_data = TensorDataset(torch.tensor(X_train, dtype=torch.long), torch.tensor(y_train, dtype=torch.float32))
test_data = TensorDataset(torch.tensor(X_test, dtype=torch.long), torch.tensor(y_test, dtype=torch.float32))
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)

# Params
embed_size = 64
hidden_size = 128
output_size = 1
num_layers = 2

model = SentimentLSTM(vocab_size, embed_size, hidden_size, output_size, num_layers)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 20
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.view(-1, 1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')

# Eval
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted.view(-1) == labels).sum().item()

accuracy = correct / total
print(f'\nTest Accuracy: {accuracy * 100:.2f}%')

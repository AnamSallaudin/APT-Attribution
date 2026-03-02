import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score

# ----------------------------
# Load Data
# ----------------------------
X_train = np.load("../data/processed/X_train.npy")
X_val   = np.load("../data/processed/X_val.npy")
X_test  = np.load("../data/processed/X_test.npy")

y_train = np.load("../data/processed/y_train.npy")
y_val   = np.load("../data/processed/y_val.npy")
y_test  = np.load("../data/processed/y_test.npy")

# Convert to tensors
X_train = torch.LongTensor(X_train)
y_train = torch.LongTensor(y_train)

X_val = torch.LongTensor(X_val)
y_val = torch.LongTensor(y_val)

X_test = torch.LongTensor(X_test)
y_test = torch.LongTensor(y_test)

# ----------------------------
# DataLoaders
# ----------------------------
train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    TensorDataset(X_val, y_val),
    batch_size=32
)

# ----------------------------
# Model Definition
# ----------------------------
class APT_LSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        final_hidden = out[:, -1, :]
        return self.fc(final_hidden)

vocab_size = int(X_train.max().item())
num_classes = len(np.unique(y_train.numpy()))

model = APT_LSTM(
    vocab_size=vocab_size,
    embed_dim=32,
    hidden_dim=64,
    num_classes=num_classes
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ----------------------------
# Training Loop
# ----------------------------
EPOCHS = 20

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    for xb, yb in train_loader:
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            outputs = model(xb)
            loss = criterion(outputs, yb)
            val_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

# ----------------------------
# Evaluation
# ----------------------------
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    _, preds = torch.max(outputs, 1)

accuracy = accuracy_score(y_test.numpy(), preds.numpy())
print("\nDL Test Accuracy:", accuracy)
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pickle

# -----------------------------
# Load mappings
# -----------------------------
with open("tech2idx.pkl", "rb") as f:
    tech2idx = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# -----------------------------
# Model Definition (must match training)
# -----------------------------
class APT_BiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        final_hidden = out[:, -1, :]
        return self.fc(final_hidden)

# -----------------------------
# Load model
# -----------------------------
vocab_size = len(tech2idx)
num_classes = len(label_encoder.classes_)

model = APT_BiLSTM(vocab_size, 64, 128, num_classes)
model.load_state_dict(torch.load("apt_bilstm.pt", map_location="cpu"))
model.eval()

# -----------------------------
# UI
# -----------------------------
st.title("APT Attribution using ATT&CK Technique Sequences")

st.write("Enter ATT&CK techniques separated by commas")
st.write("Example: T1059, T1105, T1027")

user_input = st.text_input("Technique Sequence")

if st.button("Predict"):
    techniques = [t.strip().upper() for t in user_input.split(",")]

    encoded = [tech2idx.get(t, 0) for t in techniques]

    MAX_LEN = 6
    if len(encoded) < MAX_LEN:
        encoded += [0] * (MAX_LEN - len(encoded))
    else:
        encoded = encoded[:MAX_LEN]

    input_tensor = torch.LongTensor([encoded])

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probabilities, 1)

    predicted_label = label_encoder.inverse_transform(pred.numpy())[0]
    confidence_score = confidence.item()

    st.success(f"Predicted APT Group: {predicted_label}")
    st.info(f"Confidence: {confidence_score:.4f}")

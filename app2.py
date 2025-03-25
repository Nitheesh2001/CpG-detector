import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from functools import partial
from torch.utils.data import Dataset, DataLoader

# Set random seed for reproducibility
def set_seed(seed=13):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(13)

# DNA Sequence Helpers
alphabet = 'NACGT'
dna2int = {a: i for a, i in zip(alphabet, range(5))}
int2dna = {i: a for a, i in zip(alphabet, range(5))}

dnaseq_to_intseq = partial(map, dna2int.get)
intseq_to_dnaseq = partial(map, int2dna.get)

# Function to count CpG sites
def count_cpgs(seq: str) -> int:
    return sum(1 for i in range(len(seq) - 1) if seq[i:i+2] == "CG")

# Dataset class
class MyDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
    
    def __getitem__(self, index):
        return torch.LongTensor(self.sequences[index]), self.labels[index]
    
    def __len__(self):
        return len(self.sequences)

# Model
class CpGPredictor(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, num_layers=2):
        super(CpGPredictor, self).__init__()
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        return self.fc(x).squeeze()

# Initialize model
model = CpGPredictor()

# Streamlit UI
st.title("CpG Predictor using LSTM")
st.write("Enter a DNA sequence to predict the number of CpG sites.")

# User input (Fix input sanitization)
user_input = st.text_input("Enter DNA Sequence (A, C, G, T only, no spaces/commas):")

if user_input:
    # Remove spaces and commas
    user_input = user_input.replace(" ", "").replace(",", "").upper()

    # Validate input
    if any(char not in "ACGT" for char in user_input):
        st.error("Invalid sequence! Use only A, C, G, T (without spaces or commas).")
    else:
        # Convert to integer sequence
        input_seq = torch.LongTensor([dna2int[char] for char in user_input]).unsqueeze(0)

        # Model prediction
        model.eval()
        with torch.no_grad():
            prediction = model(input_seq).item()

        st.success(f"Predicted CpG Count: {prediction:.2f}")

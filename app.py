import streamlit as st
import torch
import torch.nn as nn
import re
import asyncio

# Fix Windows event loop issue
#asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Define model architecture
class CpGPredictorVarLen(nn.Module):
    def __init__(self):
        super(CpGPredictorVarLen, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=6, embedding_dim=16)
        self.lstm = nn.LSTM(input_size=16, hidden_size=64, num_layers=2, batch_first=True)
        self.classifier = nn.Linear(64, 1)

    def forward(self, x, lengths):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        logits = self.classifier(lstm_out[:, -1, :])
        return logits.squeeze()

# Load trained model
model_path = r"CpG detector model/cpg_predictor.pth"
model = CpGPredictorVarLen()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# DNA character mapping
alphabet = "NACGT"
dna2int = {a: i for i, a in enumerate(alphabet, start=1)}
dna2int["pad"] = 0  # Padding token

# Function to validate input DNA sequence
def validate_input(sequence):
    if not sequence or not re.fullmatch(r'[NACGT]+', sequence, re.IGNORECASE):
        return None  # Return None if invalid
    return sequence.upper()  # Convert to uppercase

# Function to predict CpG count
def predict_cpg(model, sequence):
    try:
        int_seq = [dna2int[char] for char in sequence]  # Convert to integers
        input_tensor = torch.LongTensor(int_seq).unsqueeze(0)
        length_tensor = torch.tensor([len(sequence)])
        
        with torch.no_grad():
            prediction = model(input_tensor, length_tensor)
        return round(prediction.item(), 4)  # Return rounded prediction
    except Exception:
        return None

# Streamlit UI
st.title("CpG Count Predictor 🧬")

user_input = st.text_input("Enter a DNA sequence (N, A, C, G, T only):", "")

if st.button("Predict"):
    validated_seq = validate_input(user_input)
    
    if validated_seq:
        prediction = predict_cpg(model, validated_seq)
        if prediction is not None:
            st.success(f"Predicted CpG count: {prediction}")
        else:
            st.error("Error processing input. Ensure it's a valid DNA sequence.")
    else:
        st.warning("❌ Invalid input! Please enter a valid DNA sequence using only 'N, A, C, G, T'.")

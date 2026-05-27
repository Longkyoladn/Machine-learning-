import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

# Markdown cell introducing the notebook
markdown_1 = """# Bitcoin Price Prediction with RNN and Bidirectional RNN (PyTorch)
This notebook reads the Kaggle dataset `btcusd_1-min_data.csv`, preprocesses it by resampling to daily frequencies, and trains an RNN and a Bidirectional RNN model to predict the Close price."""

# Code cell 1: Imports
code_1 = """import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import os

import warnings
warnings.filterwarnings('ignore')"""

# Code cell 2: Data Loading
code_2 = """csv_file = "btcusd_1-min_data.csv"
if not os.path.exists(csv_file):
    print(f"Error: {csv_file} not found.\\nPlease download from Kaggle and place it in the same directory.")
else:
    print("Loading dataset...")
    df = pd.read_csv(csv_file)
    print(df.head())"""

# Code cell 3: Preprocessing
code_3 = """if os.path.exists(csv_file):
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    df.set_index('Timestamp', inplace=True)
    
    df_daily = df.resample('D').mean()
    df_daily.fillna(method='ffill', inplace=True)
    
    data = df_daily[['Close']].values
    print("Data shape after resampling:", data.shape)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
"""

# Code cell 4: Creating sequences
code_4 = """def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

if os.path.exists(csv_file):
    seq_length = 60
    X, y = create_sequences(scaled_data, seq_length)
    
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Reshape for PyTorch [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.Tensor(X_train)
    y_train_tensor = torch.Tensor(y_train).view(-1, 1)
    X_test_tensor = torch.Tensor(X_test)
    y_test_tensor = torch.Tensor(y_test).view(-1, 1)
    
    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=False)
    print("X_train shape:", X_train.shape)"""

# Code cell 5: RNN Model
code_5 = """class SimpleRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

if os.path.exists(csv_file):
    rnn_model = SimpleRNN()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(rnn_model.parameters(), lr=0.001)
    
    print("Training Simple RNN...")
    epochs = 10
    for epoch in range(epochs):
        rnn_model.train()
        epoch_loss = 0
        for seq, labels in train_loader:
            optimizer.zero_grad()
            outputs = rnn_model(seq)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}')"""

# Code cell 6: Bidirectional Model
code_6 = """class BiLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size * 2, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

if os.path.exists(csv_file):
    bi_model = BiLSTM()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(bi_model.parameters(), lr=0.001)
    
    print("Training Bidirectional LSTM...")
    epochs = 10
    for epoch in range(epochs):
        bi_model.train()
        epoch_loss = 0
        for seq, labels in train_loader:
            optimizer.zero_grad()
            outputs = bi_model(seq)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}')"""

# Code cell 7: Evaluation
code_7 = """if os.path.exists(csv_file):
    rnn_model.eval()
    bi_model.eval()
    
    with torch.no_grad():
        rnn_preds = rnn_model(X_test_tensor).numpy()
        bi_preds = bi_model(X_test_tensor).numpy()
    
    # Inverse transform
    rnn_predictions = scaler.inverse_transform(rnn_preds)
    bi_predictions = scaler.inverse_transform(bi_preds)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    plt.figure(figsize=(14,5))
    plt.plot(y_test_inv, color='red', label='Actual Bitcoin Price')
    plt.plot(rnn_predictions, color='blue', label='Simple RNN Predicted Price')
    plt.plot(bi_predictions, color='green', label='Bidirectional LSTM Predicted Price')
    plt.title('Bitcoin Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()"""

# Cell 0: Install dependencies
code_0 = """%pip install torch numpy pandas matplotlib scikit-learn"""

nb.cells = [
    nbf.v4.new_code_cell(code_0),
    nbf.v4.new_markdown_cell(markdown_1),
    nbf.v4.new_code_cell(code_1),
    nbf.v4.new_code_cell(code_2),
    nbf.v4.new_code_cell(code_3),
    nbf.v4.new_code_cell(code_4),
    nbf.v4.new_markdown_cell("## 1. Simple RNN Model (PyTorch)"),
    nbf.v4.new_code_cell(code_5),
    nbf.v4.new_markdown_cell("## 2. Bidirectional RNN Model (PyTorch)"),
    nbf.v4.new_code_cell(code_6),
    nbf.v4.new_markdown_cell("## 3. Evaluation & Comparison"),
    nbf.v4.new_code_cell(code_7)
]

with open('bitcoin_rnn.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

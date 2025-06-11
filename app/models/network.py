import torch
import torch.nn as nn

class LSTMNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        last_h = h_n[-1]
        return self.fc(last_h).unsqueeze(1)

class TrajectoryPredictor(nn.Module):
    """GRU-based trajectory predictor matching the trained models"""
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=3, num_layers=2, dropout=0.5):
        super(TrajectoryPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru1 = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.gru2 = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        out, h_n = self.gru1(x)
        # Decoder generates 10 steps, but we'll take the last one
        dec_input = torch.zeros(x.size(0), 10, self.hidden_dim).to(x.device)
        out, _ = self.gru2(dec_input, h_n)
        out = self.fc(out)  # shape: (batch_size, 10, output_dim)
        return out

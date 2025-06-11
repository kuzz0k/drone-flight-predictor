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

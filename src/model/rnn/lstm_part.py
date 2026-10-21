import torch.nn as nn
import torch.nn.functional as F


class LSTM_part(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional, num_layers=1):
        super(LSTM_part, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.bidirectional = bidirectional

    def forward(self, x):
        N, C, T, V = x.size()
        x = x.permute(0, 3, 2, 1).contiguous()
        x = x.view(N * V, T, C)
        x, _ = self.lstm(x)
        C = x.size(2)
        x = x.view(N, V, T, C)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = x.view(N, C, T, V)

        return x

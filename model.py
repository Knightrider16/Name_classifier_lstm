import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq):
        inputs = torch.stack(input_seq).view(len(input_seq), 1, -1)
        output, (hn, cn) = self.lstm(inputs)
        output = self.fc(hn[-1])
        return F.log_softmax(output, dim=1)

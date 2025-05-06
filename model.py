import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size

        self.W_f = nn.Linear(input_size, hidden_size)
        self.U_f = nn.Linear(hidden_size, hidden_size)
        
        self.W_i = nn.Linear(input_size, hidden_size)
        self.U_i = nn.Linear(hidden_size, hidden_size)
        
        self.W_c = nn.Linear(input_size, hidden_size)
        self.U_c = nn.Linear(hidden_size, hidden_size)
        
        self.W_o = nn.Linear(input_size, hidden_size)
        self.U_o = nn.Linear(hidden_size, hidden_size)
        
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq):
        device = next(self.parameters()).device

        h_t = torch.zeros(1, self.hidden_size, device=device)
        c_t = torch.zeros(1, self.hidden_size, device=device)

        for x_t in input_seq:
            x_t = x_t.view(1, -1).to(device)

            f_t = torch.sigmoid(self.W_f(x_t) + self.U_f(h_t))
            i_t = torch.sigmoid(self.W_i(x_t) + self.U_i(h_t))
            c_hat_t = torch.tanh(self.W_c(x_t) + self.U_c(h_t))
            c_t = f_t * c_t + i_t * c_hat_t
            o_t = torch.sigmoid(self.W_o(x_t) + self.U_o(h_t))
            h_t = o_t * torch.tanh(c_t)

        out = self.fc(h_t)
        return F.log_softmax(out, dim=1)

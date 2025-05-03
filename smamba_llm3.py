import torch
import torch.nn as nn
from smamba3 import SMamba

class MambaLLM(nn.Module):
    def __init__(self, 
                 num_features,
                 seq_len,
                 use_norm,
                 d_model,
                 d_state,
                 d_ff,
                 dropout,
                 activation,
                 e_layers,
                 mamba_output_dim,
                 finbert_output_dim
                 ):
        super(MambaLLM, self).__init__()        
        self.mamba_net = SMamba(num_features=num_features, seq_len=seq_len, use_norm=use_norm, d_model=d_model, d_state=d_state, d_ff=d_ff, dropout=dropout, activation=activation, e_layers=e_layers, output_dim=mamba_output_dim)
        self.finbert_output_dim = finbert_output_dim
        self.lstm = nn.LSTM(input_size=finbert_output_dim, hidden_size=finbert_output_dim, batch_first=True)

        dim1 = 500
        dim2 = 100
        dim3 = 25
        # self.fc1 = nn.Linear(mamba_output_dim+finbert_output_dim, dim1)
        # self.fc2 = nn.Linear(dim1, dim2)
        # self.fc3 = nn.Linear(dim2, dim3)
        self.fc1 = nn.Sequential(
            nn.Linear(mamba_output_dim + finbert_output_dim, dim1),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(dim2, dim3),
            nn.ReLU()
        )        
        self.fc4 = nn.Linear(dim3, 1)
    
    def forward(self, x_enc):
        x1, x1_finbert = torch.split(x_enc, [x_enc.shape[-1]-self.finbert_output_dim, self.finbert_output_dim], dim=-1)
        x2 = x1.permute(0,2,1)
        x2_output = self.mamba_net(x2)
        x1_finbert_compressed = torch.mean(x1_finbert, dim=1)  # [32, 768]
        x_concat = torch.cat((x1_finbert_compressed, x2_output), dim=1)  # [32, 868]
        x_concat1 = self.fc1(x_concat)
        x_concat2 = self.fc2(x_concat1)
        x_concat3 = self.fc3(x_concat2)
        x_concat4 = self.fc4(x_concat3)
        return x_concat4

        
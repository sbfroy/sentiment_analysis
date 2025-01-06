import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size) 
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, attention_mask):
        x = self.embedding(x)  
        x = torch.mul(x, attention_mask.unsqueeze(2)) 
        lstm_out, _ = self.lstm(x)  
        lstm_out = self.layer_norm(lstm_out)  
        lstm_out = self.dropout(lstm_out)
        out = self.fc(lstm_out[:, -1, :])  
        return out
        
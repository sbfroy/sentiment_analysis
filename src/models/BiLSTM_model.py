import torch
import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x, attention_mask):
        x = self.embedding(x)  
        x = torch.mul(x, attention_mask.unsqueeze(2)) 
        lstm_out, _ = self.lstm(x)  
        lstm_out = self.dropout(lstm_out)
        out = self.fc(lstm_out[:, -1, :])  
        return out
        
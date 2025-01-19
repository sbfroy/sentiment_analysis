import torch.nn as nn
from transformers import AutoModel

class TransformerModel(nn.Module):
    def __init__(self, transformer_name, dropout):
        super(TransformerModel, self).__init__()
        self.transformer = AutoModel.from_pretrained(transformer_name)
        self.dropout = nn.Dropout(p=dropout)
        self.regressor = nn.Linear(self.transformer.config.hidden_size, 1) # linear 

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.regressor(cls_output)
        return logits

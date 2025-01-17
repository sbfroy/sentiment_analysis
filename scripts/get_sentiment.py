import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from transformers import AutoTokenizer
from transformers import AutoModel
from src.utils.config_loader import load_config
from src.models.BiLSTM_model import BiLSTM
import pandas as pd 
from pathlib import Path
from tqdm import tqdm

base_dir = Path(__file__).parent.parent

config = load_config(base_dir / 'model_params.yaml')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained('NbAiLab/nb-bert-base')
transformer = AutoModel.from_pretrained('NbAiLab/nb-bert-base')

embed_layer = transformer.get_input_embeddings()
pretrained_embed = embed_layer.weight.detach().cpu().numpy()

model = BiLSTM(
    vocab_size=tokenizer.vocab_size,
    embed_size=pretrained_embed.shape[1],
    hidden_size=config['model']['hidden_size'],
    num_layers=config['model']['num_layers'],
    dropout=config['model']['dropout'],
    pretrained_embed=pretrained_embed
)

model.load_state_dict(torch.load(base_dir / 'src' /'models' / 'BiLSTM_model.pth', weights_only=True)) 
model.eval()

def get_sentiment(text):
    model.eval()

    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=config['data']['max_seq_len'],
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        output = model(input_ids, attention_mask).squeeze()
        
    return output

input = base_dir / 'data' / 'n_a_all_ap.csv' 
output = base_dir / 'data' / 'dates_with_sentiment.csv'

data = pd.read_csv(input)

tqdm.pandas()

data['score'] = data['text'].progress_apply(get_sentiment) # Score column

result_data = data[['date', 'id', 'score']]
result_data.to_csv(output, index=False)

print(f"Done! Results saved in {output}")

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from transformers import AutoTokenizer
from src.models.transformer_model import TransformerModel
from src.utils.config_loader import load_config
import pandas as pd 
from pathlib import Path
from tqdm import tqdm

base_dir = Path(__file__).parent.parent

config = load_config(base_dir / 'transformer_params.yaml')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained('NbAiLab/nb-bert-base')

model = TransformerModel(
    transformer_name='NbAiLab/nb-bert-base',
    dropout=config['model']['dropout']
)

model.load_state_dict(torch.load(base_dir / 'src' /'models' / 'transformer_model.pth', weights_only=True)) 
model.eval()

def get_sentiment(text):

    encoding = tokenizer(
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

input = base_dir / 'data' / 'all_articles_from_norsk_aviskorpus.csv' 
output = base_dir / 'data' / 'all_articles_with_sentiment.csv'

data = pd.read_csv(input)

tqdm.pandas()

data['score'] = data['text'].progress_apply(get_sentiment) # Score column

result_data = data[['id', 'date', 'paper', 'score']]
result_data.to_csv(output, index=False)

print(f"Done! Results saved in {output}")

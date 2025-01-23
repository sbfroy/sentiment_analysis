import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from src.utils.config_loader import load_config
from src.data.preprocessing import create_df
from src.data.dataset import TokenizedDataset
from src.models.transformer_model import TransformerModel
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path
import numpy as np
import yaml

base_dir = Path(__file__).parent.parent

config = load_config(base_dir / 'transformer_params.yaml')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained('NbAiLab/nb-bert-base')

test_df = create_df(base_dir / 'data' / 'test')
test_dataset = TokenizedDataset(test_df, tokenizer, config['data']['max_seq_len'])
test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)

model = TransformerModel(
    transformer_name='NbAiLab/nb-bert-base',
    dropout=config['model']['dropout']
)

model.load_state_dict(torch.load(base_dir / 'src' / 'models' / 'transformer_model.pth', weights_only=True)) 
model.to(device)
model.eval()

test_preds = []
test_labels = []

with torch.no_grad():
    for batch in test_loader:
        inputs = batch['input_ids'].to(device)
        masks = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(inputs, attention_mask=masks).squeeze()

        test_preds.extend(outputs.detach().cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

# Compute metrics
test_rmse = np.sqrt(mean_squared_error(test_labels, test_preds))
test_r2 = r2_score(test_labels, test_preds)

# Print results

print(f"Test RMSE: {test_rmse}")
print(f"Test R^2: {test_r2}")

output_dir = base_dir / 'evaluation_logs'
output_dir.mkdir(parents=True, exist_ok=True)

results = {
    'Test RMSE': test_rmse,
    'Test R^2': test_r2 
}

with open(output_dir / 'transformer_evaluation_results.yaml', 'w') as f:
    yaml.dump(results, f)

print(f'Done! Saved stuff in {output_dir}')

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer
from src.utils.config_loader import load_config
from src.utils.seed import seed_everything
from src.data.dataset import TokenizedDataset
from src.data.preprocessing import create_df
from src.models.lstm_model import LSTM
from src.training.train import train_model
from src.training.early_stopping import EarlyStopping
from pathlib import Path
from datetime import datetime
import pandas as pd
import yaml

base_dir = Path(__file__).parent.parent

config = load_config(base_dir / 'model_params.yaml')

seed_everything(config['general']['seed'])

tokenizer = AutoTokenizer.from_pretrained('NbAiLab/nb-bert-base')

# Load data
train_df = create_df(base_dir / 'data' / 'train')
val_df = create_df(base_dir / 'data' / 'dev')

train_dataset = TokenizedDataset(train_df, tokenizer, config['data']['max_seq_len'])
val_dataset = TokenizedDataset(val_df, tokenizer, config['data']['max_seq_len'])

# Initialize model
model = LSTM(
    vocab_size=config['model']['vocab_size'],
    embed_size=config['model']['embed_size'],
    hidden_size=config['model']['hidden_size'],
    num_layers=config['model']['num_layers'],
    dropout=config['model']['dropout']
)

# Optimizer and loss function
optimizer = optim.Adam(model.parameters(), 
                       lr=config['training']['learning_rate'],
                       weight_decay=config['training']['weight_decay'])

criterion = nn.MSELoss() 

early_stopping = EarlyStopping(patience=config['training']['patience'], 
                               min_delta=config['training']['min_delta'])

history = train_model(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    optimizer=optimizer,
    criterion=criterion,
    batch_size=config['training']['batch_size'],
    num_epochs=config['training']['num_epochs'],
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Create individual folders
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = base_dir / 'logs' / f'run_{timestamp}'
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / 'params.yaml', 'w') as f:
    yaml.dump(config, f)

pd.DataFrame(history).to_csv(output_dir / 'history.csv', index=False)

torch.save(model.state_dict(), output_dir / 'lstm_model.pth')
           
print(f"Training complete! Saved stuff in {output_dir}")
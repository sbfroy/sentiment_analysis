import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer
from transformers import AutoModel
from src.utils.config_loader import load_config
from src.utils.seed import seed_everything
from src.data.dataset import TokenizedDataset
from src.data.preprocessing import create_df
from src.models.lstm_model import LSTM
from src.models.BiLSTM_model import BiLSTM
from src.training.train import train_model
from pathlib import Path
import yaml
import optuna

base_dir = Path(__file__).parent.parent

config = load_config(base_dir / 'model_params.yaml')

seed_everything(config['general']['seed'])

optuna.logging.set_verbosity(optuna.logging.INFO)

tokenizer = AutoTokenizer.from_pretrained('NbAiLab/nb-bert-base')
transformer = AutoModel.from_pretrained('NbAiLab/nb-bert-base')

# Load data
train_df = create_df(base_dir / 'data' / 'train')
val_df = create_df(base_dir / 'data' / 'dev')

embed_layer = transformer.get_input_embeddings()
pretrained_embed = embed_layer.weight.detach().cpu().numpy()

def objective(trial):
    
    # params to tune
    #hidden_size = trial.suggest_int('hidden_size', 128, 320, step=32)
    #num_layers = trial.suggest_int('num_layers', 4, 7)
    dropout = trial.suggest_float('dropout', 0.0, 0.3, step=0.05)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 5e-7, 1e-5, log=True)
    #gradient_clip_val = trial.suggest_float("gradient_clip_val", 0.1, 0.4, log=True)
    #max_seq_len = trial.suggest_int('max_seq_len', 256, 768, step=32) 

    train_dataset = TokenizedDataset(train_df, tokenizer, config['data']['max_seq_len'])
    val_dataset = TokenizedDataset(val_df, tokenizer, config['data']['max_seq_len'])

    model = BiLSTM(
        vocab_size=tokenizer.vocab_size,
        embed_size=pretrained_embed.shape[1],
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        dropout=dropout,
        pretrained_embed=pretrained_embed
    )

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    history = train_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        optimizer=optimizer,
        criterion=criterion,
        batch_size=config['training']['batch_size'],
        num_epochs=config['training']['num_epochs'],
        device='cuda' if torch.cuda.is_available() else 'cpu',
        #gradient_clip_val=gradient_clip_val,
        trial=trial,
        verbose=False
    )

    best_val_rmse = min(history['val_rmse'])
    
    return best_val_rmse

# Optuna study
study = optuna.create_study(
    study_name='the_final_study',
    direction='minimize', 
    sampler=optuna.samplers.TPESampler(seed=config['general']['seed']),
    load_if_exists=True, 
    storage='sqlite:///the_final_study.db'
)

study.optimize(objective, n_trials=50, show_progress_bar=True)

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
from pathlib import Path
import yaml
from optuna.visualization import plot_param_importances
import optuna

base_dir = Path(__file__).parent.parent

config = load_config(base_dir / 'model_params.yaml')

seed_everything(config['general']['seed'])

optuna.logging.set_verbosity(optuna.logging.INFO)

tokenizer = AutoTokenizer.from_pretrained('NbAiLab/nb-bert-base')

# Load data
train_df = create_df(base_dir / 'data' / 'train')
val_df = create_df(base_dir / 'data' / 'dev')

train_dataset = TokenizedDataset(train_df, tokenizer, config['data']['max_seq_len'])
val_dataset = TokenizedDataset(val_df, tokenizer, config['data']['max_seq_len'])

def objective(trial):
    
    # params to tune
    embed_size = trial.suggest_categorical('embed_size', [64, 128, 256])
    hidden_size = trial.suggest_int('hidden_size', 64, 256, step=32)
    num_layers = trial.suggest_int('num_layers', 1, 4)
    dropout = trial.suggest_float('dropout', 0.1, 0.4, step=0.1)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    gradient_clip_val = trial.suggest_float("gradient_clip_val", 0.1, 1.0, log=True)


    model = LSTM(
        vocab_size=config['model']['vocab_size'],
        embed_size=embed_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    history = train_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        optimizer=optimizer,
        criterion=criterion,
        batch_size=batch_size,
        num_epochs=config['training']['num_epochs'],
        device='cuda' if torch.cuda.is_available() else 'cpu',
        gradient_clip_val=gradient_clip_val,
        trial=trial,
        verbose=False
    )

    best_val_rmse = min(history['val_rmse'])
    
    return best_val_rmse

# Optuna study
study = optuna.create_study(
    study_name='my_study',
    direction='minimize', 
    sampler=optuna.samplers.TPESampler(seed=config['general']['seed']),
    load_if_exists=True, 
    storage='sqlite:///my_study.db'
)

study.optimize(objective, n_trials=50, show_progress_bar=True)

pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

print('Study statistics: ')

print(' Number of finished trials: ', len(study.trials))
print(' Number of pruned trials: ', len(pruned_trials))
print(' Number of complete trials: ', len(complete_trials))

print('Best trial: ')
trial = study.best_trial

print(' Value (RMSE): ', trial.value)
print('Params: ')
for key, value in trial.params.items():
    print('    {}: {}'.format(key, value))

with open('best_params.yaml', 'w') as f:
    yaml.dump(study.best_params, f)

importance_plot = plot_param_importances(study)
importance_plot.write_image('param_importances.png')
importance_plot.show()

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.metrics import mean_squared_error
from src.utils.seed import seed_everything
from src.utils.config_loader import load_config
from src.data.preprocessing import create_df
from pathlib import Path
import numpy as np


base_dir = Path(__file__).parent.parent

config = load_config(base_dir / 'model_params.yaml')

seed_everything(config['general']['seed'])

train_df = create_df(base_dir / 'data' / 'train')
val_df = create_df(base_dir / 'data' / 'dev')

train_labels = train_df['score'].values

mean_score = np.mean(train_labels)

baseline_model = [mean_score] * len(val_df)

val_labels = val_df['score'].values
val_predictions = baseline_model(val_df)

rmse = np.sqrt(mean_squared_error(val_labels, val_predictions))

# Got roughly 0.45 RMSE

print(f"Baseline RMSE on Validation Set: {rmse:.4f}")
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader
from src.utils.weighting import get_weight
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
import numpy as np  
import optuna

def evaluate_score_ranges(predictions, targets):
    ranges = [(-1, -0.5), (-0.5, 0), (0, 0.5), (0.5, 1)]
    results = {}
    
    for min, max in ranges:
        indices = [i for i, t in enumerate(targets) if min <= t < max]
        range_preds = [predictions[i] for i in indices]
        range_targets = [targets[i] for i in indices]
        
        if range_targets: 
            rmse = np.sqrt(mean_squared_error(range_targets, range_preds))
            results[f"{min} to {max}"] = rmse
        else:
            results[f"{min} to {max}"] = "No samples in range"
    
    return results

def train_model(model, train_dataset, val_dataset, optimizer, criterion, batch_size, num_epochs, device, 
                trial=None, early_stopping=None, verbose=True):

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model.to(device)

    history = {
        'train_rmse': [], 
        'val_rmse': [],
        'train_r2': [], 
        'val_r2': [],
        #'lr': []
    }

    for epoch in range(num_epochs):

        model.train()  ### training mode ###

        train_preds = []
        train_targets = []

        for batch in tqdm(train_loader, desc='train', leave=False, ncols=75, disable=not verbose):
            inputs = batch['input_ids'].to(device)
            masks = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            outputs = model(inputs, attention_mask=masks).squeeze()

            weights = torch.tensor(
                [get_weight(label.item()) for label in labels],
                dtype=torch.float
            ).to(device)

            loss = criterion(outputs, labels.to(device))
            weighted_loss = (loss * weights).mean() 
            weighted_loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
            optimizer.step()

            train_preds.extend(outputs.detach().cpu().numpy())  
            train_targets.extend(labels.cpu().numpy())

        train_rmse = np.sqrt(mean_squared_error(train_targets, train_preds))  
        train_r2 = r2_score(train_targets, train_preds)

        model.eval()  ### evaluation mode ###

        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc='val', leave=False, ncols=75, disable=not verbose):
                inputs = batch['input_ids'].to(device)
                masks = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(inputs, attention_mask=masks).squeeze()
                loss = criterion(outputs, labels.to(device))

                val_preds.extend(outputs.detach().cpu().numpy()) 
                val_targets.extend(labels.cpu().numpy())

        val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
        val_r2 = r2_score(val_targets, val_preds)

        #scheduler.step(val_rmse)

        #current_lr = optimizer.param_groups[0]['lr']

        # Logging
        history['train_rmse'].append(train_rmse)
        history['train_r2'].append(train_r2)
        history['val_rmse'].append(val_rmse)
        history['val_r2'].append(val_r2)
        #history['lr'].append(current_lr)

        if verbose:
            print(f'Epoch {epoch+1}/{num_epochs} | '
                  f'Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f} | '
                  f'Train R2: {train_r2:.4f}, Val R2: {val_r2:.4f}')
        
            range_results = evaluate_score_ranges(val_preds, val_targets)
            
            print("Validation RMSE by score range:")
            for score_range, rmse in range_results.items():
                print(f"  {score_range}: {rmse}")

        # Optuna
        if trial:
            trial.report(val_rmse, step=epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        if early_stopping:
            early_stopping(val_rmse)
            if early_stopping.stop_training:
                print("Early stopping!")
                break

    return history

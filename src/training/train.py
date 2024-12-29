import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
import numpy as np  
import optuna

def train_model(model, train_dataset, val_dataset, optimizer, criterion, batch_size, num_epochs, device, trial=None, early_stopping=None):

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model.to(device)

    history = {'train_rmse': [], 'val_rmse': [],
               'train_r2': [], 'val_r2': []}

    for epoch in range(num_epochs):

        model.train()  ### training mode ###

        train_preds = []
        train_targets = []

        for batch in tqdm(train_loader, leave=False, ncols=100):
            inputs = batch['input_ids'].to(device)
            masks = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            outputs = model(inputs, masks).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_preds.extend(outputs.detach().cpu().numpy())  
            train_targets.extend(labels.cpu().numpy())

        train_rmse = np.sqrt(mean_squared_error(train_targets, train_preds))  
        train_r2 = r2_score(train_targets, train_preds)

        model.eval()  ### evaluation mode ###

        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in tqdm(val_loader, leave=False, ncols=100):
                inputs = batch['input_ids'].to(device)
                masks = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(inputs, masks).squeeze()
                loss = criterion(outputs, labels)

                val_preds.extend(outputs.detach().cpu().numpy()) 
                val_targets.extend(labels.cpu().numpy())

        val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
        val_r2 = r2_score(val_targets, val_preds)

        # Logging
        history['train_rmse'].append(train_rmse)
        history['train_r2'].append(train_r2)
        history['val_rmse'].append(val_rmse)
        history['val_r2'].append(val_r2)

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train RMSE: {train_rmse:.4f}, Train R^2: {train_r2:.4f} | "
            f"Val RMSE: {val_rmse:.4f}, Val R^2: {val_r2:.4f}"
            )

        # Optuna
        if trial:
            trial.report(val_rmse, step=epoch)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        if early_stopping:
            early_stopping(val_rmse)
            if early_stopping.stop_training:
                print("Early stopping triggered!")
                break

    return history
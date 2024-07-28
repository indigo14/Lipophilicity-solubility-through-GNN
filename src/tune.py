import logging
import optuna
import torch
from torch_geometric.data import DataLoader
from torch_geometric.nn import AttentiveFP
from src.runner import Runner
from sklearn.metrics import r2_score
from typing import Dict

logger = logging.getLogger(__name__)

def objective(trial, train_loader, test_loader):
    # Define the hyperparameters to tune
    hidden_channels = trial.suggest_int('hidden_channels', 16, 128)
    num_layers = trial.suggest_int('num_layers', 2, 6)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-3)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD'])

    # Log the hyperparameters
    logger.info(f"Trial {trial.number}: hidden_channels={hidden_channels}, num_layers={num_layers}, dropout={dropout}, lr={lr}, weight_decay={weight_decay}, optimizer={optimizer_name}")
    
    # Define model parameters
    model_params = {
        'in_channels': 9,
        'hidden_channels': hidden_channels,
        'out_channels': 1,
        'edge_dim': 3,
        'num_layers': num_layers,
        'num_timesteps': 2,
        'dropout': dropout
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AttentiveFP(**model_params).to(device)

    # Select the optimizer
    optimizer = None
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_runner = Runner(train_loader, model, device, optimizer)
    test_runner = Runner(test_loader, model, device)
    
    # Train the model
    train_runner.fit(train_loader, test_loader, epochs=50)
    
    # Evaluate the model
    test_rmse = test_runner.test(test_loader)
    
    # Log the result
    logger.info(f"Trial {trial.number} Test RMSE: {test_rmse:.4f}")
    
    return test_rmse

def tune_hyperparameters(train_loader: DataLoader, test_loader: DataLoader) -> Dict:
    def wrapped_objective(trial):
        return objective(trial, train_loader, test_loader)
    
    # Create a study object and optimize the objective function
    study = optuna.create_study(direction='minimize')
    study.optimize(wrapped_objective, n_trials=400)
    
    # Log the best hyperparameters
    logger.info(f"Best hyperparameters: {study.best_params}")
    
    # Visualize the optimization history
    optuna.visualization.plot_optimization_history(study).show()
    
    # Visualize the hyperparameter importances
    optuna.visualization.plot_param_importances(study).show()
    
    return study.best_params

if __name__ == "__main__":
    # Assuming train_loader and test_loader are defined globally
    best_params = tune_hyperparameters(train_loader, test_loader)
    logger.info(f"Best parameters: {best_params}")

import pandas as pd
import logging
from src.dataset import MyOwnDataset, split_data, create_dataloaders
from src.runner import Runner
from src.eval import eval
from src.tune import tune_hyperparameters
from torch_geometric.nn import AttentiveFP
import torch
from torch_geometric.data import DataLoader

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("program.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Path to your CSV file
DATASET_PATH = './data/processed/df_final.csv'

# Read the CSV into a DataFrame
df_final = pd.read_csv(DATASET_PATH)

# Ensure df_final is correctly populated
print(df_final.head())

# Instantiate the dataset with the DataFrame and generatee
dataset = MyOwnDataset(root='.', df_final=df_final)

# Get the graph_list
graph_list = dataset.get_graph_list()

# Split the graph_list into training and testing sets
train_subset, test_subset = split_data(graph_list)

# Create DataLoaders for the subsets using torch_geometric's DataLoader
train_loader = DataLoader(train_subset, batch_size=96, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=96, shuffle=False)

# Tune hyperparameters using Optuna
best_params = tune_hyperparameters(train_loader, test_loader)

# Define model parameters with the best hyperparameters
model_params = {
    'in_channels': 9,  # Size of each input sample
    'hidden_channels': best_params['hidden_channels'],  # Hidden node feature dimensionality
    'out_channels': 1,  # Size of each output sample
    'edge_dim': 3,  # Edge feature dimensionality
    'num_layers': best_params['num_layers'],  # Number of GNN layers
    'num_timesteps': 2,  # Number of iterative refinement steps for global readout
    'dropout': best_params['dropout']  # Dropout probability
}

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AttentiveFP(**model_params).to(device)
optimizer_name = best_params['optimizer']
lr = best_params['lr']
weight_decay = best_params['weight_decay']

# Select the optimizer based on the best hyperparameters
optimizer = None
if optimizer_name == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
elif optimizer_name == 'RMSprop':
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
elif optimizer_name == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

# Create the runners
train_runner = Runner(train_loader, model, device, optimizer)
test_runner = Runner(test_loader, model, device)

# Training loop
train_runner.fit(train_loader, test_loader, epochs=75)

# Evaluate the model
results = eval(test_loader, model, device)
print(results)
results = eval(train_loader, model, device)
print(results)






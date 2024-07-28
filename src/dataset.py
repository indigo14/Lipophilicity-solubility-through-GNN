from pathlib import Path
from typing import Any, Tuple
import numpy as np
import random
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import from_smiles
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Subset
from src.load_data import load_split_csv_data


class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, df_final, transform=None, pre_transform=None, pre_filter=None):
        self.df_final = df_final  # Store the DataFrame as an instance variable
        super().__init__(root, transform, pre_transform, pre_filter)
        self.graph_list = []  # becomes the list that is split into train and test
        
        self.load(self.processed_paths[0])
        self.process()  # Ensure process is called to populate graph_list
        # For PyG<2.4:
        # self.data, self.slices = torch.load(self.processed_paths[0])
        # Code from https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html
    @property
    def raw_file_names(self):
        return 'Lipophilicity.csv'

    @property
    def processed_file_names(self):
        return 'data.dt'

    def download(self):
        # Download to `self.raw_dir`.
        # download_url(url, self.raw_dir)
        pass

    def process(self):
        #print("Processing dataset...")
        #print(f"Number of smiles: {len(self.df_final['smiles'])}")
        for i, smile in enumerate(self.df_final['smiles']):
            #print(f"Processing smile {i}: {smile}")
            g = from_smiles(smile)
            g.x = g.x.float()
            y = torch.tensor(self.df_final['exp'][i], dtype=torch.float).view(1, -1)
            g.y = y
            self.graph_list.append(g)
        print(f"Processed {len(self.graph_list)} graphs")



        data_list = self.graph_list

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])
        
    def get_graph_list(self):
        return self.graph_list


def split_data(graph_list):
    random.seed(42)  # For reproducibility
    random.shuffle(graph_list)
    split_idx = int(len(graph_list) * 0.8)
    
    # Create Subsets
    train_subset = Subset(graph_list, list(range(split_idx)))
    test_subset = Subset(graph_list, list(range(split_idx, len(graph_list))))

    return train_subset, test_subset

def create_dataloaders(train_subset, test_subset, batch_size=96):
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader  


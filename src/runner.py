import os
import logging
from typing import Optional
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader as GeoDataLoader
from torch.optim import Optimizer
from torch import device as torch_device

class Runner:
    def __init__(
        self,
        loader: GeoDataLoader,
        model: torch.nn.Module,
        device: torch_device,
        optimizer: Optional[Optimizer] = None
    ) -> None:
        """
        Initializes the Runner.

        Parameters:
        loader (GeoDataLoader): The data loader for fetching batches of data.
        model (torch.nn.Module): The neural network model to train or evaluate.
        device (torch_device): The device (CPU or GPU) to perform computations on.
        optimizer (Optional[Optimizer]): The optimizer for training the model. If None, the runner is in evaluation mode.
        """
        self.device = device
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loader = loader
        self.batch_size = 96
        self.epochs = 1
        self.score_train = []
        self.score_test = []

        # Assume training stage based on presence of optimizer
        self.is_train = optimizer is not None

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing Runner")

        self._prepare_data()

    def _prepare_data(self):
        self.logger.info("Preparing data")
        # Your data preparation logic here
        self.logger.info("Data preparation complete")

    def train(self):
        self.model.train()
        total_loss = total_samples = 0

        for data in self.loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = F.mse_loss(out, data.y)
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss) * data.num_graphs
            total_samples += data.num_graphs

        rmse = (total_loss / total_samples) ** 0.5
        #self.logger.info(f"Training RMSE: {rmse:.4f}")
        return rmse

    @torch.no_grad()
    def test(self, loader: GeoDataLoader):
        self.model.eval()
        mse = []

        for data in loader:
            data = data.to(self.device)
            out = self.model(data.x, data.edge_index, data.edge_attr, data.batch)
            l = F.mse_loss(out, data.y, reduction='none').cpu()
            mse.append(l)

        rmse = float(torch.cat(mse, dim=0).mean().sqrt())
        #self.logger.info(f"Test RMSE: {rmse:.4f}")
        return rmse

    def fit(self, train_loader: GeoDataLoader, test_loader: GeoDataLoader, epochs: int):
        self.epochs = epochs
        for epoch in range(epochs):
            train_rmse = self.train()
            test_rmse = self.test(test_loader)
            self.score_train.append(train_rmse)
            self.score_test.append(test_rmse)
            #self.logger.info(f'Epoch: {epoch:03d}, Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}')
        self.plot()

    def plot(self, output_dir='plots', filename_prefix='plot'):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        plt.figure()
        plt.plot(range(self.epochs), self.score_train, color='goldenrod')
        plt.plot(range(self.epochs), self.score_test, color='blue')
        plt.xlabel('Epochs')
        plt.ylabel('RMSE')
        plt.legend(['train', 'test'])

        plot_filename = os.path.join(output_dir, f'{filename_prefix}_epochs_{self.epochs}.png')
        plt.savefig(plot_filename)
        plt.close()
        self.logger.info(f"Saved plot to {plot_filename}")


''' Some code from https://github.com/pyg-team/pytorch_geometric/blob/master/examples/attentive_fp.py'''
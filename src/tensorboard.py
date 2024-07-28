from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter

from src.tracking import Stage
from src.utils import create_experiment_log_dir

class TensorboardExperiment:
    def __init__(self, log_path: str, create: bool = True) -> None:
        """
        Initializes the TensorboardExperiment.

        Parameters:
        log_path (str): The root directory where logs will be stored.
        create (bool): Whether to create the log directory if it does not exist.
        """
        log_dir = create_experiment_log_dir(root=log_path)
        self.stage = Stage.TRAIN
        self._validate_log_dir(log_dir, create=create)
        self._writer = SummaryWriter(log_dir=log_dir)
        plt.ioff()

    def set_stage(self, stage: Stage) -> None:
        """
        Sets the current stage of the experiment.

        Parameters:
        stage (Stage): The current stage (TRAIN, VAL, TEST).
        """
        self.stage = stage

    def flush(self) -> None:
        """Flushes the writer's buffer."""
        self._writer.flush()

    @staticmethod
    def _validate_log_dir(log_dir: str, create: bool = True) -> None:
        """
        Validates the log directory.

        Parameters:
        log_dir (str): The directory to validate.
        create (bool): Whether to create the directory if it does not exist.

        Raises:
        NotADirectoryError: If the directory does not exist and `create` is False.
        """
        log_path = Path(log_dir).resolve()
        if log_path.exists():
            return
        elif not log_path.exists() and create:
            log_path.mkdir(parents=True)
        else:
            raise NotADirectoryError(f"log_dir {log_dir} does not exist.")

    def add_batch_metric(self, name: str, value: float, step: int) -> None:
        """
        Logs a batch-level metric.

        Parameters:
        name (str): The name of the metric.
        value (float): The value of the metric.
        step (int): The step (batch number).
        """
        tag = f"{self.stage.name}/batch/{name}"
        self._writer.add_scalar(tag, value, step)

    def add_epoch_metric(self, name: str, value: float, step: int) -> None:
        """
        Logs an epoch-level metric.

        Parameters:
        name (str): The name of the metric.
        value (float): The value of the metric.
        step (int): The step (epoch number).
        """
        tag = f"{self.stage.name}/epoch/{name}"
        self._writer.add_scalar(tag, value, step)

    def add_epoch_scatter_plot(
        self, y_true: list[np.array], y_pred: list[np.array], step: int
    ) -> None:
        """
        Logs a scatter plot of true vs. predicted values at the epoch level.

        Parameters:
        y_true (list[np.array]): List of true labels.
        y_pred (list[np.array]): List of predicted labels.
        step (int): The step (epoch number).
        """
        y_true, y_pred = self.collapse_batches(y_true, y_pred)
        fig = self.create_scatter_plot(y_true, y_pred, step)
        tag = f"{self.stage.name}/epoch/scatter_plot"
        self._writer.add_figure(tag, fig, step)

    @staticmethod
    def collapse_batches(
        y_true: list[np.array], y_pred: list[np.array]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Collapses multiple batches into a single array.

        Parameters:
        y_true (list[np.array]): List of true labels.
        y_pred (list[np.array]): List of predicted labels.

        Returns:
        tuple[np.ndarray, np.ndarray]: Collapsed true and predicted labels.
        """
        return np.concatenate(y_true), np.concatenate(y_pred)

    def create_scatter_plot(
        self, y_true: list[np.array], y_pred: list[np.array], step: int
    ) -> plt.Figure:
        """
        Creates a scatter plot of true vs. predicted values.

        Parameters:
        y_true (list[np.array]): True labels.
        y_pred (list[np.array]): Predicted labels.
        step (int): The step (epoch number).

        Returns:
        plt.Figure: The scatter plot figure.
        """
        fig, ax = plt.subplots()
        ax.scatter(y_true, y_pred, alpha=0.3)
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predictions')
        ax.set_title(f"{self.stage.name} Epoch: {step}")
        return fig

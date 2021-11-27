from typing import Any, Dict, List, Sequence, Tuple, Union

import pytorch_lightning as pl
import torch
from torch.optim import Optimizer
from torchmetrics import MetricCollection, MeanAbsoluteError, MeanSquaredError
from src.utils.model_utils import heatmaps_gaussian_1D
from .metrics.submission_metric import SubmissionMetric
from .losses.divergence_loss import DivergenceJS1D, DivergenceKL1D
from ..utils.data_utils import Scale, Hand

class RegularizedModel(pl.LightningModule):
    """ 
        This model is used with architectures that directly estimate the 
        output points with regression, i.e. outputs of size (B, 21, 3) 
        and applies regularization to the output 1D heatmaps to force them
        to be gaussian distributions.
    """
    def __init__(
        self,
        loss: torch.nn.Module,
        regularization: torch.nn.Module,
        architecture: torch.nn.Module,
        heatmap_size: int,
        heatmap_std: float,
        lr: float,
        use_scheduler: bool = False,
        use_normalize: bool = True,
        dir_preds_test: str = "path",
    ) -> None:
        super().__init__()

        self.model = architecture
        self.loss = loss
        self.regularization = regularization
        self.lr = lr
        self.use_normalize = use_normalize
        self.use_scheduler = use_scheduler
        self.dir_preds_test = dir_preds_test
        self.heatmap_size = heatmap_size 
        self.heatmap_std = heatmap_std
        self.save_hyperparameters()

        self.train_metrics = MetricCollection([
            MeanSquaredError(),
            MeanAbsoluteError()
        ], prefix='train')
        self.train_submission_metric = SubmissionMetric()

        self.valid_metrics = MetricCollection([
            MeanSquaredError(), 
            MeanAbsoluteError()
        ], prefix='valid')
        self.valid_submission_metric = SubmissionMetric()

        self.test_predictions = []

    def predict(self, x_image: torch.Tensor) -> torch.Tensor:
        y_hm_points_pred, y_heatmaps_pred = self.model.forward_with_heatmaps(x_image)
        y_points_pred = Scale.linear(
            y_hm_points_pred, 
            domain = Scale.HM(self.heatmap_size), 
            range = Scale.KPN if self.use_normalize else Scale.KP 
        )
        return y_points_pred, y_heatmaps_pred

    def step(self, batch: List[torch.Tensor]) -> tuple:
        x_image, x_camera, y_points, *other = batch

        y_points_pred, y_heatmaps_pred = self.predict(x_image)
        y_points_normalized = Hand.normalize_points(y_points) if self.use_normalize else y_points
        
        loss = self.loss(y_points_pred.float(), y_points_normalized.float())
        # -- Regularization with gaussian heatmaps
        y_hm_points = Scale.linear(
            y_points_normalized, 
            domain=Scale.KPN if self.use_normalize else Scale.KP, 
            range=Scale.HM(self.heatmap_size)
        ) 
        y_heatmaps = heatmaps_gaussian_1D(size = self.heatmap_size, means = y_hm_points, std = self.heatmap_std)
        y_heatmaps = y_heatmaps.view(-1, self.heatmap_size).softmax(dim = 1)
        loss += self.regularization(y_heatmaps_pred.float(), y_heatmaps.float())
        # -- 
        return loss, y_points_pred, y_points, y_points_normalized

    def training_step(
        self, batch: List[torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        loss, y_points_pred, y_points, y_points_normalized = self.step(batch)
        # Loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        # Mertics 
        self.train_metrics(y_points_pred, y_points_normalized)
        self.log_dict(self.train_metrics, on_step=True, on_epoch=True)
        self.train_submission_metric(y_points_pred, y_points)
        self.log("train_submission_metric", self.train_submission_metric, on_step=True, on_epoch=True, prog_bar=False)
        return { "loss": loss }

    def validation_step(
        self, batch: List[torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        loss, y_points_pred, y_points, y_points_normalized = self.step(batch)
        # Loss
        self.log("valid_loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        # Metrics 
        self.valid_metrics(y_points_pred, y_points_normalized)
        self.log_dict(self.valid_metrics, on_step=True, on_epoch=True)
        self.valid_submission_metric(y_points_pred, y_points)
        self.log("valid_submission_metric", self.valid_submission_metric, on_step=True, on_epoch=True, prog_bar=False)
        return { "loss": loss }

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, torch.Tensor]:
        x_image, x_camera = batch
        y_points_pred, *other = self.predict(x_image)
        self.test_predictions.append(y_points_pred.detach().cpu().numpy())
        return {}

    def on_test_end(self) -> None:
        # TODO: generate submission file
        pass  
        
    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Tuple[Sequence[Optimizer], Sequence[Any]]]:
        parameters = list(self.parameters())
        optimizer = torch.optim.Adam(
            list(filter(lambda p: p.requires_grad, parameters)), lr=self.lr
        )

        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=[lambda epoch: 1.0] # Disabled 
            ),
            "name": "lr_scheduler",
            "interval": "epoch",
        }

        return [optimizer], [lr_scheduler]

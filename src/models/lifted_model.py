from typing import Any, Dict, List, Sequence, Tuple, Union

import pytorch_lightning as pl
import torch
from torch.optim import Optimizer
from torchmetrics import MetricCollection, MeanAbsoluteError, MeanSquaredError
import src.utils.model_utils as utils
from .metrics.submission_metric import SubmissionMetric
from ..utils.data_utils import Scale, Hand

class LiftedModel(pl.LightningModule):
    """ 
        This model is used with architectures that estimates first the 2d points 
        then raise them to 3d points, a loss function is applied at both steps. 
    """
    def __init__(
       self,
        loss: torch.nn.Module,
        architecture: torch.nn.Module,
        lr: float,
        use_scheduler: bool = False,
        use_normalize: bool = True,
        dir_preds_test: str = "path"
    ) -> None:
        super().__init__()
        self.model = architecture
        self.loss = loss
        self.lr = lr
        self.use_scheduler = use_scheduler
        self.use_normalize = use_normalize
        self.dir_preds_test = dir_preds_test
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
        y_2d_points_pred, y_points_pred = self.model(x_image)
        return y_points_pred, y_2d_points_pred

    def step(self, batch: List[torch.Tensor]) -> tuple:
        x_image, x_camera, y_points, *other = batch

        y_points_pred, y_2d_points_pred = self.predict(x_image)
        y_points_normalized = Hand.normalize_points(y_points) if self.use_normalize else y_points 
        #y_2d_points = Hand.to_2d_points(y_points_normalized, x_camera)
        # 3D points loss
        loss = self.loss(y_points_pred.float(), y_points_normalized.float())
        # 2D points loss
        #loss += self.loss(y_2d_points_pred.float(), y_2d_points.float())

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


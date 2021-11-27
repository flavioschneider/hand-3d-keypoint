from typing import Any, Dict, List, Sequence, Tuple, Union

import pytorch_lightning as pl
import torch
import os 
import json 
import subprocess
import numpy as np 
from torch.optim import Optimizer
from torchmetrics import MetricCollection, MeanAbsoluteError, MeanSquaredError
import src.utils.model_utils as utils
from .metrics.submission_metric import SubmissionMetric
from src.utils.model_utils import heatmaps_gaussian_1D, rotate_points, rotate_image, rotate_mirrored_points, rotate_mirrored_image
from .losses.divergence_loss import DivergenceJS1D, DivergenceKL1D
from ..utils.data_utils import Scale, Hand

class DefaultModel(pl.LightningModule):
    """ 
        This model is used with architectures that directly estimate the 
        output points with regression, i.e. outputs of size (B, 21, 3).
    """
    def __init__(
       self,
        loss: torch.nn.Module,
        architecture: torch.nn.Module,
        heatmap_size: int,
        lr: float,
        use_scheduler: bool = False,
        use_normalize: bool = True,
        dir_preds_test: str = "path",
        robust_prediction: bool = False,
        robust_prediction_mirror: bool = False, 
        num_rotations: int = 8,
    ) -> None:
        super().__init__()
        self.model = architecture
        self.loss = loss
        self.lr = lr
        self.use_scheduler = use_scheduler
        self.use_normalize = use_normalize
        self.dir_preds_test = dir_preds_test
        self.heatmap_size = heatmap_size 
        self.robust_prediction = robust_prediction
        self.robust_prediction_mirror = robust_prediction_mirror
        self.num_rotations = num_rotations
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

        if self.robust_prediction:
            preds_rot = torch.tensor([])
            pred_rot_mirr = torch.tensor([])
            rotation_degrees = [i*360/self.num_rotations for i in range(self.num_rotations)]

            # Rotations 
            for degrees in rotation_degrees:
                img_rotated = rotate_image(x_image, degrees)
                y_points_pred, *other = self.predict(img_rotated) 
                points_aligned = torch.cat([rotate_points(points.squeeze().detach().cpu(), -degrees).unsqueeze(0).float() for points in y_points_pred]) # ()
                preds_rot = torch.cat((preds_rot, points_aligned.unsqueeze(0).detach()))
            
            if self.robust_prediction_mirror:
                # Mirrored Rotations 
                for degrees in rotation_degrees:
                    img_rotated = rotate_mirrored_image(x_image, degrees)
                    y_points_pred, *other = self.predict(img_rotated)
                    points_aligned = torch.cat([rotate_mirrored_points(points.squeeze().detach().cpu(), degrees).unsqueeze(0).float() for points in y_points_pred])
                    pred_rot_mirr = torch.cat((pred_rot_mirr, points_aligned.unsqueeze(0).detach()))
                test_preds_rot_mirr = torch.mean(pred_rot_mirr, 0)
                test_preds_rot = torch.mean(preds_rot, 0)
                # Mean between rotations and mirrored rotations 
                test_preds = torch.mean(torch.cat((test_preds_rot.unsqueeze(0), test_preds_rot_mirr.unsqueeze(0))), dim=0)
            else:
                test_preds = torch.mean(preds_rot, 0)
                
            self.test_predictions.append(test_preds.detach().cpu().numpy())
        else:
            y_points_pred, *other = self.predict(x_image)
            self.test_predictions.append(y_points_pred.detach().cpu().numpy())
        return {}

    def on_test_end(self) -> None:
        path = self.dir_preds_test
        test_file = os.path.join(path, "test_preds.json")
        os.makedirs(path, exist_ok=True)
        y_points_pred = np.concatenate(self.test_predictions).tolist()
        with open(test_file, "w") as f:
            json.dump(y_points_pred, f)
        subprocess.call(['gzip', test_file])
        
    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Tuple[Sequence[Optimizer], Sequence[Any]]]:
        parameters = list(self.parameters())
        optimizer = torch.optim.Adam(
            list(filter(lambda p: p.requires_grad, parameters)), lr=self.lr
        )

        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[100, 140], gamma=0.1
            ),
            "name": "lr_scheduler",
            "interval": "epoch",
        }

        return [optimizer], [lr_scheduler] if self.use_scheduler else []


from typing import Any, Dict, List, Sequence, Tuple, Union

import pytorch_lightning as pl
import torch
from torch.optim import Optimizer
from torchmetrics import MetricCollection, MeanAbsoluteError, MeanSquaredError
import src.utils.model_utils as utils
from .metrics.submission_metric import SubmissionMetric
from ..utils.data_utils import Scale, Hand, Heatmap
from .architectures.unet import UNet 
from .architectures.graphnets import GraphNet
from .architectures.functional import soft_argmax2d
from .architectures.decoders.featpose_decoder import FeatPoseDecoder

class HMLiftedModel(pl.LightningModule):
    """ 
        This model is used with architectures that estimates first the 2d points 
        then raise them to 3d points, a loss function is applied at both steps. 
    """
    def __init__(
       self,
        loss: torch.nn.Module,
        lr: float,
        lambdas: List[float], 
        use_scheduler: bool = False,
        use_normalize: bool = True,
        dir_preds_test: str = "path"
    ) -> None:
        super().__init__()
        self.loss = loss
        self.lr = lr
        self.lambdas = lambdas
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

        # Model 
        self.img_size = 224
        self.heatmap_size = 112 
        self.heatmap_std = 3.0 
        self.model_img_to_hm = UNet(
            in_channels=3,
            out_channels=21+21, 
            encoder_name='resnet34',
            decoder_channels=[256, 128, 64, 32]
        )

        self.model_feat_to_3d = FeatPoseDecoder(
            in_channels = 21+21,
            out_channels = 21,
            mid_channels = 128,
            mid_blocks = 4,
            heatmap_size = self.heatmap_size
        )

    def predict_img_to_points_2d(self, x_image: torch.Tensor) -> torch.Tensor:
        features = self.model_img_to_hm(x_image)
        heatmaps = features[:,0:21]
        points_2d_heatmap = soft_argmax2d(heatmaps, return_xy=True)
        points_2d = Scale.linear(points_2d_heatmap, domain=Scale.P2D(self.heatmap_size), range=Scale.P2D(self.img_size))
        return points_2d, heatmaps, features

    def predict_features_to_points_3d(self, features: torch.Tensor) -> torch.Tensor:
        points_3d, points_3d_heatmaps_1d = self.model_feat_to_3d(features)
        points_3d = Scale.linear(points_3d, domain=Scale.P3D(self.heatmap_size), range=Scale.KPN if self.use_normalize else Scale.KP)
        return points_3d, points_3d_heatmaps_1d

    def predict(self, x_image: torch.Tensor) -> torch.Tensor:
        points_2d, points_2d_heatmaps_2d, features = self.predict_img_to_points_2d(x_image)
        points_3d, points_3d_heatmaps_1d = self.predict_features_to_points_3d(features)
        return points_3d, points_2d, points_2d_heatmaps_2d, points_3d_heatmaps_1d

    def step(self, batch: List[torch.Tensor]) -> tuple:
        x_image, x_camera, y_points_3d, *other = batch

        y_points_3d_pred, y_points_2d_pred, heatmaps_pred, *other = self.predict(x_image)
        y_points_2d = Hand.to_2d_points(y_points_3d, x_camera)
        heatmaps = Heatmap.make_gaussians(
            means=Scale.linear(y_points_2d, domain=Scale.P2D(self.img_size), range=Scale.P2D(self.heatmap_size)),
            size = self.heatmap_size, 
            sigma = self.heatmap_std 
        )
        y_points_3d_normalized = Hand.normalize_points(y_points_3d) if self.use_normalize else y_points_3d 
        # Heatmap loss
        loss = self.loss(heatmaps_pred.float(), heatmaps.float()) * self.lambdas[0]
        # Points 2d loss
        loss += self.loss(y_points_2d_pred.float(), y_points_2d.float()) * self.lambdas[1]
        # Points 3d loss 
        loss += self.loss(y_points_3d_pred.float(), y_points_3d_normalized.float()) * self.lambdas[2]
        return loss, y_points_3d, y_points_3d_normalized, y_points_3d_pred 

    def training_step(
        self, batch: List[torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        loss, y_points_3d, y_points_3d_normalized, y_points_3d_pred  = self.step(batch)
        # Loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        # Mertics 
        self.train_metrics(y_points_3d_pred, y_points_3d_normalized)
        self.log_dict(self.train_metrics, on_step=True, on_epoch=True)
        self.train_submission_metric(y_points_3d_pred, y_points_3d)
        self.log("train_submission_metric", self.train_submission_metric, on_step=True, on_epoch=True, prog_bar=False)
        return { "loss": loss }

    def validation_step(
        self, batch: List[torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        loss, y_points_3d, y_points_3d_normalized, y_points_3d_pred  = self.step(batch)
        # Loss
        self.log("valid_loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        # Metrics 
        self.valid_metrics(y_points_3d_pred, y_points_3d_normalized)
        self.log_dict(self.valid_metrics, on_step=True, on_epoch=True)
        self.valid_submission_metric(y_points_3d_pred, y_points_3d)
        self.log("valid_submission_metric", self.valid_submission_metric, on_step=True, on_epoch=True, prog_bar=False)
        return { "loss": loss }

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, torch.Tensor]:
        x_image, x_camera = batch
        y_points_3d_pred, *other = self.predict(x_image)
        self.test_predictions.append(y_points_3d_pred.detach().cpu().numpy())
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


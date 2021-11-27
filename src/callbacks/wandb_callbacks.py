import wandb
import torch 
import os 
import json 
import subprocess
import numpy as np 
import plotly.graph_objects as go
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import LoggerCollection, WandbLogger
from torchvision import transforms
from src.utils.plot_utils import plot_hand, plot_hand_3D
from src.utils.data_utils import Hand, Scale, Heatmap


def get_wandb_logger(trainer: Trainer) -> WandbLogger:
    """Safely get Weights&Biases logger from Trainer."""

    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    if isinstance(trainer.logger, LoggerCollection):
        for logger in trainer.logger:
            if isinstance(logger, WandbLogger):
                return logger

    raise Exception(
        "You are using wandb related callback, but WandbLogger was not found for some reason..."
    )


class Hand2DLogger(Callback):
    
    def __init__(self, 
        log_n_samples: int = 8,
        log_every_n_steps: int = 1000
    ) -> None:
        super().__init__()
        self.log_n_samples = log_n_samples
        self.log_every_n_steps = log_every_n_steps
        self.is_ready = True

    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        self.ready = True

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if self.is_ready and batch_idx % self.log_every_n_steps == 0:
            logger = get_wandb_logger(trainer=trainer)
            experiment = logger.experiment
            # Get a validation batch from the validation data loader
            val_batch = next(iter(trainer.datamodule.val_dataloader()))
            x_image, x_camera, y_points, *other = val_batch
            x_image = x_image.to(device=pl_module.device)
            y_points_pred, *other = pl_module.predict(x_image)
            y_points_pred = y_points_pred.detach().cpu() 
            figs = [plot_hand(transforms.ToPILImage()(x_image[i]), x_camera[i].float(), y_points_pred[i].float()) for i in range(self.log_n_samples)]
            # Log the images as wandb Image
            experiment.log({"Hand2DLogger_valid": [wandb.Image(fig) for fig in figs]})


class Hand3DLogger(Callback):

    def __init__(
        self,
        log_n_samples: int = 3,
        log_every_n_steps: int = 1000
    ) -> None: 
        super().__init__()
        self.log_n_samples = log_n_samples
        self.log_every_n_steps = log_every_n_steps
        self.is_ready = True

    def on_sanity_check_start(self, trainer, pl_module):
        self.is_ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        self.is_ready = True
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if self.is_ready and batch_idx % self.log_every_n_steps == 0:
            logger = get_wandb_logger(trainer=trainer)
            experiment = logger.experiment
            # Get a validation batch from the validation data loader
            x_image, x_camera, y_points, *other = next(iter(trainer.datamodule.val_dataloader()))
            y_points_normalized = Hand.normalize_points(y_points) if pl_module.use_normalize else y_points 
            x_image = x_image.to(device=pl_module.device)
            y_points_pred, *other = pl_module.predict(x_image)
            y_points_pred = y_points_pred.detach().cpu() 
            # Log the 3D hand keypoints as plotly plots
            for i in range(self.log_n_samples):
                experiment.log({f"Hand3DLogger_valid_{i}": plot_hand_3D(y_points_pred[i], y_points_normalized[i]) })
            

class Heatmap1DLogger(Callback):

    def __init__(
        self,
        log_n_samples: int = 3,
        log_every_n_steps: int = 1000
    ) -> None: 
        self.log_n_samples = log_n_samples
        self.log_every_n_steps = log_every_n_steps
        self.is_ready = True

    def on_sanity_check_start(self, trainer, pl_module):
        self.is_ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        self.is_ready = True

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if self.is_ready and batch_idx % self.log_every_n_steps == 0:
            logger = get_wandb_logger(trainer=trainer)
            experiment = logger.experiment
            # Get a validation batch from the validation data loader
            x_image, x_camera, y_points, *other = next(iter(trainer.datamodule.val_dataloader()))
            x_image = x_image.to(device=pl_module.device)
            y_points_pred, y_heatmap_pred = pl_module.predict(x_image)
            heatmap_size = pl_module.heatmap_size
            # Log the 3D hand keypoints as plotly plots
            for i in range(self.log_n_samples):
                fig = go.Figure(
                    data=go.Heatmap(
                    z=y_heatmap_pred[i].detach().cpu().view(-1, heatmap_size),
                    colorscale='Viridis')
                )
                experiment.log({f"Heatmap1DLogger_valid_{i}": fig })


class Heatmap2DLogger(Callback):

    def __init__(
        self,
        log_n_samples: int = 3,
        log_every_n_steps: int = 1000
    ) -> None: 
        self.log_n_samples = log_n_samples
        self.log_every_n_steps = log_every_n_steps
        self.is_ready = True

    def on_sanity_check_start(self, trainer, pl_module):
        self.is_ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        self.is_ready = True

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if self.is_ready and batch_idx % self.log_every_n_steps == 0:
            logger = get_wandb_logger(trainer=trainer)
            experiment = logger.experiment
            # Get a validation batch from the validation data loader

            x_image, x_camera, y_points_3d, *other = next(iter(trainer.datamodule.val_dataloader()))
            x_image = x_image.to(device=pl_module.device)
            y_points_2d_pred, heatmap_pred, *other = pl_module.predict_img_to_points_2d(x_image)
            heatmap_pred = heatmap_pred.detach().cpu() 

            heatmap_size = pl_module.heatmap_size
            y_points_2d = Hand.to_2d_points(y_points_3d, x_camera)
            heatmaps = Heatmap.make_gaussians(
                means=Scale.linear(y_points_2d, domain=Scale.P2D(224), range=Scale.P2D(heatmap_size)),
                size = heatmap_size, 
                sigma = pl_module.heatmap_std 
            ).detach().cpu() 

            for i in range(self.log_n_samples):
                fig = Heatmap.plot_2d(heatmap_pred[i], heatmaps[i])
                experiment.log({f"Heatmap2DLogger_{i}": fig })

class HandP2DLogger(Callback):

    def __init__(
        self,
        log_n_samples: int = 8,
        log_every_n_steps: int = 1000
    ) -> None: 
        self.log_n_samples = log_n_samples
        self.log_every_n_steps = log_every_n_steps
        self.is_ready = True

    def on_sanity_check_start(self, trainer, pl_module):
        self.is_ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        self.is_ready = True

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if self.is_ready and batch_idx % self.log_every_n_steps == 0:
            logger = get_wandb_logger(trainer=trainer)
            experiment = logger.experiment
            # Get a validation batch from the validation data loader

            x_image, x_camera, y_points_3d, *other = next(iter(trainer.datamodule.val_dataloader()))
            x_image = x_image.to(device=pl_module.device)
            y_points_2d_pred, heatmap_pred, *other = pl_module.predict_img_to_points_2d(x_image)
            
            y_points_2d_pred = y_points_2d_pred.detach().cpu() 
            y_points_2d = Hand.to_2d_points(y_points_3d, x_camera).detach().cpu() 

            fig = Hand.plot_2d(
                images = x_image[0:self.log_n_samples].detach().cpu(), 
                points = y_points_2d_pred[0:self.log_n_samples], 
                points_secondary = y_points_2d[0:self.log_n_samples]
            )
            experiment.log({f"HeatmapP2DLogger_valid": fig })


class TestResultFileLogger(Callback):

    def __init__(self):
        pass 

    def on_test_end(self, trainer, pl_module):
        # Save file to preds dir 
        path = pl_module.dir_preds_test
        test_file = os.path.join(path, "test_preds.json")
        os.makedirs(path, exist_ok=True)
        y_points_pred = np.concatenate(pl_module.test_predictions).tolist()
        with open(test_file, "w") as f:
            json.dump(y_points_pred, f)
        subprocess.call(['gzip', test_file])
        # Save file to wandb 
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment 
        experiment.save(test_file+'.gz')

"""
    File adapted from https://github.com/hobogalaxy/lightning-hydra-template/blob/main/src/utils/template_utils.py
    Helps setting up configurations (e.g. diable warnings, setting pytorch lightning in debug mode, logging, ...)
"""

import logging
import warnings
from typing import Any, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from rich import print
from rich.syntax import Syntax
from rich.tree import Tree

log = logging.getLogger(__name__)


def extras(config: DictConfig) -> None:
    # Add extra helper configurations 

    # Enable adding new keys to config
    OmegaConf.set_struct(config, False)

    # Disable python warnings if <config.disable_warnings=True>
    if config.get("disable_warnings"):
        log.info("Disabling python warnings! <config.disable_warnings=True>")
        warnings.filterwarnings("ignore")

    # Set <config.trainer.fast_dev_run=True> if <config.debug=True>
    if config.get("debug"):
        log.info("Running in debug mode! <config.debug=True>")
        config.trainer.fast_dev_run = True

    # Force debugger friendly configuration if <config.trainer.fast_dev_run=True>
    if config.trainer.get("fast_dev_run"):
        log.info(
            "Forcing debugger friendly configuration! <config.trainer.fast_dev_run=True>"
        )
        # Debuggers don't like GPUs or multiprocessing
        if config.trainer.get("gpus"):
            config.trainer.gpus = 0

    # disable adding new keys to config
    OmegaConf.set_struct(config, True)


def print_config(
    config: DictConfig,
    fields: Sequence[str] = (
        "trainer",
        "model",
        "optimizer",
        "datamodule",
        "callbacks",
        "logger",
        "seed",
    ),
    resolve: bool = True,
) -> None:
    # Prints content of DictConfig using Rich library and its tree structure.

    style = "dim"
    tree = Tree(":gear: CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(Syntax(branch_content, "yaml"))

    print(tree)


def empty(*args: str, **kwargs: int) -> None:
    pass


def log_hyperparameters(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    # This method controls which parameters from Hydra config are saved by Lightning loggers.

    hparams = {}

    # Choose which parts of hydra config will be saved to loggers
    hparams["trainer"] = config["trainer"]
    hparams["model"] = config["model"]
    hparams["datamodule"] = config["datamodule"]

    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]

    hparams["model/params_total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params_trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params_not_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    # Send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)

    # Disable logging any more hyperparameters for all loggers
    # (this is just a trick to prevent trainer from logging hparams of model, since we already did that above)
    trainer.logger.log_hyperparams = empty
import os
import dotenv
import hydra
import comet_ml
import pytorch_lightning as pl
import logging
import src.utils as utils
from omegaconf import DictConfig
from src.utils import template_utils
from typing import List, Optional
from hydra.utils import call, instantiate
from omegaconf import DictConfig

log = logging.getLogger(__name__)

# Load environment variables from `.env`.
dotenv.load_dotenv(override=True)

# Load hydra configs and initialize pytorch lightning modules
@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig) -> None:

    # Setup utilities
    template_utils.extras(config)

    # Pretty print current configs in a tree
    if config.get("print_config"):
        template_utils.print_config(config, resolve=True)

    # Apply seed to everything if avaiable from hydra config
    if "seed" in config:
        pl.seed_everything(config.seed)

    # Initialize datamodule from hydra config 
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>.")
    datamodule = instantiate(config.datamodule)

    # Initialize model from hydra config 
    log.info(f"Instantiating model <{config.model._target_}>.")
    model = instantiate(config.model)

    # Initaliza all callbacks (e.g. checkpoints, early stopping) from hydra config 
    callbacks = []
    if "callbacks" in config:
        for _, cb_conf in config["callbacks"].items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>.")
                callbacks.append(instantiate(cb_conf))

    # Initialize loggers (e.g. comet-ml) from hydra config 
    loggers = []
    if "logger" in config:
        for _, lg_conf in config["logger"].items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>.")
                loggers.append(instantiate(lg_conf))

    # Initalize lightning trainer from hydra config 
    trainer = instantiate(
        config.trainer, callbacks=callbacks, logger=loggers, _convert_="partial"
    )

    # Send config params to loggers
    log.info("Logging hyperparameters.")

    utils.template_utils.log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=loggers,
    )

    log.info(f"Test mode: {config.test}")

    # If training ...
    if not config.test:

        # Train model
        log.info("Starting training.")
        trainer.fit(model=model, datamodule=datamodule)

        # Evaluate model on test set
        log.info("Starting test set evaluation.")
        trainer.test(ckpt_path="best" if config.test_on_best else None)

        if config.test_on_best:
            log.info(
                f"Tested on best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}"
            )
        else:
            log.info("Tested on last epoch.")

    else:
        # Test mode, test only: model loaded from chekpoint
        log.info(
            f"Starting test set evaluation on checkpoint {config.model.checkpoint_path}"
        )
        trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
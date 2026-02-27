#!/usr/bin/env python3
"""
Main entry point for the digit correction deep learning project.
"""

import click
import os
from pathlib import Path
from omegaconf import OmegaConf
import lightning as L
from pytorch_lightning.loggers import TensorBoardLogger

from src.lit_module import DigitCorrectionLitModule


@click.group()
@click.version_option("1.0.0")
def cli():
    """Digit Correction Deep Learning Project CLI"""
    pass


@cli.command()
@click.option("--name", default="World", help="Name to greet")
def hello(name):
    """Say hello to someone"""
    print(f"Hello, {name}!")
    print("Welcome to the Digit Correction Deep Learning Project! 🚀")


@cli.command()
@click.option("--config", default="config.yaml", help="Path to configuration file")
def train(config):
    """Train the digit correction model"""
    print(f"Loading configuration from {config}...")
    
    # Load configuration
    if not os.path.exists(config):
        print(f"Error: Configuration file {config} not found!")
        return
    
    cfg = OmegaConf.load(config)
    print("Configuration loaded successfully!")
    print(f"Model config: {cfg.model}")
    print(f"Training config: {cfg.training}")
    
    # Create log directory
    log_dir = Path(cfg.logging.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Instantiate the Lightning module
    print("Instantiating Lightning module...")
    model = DigitCorrectionLitModule(cfg.lit_module)
    
    # Create TensorBoard logger
    print("Setting up TensorBoard logger...")
    logger = TensorBoardLogger(
        save_dir=cfg.logging.log_dir,
        name=cfg.logging.experiment_name,
        log_graph=True
    )
    
    # Create Lightning trainer
    print("Creating Lightning trainer...")
    trainer = L.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        precision=cfg.training.precision,
        logger=logger,
        log_every_n_steps=cfg.logging.log_every_n_steps,
        enable_checkpointing=True,
        enable_progress_bar=True,
        check_val_every_n_epoch=cfg.training.check_val_every_n_epoch,
    )
    
   
    trainer.fit(model)
    

@cli.command()
def predict():
    """Run inference with the trained model"""
    print("Prediction mode not implemented yet...")
    print("This will load your trained model and run predictions.")


if __name__ == "__main__":
    cli()

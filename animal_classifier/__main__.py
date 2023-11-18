from calendar import c
import os
from pathlib import Path
from py import test

import torch
import typer
from torch.utils.data import DataLoader

from animal_classifier.cfg import cfg
from animal_classifier.datasets import AnimalDataset
from animal_classifier.losses import AnimalLoss
from animal_classifier.models import AnimalNet
from animal_classifier.training import model_evaluation, model_training
from animal_classifier.utils.enums import AnimalLabel

app = typer.Typer()


@app.command()
def training(
    frames_dir: str = cfg.FRAMES_DIR,
    annotations_dir: str = cfg.ANNOTATIONS_DIR,
    model_dir: str = cfg.MODEL_DIR,
    batch_size: int = cfg.BATCH_SIZE,
    train_frac: float = cfg.TRAIN_FRAC,
    val_frac: float = cfg.VAL_FRAC,
    test_frac: float = cfg.TEST_FRAC,
    lr: float = cfg.LEARNING_RATE,
    weight_decay: float = cfg.WEIGHT_DECAY,
    num_epochs: int = cfg.NUM_EPOCHS,
):
    """
    Pipeline to train cats/dogs classifier

    Args:
        frames_dir: path to the directory containing the frames
        annotations_dir: path to the directory containing the annotations
        model_dir: path to folder where model will be stored locally to cache them
        batch_size: batch size in datasets
        train_frac: float indicating the fraction of the split to use for training
        val_frac: float indicating the fraction of the split to use for validation
        test_frac: float indicating the fraction of the split to use for testing
        lr: learning rate
        weight_decay: weight decay (L2 penalty)
        num_epochs: number of epochs to train for
    """
    assert (train_frac + val_frac + test_frac) == 1, "Train/val/test fractions must add up to 1"
    # generate datasets and split them
    train_dataset, val_dataset, test_dataset = AnimalDataset.get_splits(
        frames_dir=frames_dir,
        annotations_dir=annotations_dir,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
    )
    # train model
    model = AnimalNet()
    loss_function = AnimalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model_filename, train_loss, val_loss, _ = model_training(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model=model,
        loss_function=loss_function,
        optimizer=optimizer,
        num_epochs=2,
        batch_size=batch_size,
        model_dir=model_dir,
    )
    # this could be sent to prometheus or tensorboard for logging purposes
    typer.echo(f"Model {model_filename} trained for {num_epochs} epochs")
    typer.echo("----------------------------------------------------------")
    typer.echo(f"Loss on train dataset: {train_loss}")
    typer.echo(f"Loss on val dataset: {val_loss}")

    # test best
    model_filepath = f"{model_dir}/{model_filename}"
    checkpoint = torch.load(model_filepath)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    test_loss = model_evaluation(model=model, loss_function=loss_function, loader=test_loader)
    typer.echo(f"Loss on test dataset: {test_loss}")
    typer.echo("----------------------------------------------------------")


@app.command()
def inference(
    frame_filename: str,
    model_filename: str,
    frames_dir: str = cfg.FRAMES_DIR,
    model_dir: str = cfg.MODEL_DIR,
    threshold: float = cfg.THRESHOLD,
) -> AnimalLabel:
    """
    Pipeline to run inference for cats/dogs classifier

    Note:
        Running pipelines for inference is not ideal. We would like to expose the model with KFServe
        directly on Kubeflow with associated transforms and explain components.

    Args:
        frame_filename: name of frame to process (stored in s3 bucket)
        model_filename: name of stored model to use for inference
        frames_dir: path to the directory containing the frames
        model_dir: path to folder where model will be stored locally to cache them
        threshold: threshold used for assigning cat/dog label based on confidence score
    """
    pass
    # load model
    model = AnimalNet()
    model.load(model_filename=model_filename, model_dir=model_dir)
    model.eval()
    model.to(cfg.DEVICE)

    # get image
    image_filepath = Path(frames_dir) / frame_filename
    frame = AnimalDataset._get_frame(image_filepath=image_filepath)
    X = AnimalDataset.transform_input_frame(frame, image_size=AnimalDataset.DEFAULT_IMAGE_SIZE)
    with torch.no_grad():
        score = model(X.unsqueeze(0).to(cfg.DEVICE)).item()
        animal_label = AnimalLabel.from_score(score=score, threshold=threshold)
        typer.echo(f"Image displays a {animal_label.name.lower()}")
        return animal_label


if __name__ == "__main__":
    app()

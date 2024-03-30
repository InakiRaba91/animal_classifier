import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

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
def dataset_split(
    train_filepath: str,
    val_filepath: str,
    test_filepath: str,
    frames_dir: str = cfg.FRAMES_DIR,
    annotations_dir: str = cfg.ANNOTATIONS_DIR,
    train_frac: float = cfg.TRAIN_FRAC,
    val_frac: float = cfg.VAL_FRAC,
    test_frac: float = cfg.TEST_FRAC,
):
    """
    Pipeline to split dataset in train/val/test

    Args:
        train_filepath: path of file to store train dataset
        val_filepath: path of file to store val dataset
        test_filepath: path of file to store test dataset
        frames_dir: path to the directory containing the frames
        annotations_dir: path to the directory containing the annotations
        train_frac: float indicating the fraction of the split to use for training
        val_frac: float indicating the fraction of the split to use for validation
        test_frac: float indicating the fraction of the split to use for testing
    """
    # if snapshots exist, extend them
    if Path(train_filepath).exists() and Path(val_filepath).exists() and Path(test_filepath).exists():
        train_dataset, val_dataset, test_dataset = AnimalDataset.extend_splits(
            train_fpath=train_filepath,
            val_fpath=val_filepath,
            test_fpath=test_filepath,
            frames_dir=frames_dir,
            annotations_dir=annotations_dir,
        )
    # otherwise, create them
    else:
        assert (train_frac + val_frac + test_frac) == 1, "Train/val/test fractions must add up to 1"
        # generate datasets and split them
        train_dataset, val_dataset, test_dataset = AnimalDataset.get_splits(
            frames_dir=frames_dir,
            annotations_dir=annotations_dir,
            train_frac=train_frac,
            val_frac=val_frac,
            test_frac=test_frac,
        )
    for dataset, fpath in zip([train_dataset, val_dataset, test_dataset], [train_filepath, val_filepath, test_filepath]):
        dataset.to_snapshot(fpath=fpath)
        typer.echo(f"Dataset stored to {fpath}")


@app.command()
def training(
    train_filepath: str,
    val_filepath: str,
    model_filename: Optional[str] = None,
    annotations_dir: str = cfg.ANNOTATIONS_DIR,
    model_dir: str = cfg.MODEL_DIR,
    batch_size: int = cfg.BATCH_SIZE,
    lr: float = cfg.LEARNING_RATE,
    weight_decay: float = cfg.WEIGHT_DECAY,
    num_epochs: int = cfg.NUM_EPOCHS,
):
    """
    Pipeline to train cats/dogs classifier

    Args:
        train_filepath: path of file to store train dataset
        val_filepath: path of file to store val dataset
        model_filename: name of stored model to use for inference
        frames_dir: path to the directory containing the frames
        annotations_dir: path to the directory containing the annotations
        model_dir: path to folder where model will be stored locally to cache them
        batch_size: batch size in datasets
        lr: learning rate
        weight_decay: weight decay (L2 penalty)
        num_epochs: number of epochs to train for

    Returns:
        model_filename: name of stored model to use for inference
    """
    train_dataset = AnimalDataset.from_snapshot(fpath=train_filepath, annotations_dir=annotations_dir)
    val_dataset = AnimalDataset.from_snapshot(fpath=val_filepath, annotations_dir=annotations_dir)

    model = AnimalNet()
    loss_function = AnimalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model_filename, train_loss, val_accuracy, _ = model_training(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model_filename=model_filename,
        model=model,
        loss_function=loss_function,
        optimizer=optimizer,
        num_epochs=num_epochs,
        batch_size=batch_size,
        model_dir=model_dir,
    )

    typer.echo(f"Model {model_filename} trained for {num_epochs} epochs")
    typer.echo("----------------------------------------------------------")
    typer.echo(f"Loss on train dataset: {train_loss}")
    typer.echo(f"Accuracy on val dataset: {val_accuracy}")
    return model_filename


@app.command()
def evaluation(
    base_model_filename: str,
    test_model_filename: str,
    dataset_filepath: str,
    annotations_dir: str = cfg.ANNOTATIONS_DIR,
    model_dir: str = cfg.MODEL_DIR,
    batch_size: int = cfg.BATCH_SIZE,
) -> bool:
    """
    Pipeline to compare two cats/dogs classifier

    Args:
        base_model_filename: name of stored base model use as reference for comparison
        dataset_filename: filename of stored dataset to use for evaluation
        dataset_filepath: path of file storing test dataset
        frames_dir: path to the directory containing the frames
        annotations_dir: path to the directory containing the annotations
        model_dir: path to folder where model will be stored locally to cache them

    Returns:
        bool: True if test model is better than base model, False otherwise
    """
    base_model = AnimalNet()
    base_model.load(model_filename=base_model_filename, model_dir=model_dir)
    base_model.eval()
    test_model = AnimalNet()
    test_model.load(model_filename=test_model_filename, model_dir=model_dir)
    test_model.eval()

    dataset = AnimalDataset.from_snapshot(fpath=dataset_filepath, annotations_dir=annotations_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    base_model_accuracy = model_evaluation(model=base_model, loader=loader)
    test_model_accuracy = model_evaluation(model=test_model, loader=loader)

    if test_model_accuracy >= base_model_accuracy:
        typer.echo(f"Test model is better than base model: {base_model_accuracy=} -> {test_model_accuracy=}")
        return True
    else:
        typer.echo(f"Test model is worse than base model: {base_model_accuracy=} -> {test_model_accuracy=}")
        return False


@app.command()
def validation(
    model_filename: str,
    dataset_filepath: str,
    annotations_dir: str = cfg.ANNOTATIONS_DIR,
    model_dir: str = cfg.MODEL_DIR,
    batch_size: int = cfg.BATCH_SIZE,
    min_accuracy_validation: float = cfg.MIN_ACCURACY_VALIDATION,
) -> bool:
    """
    Pipeline to validate two cats/dogs classifier

    Args:
        model_filename: name of stored model to validate
        dataset_filepath: path of stored dataset to use for validation
        frames_dir: path to the directory containing the frames
        annotations_dir: path to the directory containing the annotations
        model_dir: path to folder where model will be stored locally to cache them
        min_accuracy_validation: minimum accuracy required for model to be deployed

    Returns:
        bool: True if test model is better than base model, False otherwise
    """
    base_model = AnimalNet()
    base_model.load(model_filename=model_filename, model_dir=model_dir)
    base_model.eval()

    dataset = AnimalDataset.from_snapshot(fpath=dataset_filepath, annotations_dir=annotations_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model_accuracy = model_evaluation(model=base_model, loader=loader)

    if model_accuracy >= min_accuracy_validation:
        typer.echo(f"Model is ready for deployment: {model_accuracy=} >= {min_accuracy_validation=}")
        return True
    else:
        typer.echo(f"Model is not ready for deployment: {model_accuracy=} < {min_accuracy_validation=}")
        return False


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
    # load model
    model = AnimalNet()
    model.load(model_filename=model_filename, model_dir=model_dir)
    model.eval()
    model.to(cfg.DEVICE)

    # get image
    image_filepath = Path(frames_dir) / frame_filename
    frame = AnimalDataset._get_frame(image_filepath=image_filepath)
    X = AnimalDataset.transform_input_frame(frame)
    with torch.no_grad():
        score = model(X.unsqueeze(0).to(cfg.DEVICE)).item()
        animal_label = AnimalLabel.from_score(score=score, threshold=threshold)
        typer.echo(f"Image displays a {animal_label.name.lower()}")
        return animal_label


@app.command()
def train_valid_model(
    model_filename: str,
    train_filepath: str,
    val_filepath: str,
    test_filepath: str,
    frames_dir: str = cfg.FRAMES_DIR,
    annotations_dir: str = cfg.ANNOTATIONS_DIR,
    model_dir: str = cfg.MODEL_DIR,
    train_frac: float = cfg.TRAIN_FRAC,
    val_frac: float = cfg.VAL_FRAC,
    test_frac: float = cfg.TEST_FRAC,
    batch_size: int = cfg.BATCH_SIZE,
    lr: float = cfg.LEARNING_RATE,
    weight_decay: float = cfg.WEIGHT_DECAY,
    num_epochs: int = cfg.NUM_EPOCHS,
    min_accuracy_validation: float = cfg.MIN_ACCURACY_VALIDATION,
    summary_file: str = cfg.SUMMARY_FILE,
):
    """
    Pipeline to train/validate cats/dogs classifier

    Args:
        model_filename: name of stored model to use for inference
        train_filepath: path of file to store train dataset
        val_filepath: path of file to store val dataset
        test_filepath: path of file to store test dataset
        frames_dir: path to the directory containing the frames
        annotations_dir: path to the directory containing the annotations
        train_frac: float indicating the fraction of the split to use for training
        val_frac: float indicating the fraction of the split to use for validation
        test_frac: float indicating the fraction of the split to use for testing
        model_dir: path to folder where model will be stored locally to cache them
        batch_size: batch size in datasets
        lr: learning rate
        weight_decay: weight decay (L2 penalty)
        num_epochs: number of epochs to train for
        min_accuracy_validation: minimum accuracy required for model to be deployed
        summary_file: path to file where to store summary of last training
    """
    # Read current version
    prev_model_fnames = [f.stem for f in Path(model_dir).glob("*.pth") if "_latest" not in f.name]
    versions = []
    for f in prev_model_fnames:
        fname_split = f.split("_v")
        if (fname_split[0] == model_filename) and fname_split[-1].isdigit():
            versions.append(int(fname_split[-1]))
    prev_version = max(versions) if versions else 0
    current_version = prev_version + 1
    current_model_filename = f"{model_filename}_v{current_version}"
    current_train_filepath = train_filepath.replace(".csv", f"_v{current_version}.csv")
    current_val_filepath = val_filepath.replace(".csv", f"_v{current_version}.csv")
    current_test_filepath = test_filepath.replace(".csv", f"_v{current_version}.csv")

    # copy prev files to new version to extend them if they exist
    if prev_version > 0:
        prev_train_filepath = train_filepath.replace(".csv", f"_v{prev_version}.csv")
        prev_val_filepath = val_filepath.replace(".csv", f"_v{prev_version}.csv")
        prev_test_filepath = test_filepath.replace(".csv", f"_v{prev_version}.csv")
        shutil.copy(prev_train_filepath, current_train_filepath)
        shutil.copy(prev_val_filepath, current_val_filepath)
        shutil.copy(prev_test_filepath, current_test_filepath)

    # 1. Create/extend dataset splits
    dataset_split(
        train_filepath=current_train_filepath,
        val_filepath=current_val_filepath,
        test_filepath=current_test_filepath,
        frames_dir=frames_dir,
        annotations_dir=annotations_dir,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
    )

    # 2. Train model
    training(
        train_filepath=current_train_filepath,
        val_filepath=current_val_filepath,
        model_filename=current_model_filename,
        annotations_dir=annotations_dir,
        model_dir=model_dir,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        num_epochs=num_epochs,
    )

    # 3. Compare against previous version, if it exists
    is_better_model = True
    if prev_version > 0:
        prev_model_filename = f"{model_filename}_v{prev_version}"
        is_better_model = evaluation(
            base_model_filename=f"{prev_model_filename}.pth",
            test_model_filename=f"{current_model_filename}.pth",
            dataset_filepath=current_val_filepath,
            annotations_dir=annotations_dir,
            model_dir=model_dir,
            batch_size=batch_size,
        )

    # 4. Validate model
    is_valid_model = validation(
        model_filename=f"{current_model_filename}.pth",
        dataset_filepath=current_test_filepath,
        annotations_dir=annotations_dir,
        model_dir=model_dir,
        batch_size=batch_size,
        min_accuracy_validation=min_accuracy_validation,
    )

    # 5. Update info of last training
    info = {"date": datetime.now().strftime("%Y-%m-%d"), "num_items": len(list(Path(annotations_dir).rglob("*.json")))}
    with open(Path(summary_file), "w") as f:  # type: ignore
        json.dump(info, f, indent=4)  # type: ignore

    # 6. Clean up
    if is_better_model and is_valid_model:
        typer.echo("Newly trained model improves current one")
        sys.exit(0)  # Indicate success with a status code of 0
    else:
        # remove model and snapshots
        (Path(model_dir) / f"{current_model_filename}.pth").unlink()
        (Path(model_dir) / f"{current_model_filename}_latest.pth").unlink()
        Path(current_train_filepath).unlink()
        Path(current_val_filepath).unlink()
        Path(current_test_filepath).unlink()
        typer.echo("Newly trained model does not improve current one")
        sys.exit(1)


if __name__ == "__main__":
    app()

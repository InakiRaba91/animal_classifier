import os
from unittest import mock

import pytest
import torch
from torch.utils.data import DataLoader

from animal_classifier.datasets import AnimalDataset
from animal_classifier.losses import AnimalLoss
from animal_classifier.models import AnimalNet
from animal_classifier.training import model_evaluation, model_training


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("shuffle", [True, False])
def test_model_evaluation(
    frames_dir: str,
    annotations_dir: str,
    batch_size: int,
    shuffle: bool,
    tol: float = 1e-6,
):
    model = AnimalNet()
    loss_function = AnimalLoss()
    dataset = AnimalDataset(frames_dir=frames_dir, annotations_dir=annotations_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    loss = model_evaluation(model=model, loss_function=loss_function, loader=dataloader)
    expected_loss = 0.601394
    assert abs(loss - expected_loss) < tol


# mock train_test_split to avoid randomnes
@mock.patch(
    "animal_classifier.datasets.dataset.train_test_split",
    wraps=lambda x, train_size: (x[: int(len(x) * train_size)], x[int(len(x) * train_size) :]),
)
@pytest.mark.parametrize(
    "batch_size, expected_train_loss, expected_best_loss", [(1, 0.374296, 0.567499), (2, 0.381158, 0.567586)]
)
def test_model_training(
    mocked_train_test_split,
    frames_dir: str,
    annotations_dir: str,
    model_dir: str,
    batch_size: int,
    expected_train_loss: float,
    expected_best_loss: float,
    tol: float = 1e-6,
):
    train_frac, val_frac, test_frac = 0.6, 0.2, 0.2
    train_dataset, val_dataset, _ = AnimalDataset.get_splits(
        frames_dir=frames_dir,
        annotations_dir=annotations_dir,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
    )
    model = AnimalNet()
    loss_function = AnimalLoss()
    optimizer = torch.optim.Adam(model.parameters())
    _, train_loss, best_loss, _ = model_training(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model=model,
        loss_function=loss_function,
        optimizer=optimizer,
        num_epochs=2,
        batch_size=batch_size,
        model_dir=model_dir,
    )
    assert abs(train_loss - expected_train_loss) < tol
    assert abs(best_loss - expected_best_loss) < tol
    _, _, files = next(os.walk(os.path.join(pytest.test_data_root_location, model_dir)))  # type: ignore
    assert len(files) == 2  # latest and best

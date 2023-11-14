""" Unit tests configuration file.
    At the moment:
    - Adds command line option parameters parsing
    - Sets test data location
"""
import json
import os
from pathlib import Path
from typing import List
from warnings import simplefilter

import cv2
import numpy as np
import pytest
import torch

from animal_classifier.cfg.config import StagingConfig
from animal_classifier.model import AnimalNet, StateInfo
from animal_classifier.utils.enums import AnimalLabel
from animal_classifier.utils.image_size import ImageSize

# ignore all future warnings


simplefilter(action="ignore", category=FutureWarning)


def pytest_configure():
    dir_name = os.path.dirname(__file__)
    # Define shared variables across tests
    pytest.test_data_root_location = os.path.abspath(os.path.join(dir_name, "../tests/data/"))


def pytest_addoption(parser):
    parser.addoption(
        "--save-to-regression-file",
        action="store_true",
        default=False,
        help="Save tests data outputs to regression file: True or False",
    )
    parser.addoption(
        "--generate-figures",
        action="store_true",
        default=False,
        help="Generate graphs from tests benches (if available): True or False",
    )


@pytest.fixture
def save_to_regression_file(request):
    return request.config.getoption("--save-to-regression-file")


@pytest.fixture
def generate_figures(request):
    return request.config.getoption("--generate-figures")


@pytest.fixture
def random_state() -> np.random.RandomState:
    return np.random.RandomState(seed=0)  # so we always generate the same labels


@pytest.fixture
def data_dir(tmpdir_factory) -> Path:
    return tmpdir_factory.mktemp("data")


@pytest.fixture
def num_frames() -> Path:
    return 5


@pytest.fixture
def image_size() -> ImageSize:
    return ImageSize(height=10, width=10)


@pytest.fixture
def frames(num_frames: int, image_size: ImageSize) -> List[np.ndarray]:
    frames = []
    for _ in range(num_frames):
        frames.append(np.random.randint(0, 255, size=(image_size.height, image_size.width, 3), dtype=np.uint8))
    return frames


@pytest.fixture
def frames_dir(data_dir: Path, frames: List[np.ndarray]) -> str:
    frames_dir = data_dir / "frames"
    frames_dir.mkdir()
    for idx, frame in enumerate(frames):
        frame_path = frames_dir / f"{idx}.png"
        cv2.imwrite(str(frame_path), frame)
    return str(frames_dir)


@pytest.fixture
def labels(num_frames: int) -> List[AnimalLabel]:
    labels = []
    for _ in range(num_frames):
        labels.append(np.random.choice(list(AnimalLabel)))
    return labels


@pytest.fixture
def annotations_dir(data_dir: Path, labels: List[AnimalLabel]) -> str:
    annotations_dir = data_dir / "annotations"
    annotations_dir.mkdir()
    for idx, label in enumerate(labels):
        annotation = {"label": label.name}
        annotation_path = annotations_dir / f"{idx}.json"
        with open(annotation_path, "w") as f:
            json.dump(annotation, f)
    return str(annotations_dir)


@pytest.fixture
def model_dir(tmpdir_factory) -> Path:
    return tmpdir_factory.mktemp("models")


@pytest.fixture
def model() -> AnimalNet:
    return AnimalNet()


@pytest.fixture
def state_info(model: AnimalNet) -> StateInfo:
    return StateInfo(epoch=0, optimizer_state_dict=torch.optim.Adam(model.parameters()).state_dict(), loss=0.0)


@pytest.fixture
def model_fpath(model: AnimalNet, model_dir: Path, state_info: StateInfo) -> Path:
    model_fpath = model_dir / "CatsAndDogsTest"
    torch.save(
        {
            "epoch": state_info.epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": state_info.optimizer_state_dict,
            "loss": state_info.loss,
        },
        f"{model_fpath}",
    )
    return Path(model_fpath)


# Prevent pytest from trying to collect TestConfig as tests since it starts with Test
# Ref: https://adamj.eu/tech/2020/07/28/how-to-fix-a-pytest-collection-warning-about-web-tests-test-app-class/
StagingConfig.__test__ = False  # type: ignore

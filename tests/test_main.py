import os
from pyexpat import model
from typing import Tuple

import pytest
from typer.testing import CliRunner

from animal_classifier.cfg import cfg
from animal_classifier.__main__ import app

runner = CliRunner()


class TestCLI:
    def test_app_training(self, frames_dir: str, annotations_dir: str, model_dir: str):
        # We need file_keys as an arg so it calls the fixture to ensure test files are in the mocked bucket
        # as well as the dirs for clean-up after test is finished
        args = [
            "training", 
            "--train-frac", 
            "0.6", "--val-frac", 
            "0.2", 
            "--test-frac", 
            "0.2",
            "--frames-dir",
            frames_dir,
            "--annotations-dir",
            annotations_dir,
            "--model-dir",
            model_dir,
        ]
        result = runner.invoke(app, args)
        assert result.exit_code == 0

        # assert model was stored
        _, _, files = next(os.walk(os.path.join(pytest.test_data_root_location, model_dir)))  # type: ignore
        assert len(files) == 2 # latest and best

    def test_app_inference(self, frames_dir: str, model_dir: str, model_fpath: str):
        args = [
            "inference",
            "0.png",
            model_fpath.stem,
            "--frames-dir",
            frames_dir,
            "--model-dir",
            model_dir
        ]
        result = runner.invoke(app, args)
        assert result.exit_code == 0
        assert "Image displays a dog" in result.stdout

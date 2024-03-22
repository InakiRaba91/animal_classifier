import os
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import torch
from typer.testing import CliRunner

from animal_classifier.__main__ import (
    app,
    dataset_split,
    evaluation,
    training,
    validation,
)

runner = CliRunner()


class TestCLI:
    def test_app_dataset_split(self, data_dir: Path, frames_dir: str, annotations_dir: str, num_frames: int):
        train_fpath = data_dir / "train.csv"
        val_fpath = data_dir / "val.csv"
        test_fpath = data_dir / "test.csv"
        train_frac, val_frac, test_frac = 0.6, 0.2, 0.2
        args = [
            "dataset-split",
            str(train_fpath),
            str(val_fpath),
            str(test_fpath),
            "--frames-dir",
            frames_dir,
            "--annotations-dir",
            annotations_dir,
            "--train-frac",
            train_frac,
            "--val-frac",
            val_frac,
            "--test-frac",
            test_frac,
        ]
        result = runner.invoke(app, args)
        assert result.exit_code == 0

        # assert datasets were stored
        assert train_fpath.exists()
        assert val_fpath.exists()
        assert test_fpath.exists()
        df_train = pd.read_csv(train_fpath)
        df_val = pd.read_csv(val_fpath)
        df_test = pd.read_csv(test_fpath)
        assert len(df_train) == num_frames * train_frac
        assert len(df_val) == num_frames * val_frac
        assert len(df_test) == num_frames * test_frac

    def test_app_dataset_split_with_prev_splits(self, data_dir: Path, frames_dir: str, annotations_dir: str, num_frames: int):
        # setup
        train_fpath = data_dir / "train.csv"
        val_fpath = data_dir / "val.csv"
        test_fpath = data_dir / "test.csv"
        train_frac, val_frac, test_frac = 0.6, 0.2, 0.2
        snapshot_filepaths = [train_fpath, val_fpath, test_fpath]
        prev_frame_filenames = [f"{frames_dir}/{i}.png" for i in range(num_frames, 2*num_frames)]
        prev_train_filenames = prev_frame_filenames[:int(num_frames * train_frac)]
        prev_val_filenames = prev_frame_filenames[int(num_frames * train_frac) : int(num_frames * (train_frac + val_frac))]
        prev_test_filenames = prev_frame_filenames[int(num_frames * (train_frac + val_frac)):]
        set_filenames = [prev_train_filenames, prev_val_filenames, prev_test_filenames]
        for filenames, fpath in zip(set_filenames, snapshot_filepaths):
            pd.DataFrame(filenames, columns=["frame_filename"]).to_csv(fpath, index=False)
        args = [
            "dataset-split",
            str(train_fpath),
            str(val_fpath),
            str(test_fpath),
            "--frames-dir",
            frames_dir,
            "--annotations-dir",
            annotations_dir,
            "--train-frac",
            train_frac,
            "--val-frac",
            val_frac,
            "--test-frac",
            test_frac,
        ]
        result = runner.invoke(app, args)
        assert result.exit_code == 0

        # assert datasets were stored
        assert train_fpath.exists()
        assert val_fpath.exists()
        assert test_fpath.exists()
        df_train = pd.read_csv(train_fpath)
        df_val = pd.read_csv(val_fpath)
        df_test = pd.read_csv(test_fpath)
        assert len(df_train) + len(df_val) + len(df_test) == 2 * num_frames
        # val sets are the same
        assert len(df_val) == len(prev_val_filenames)
        # test sets are completely different but with equal length
        assert len(df_test) == len(prev_test_filenames)
        assert len(set(df_test.frame_filename) & set(prev_test_filenames)) == 0
        # train sets include old train/test files but not new test/val files
        assert len(set(df_train.frame_filename) & set(df_test.frame_filename)) == 0
        assert len(set(df_train.frame_filename) & set(df_val.frame_filename)) == 0
        assert all(f in df_train.frame_filename.values for f in prev_train_filenames + prev_test_filenames)

    @pytest.mark.parametrize("model_filename", [None, "base_model"])
    def test_app_training(
        self, num_frames: int, data_dir: Path, frames_dir: str, annotations_dir: str, model_dir: str, model_filename: str
    ):
        # setup
        # We need file_keys as an arg so it calls the fixture to ensure test files are in the mocked bucket
        # as well as the dirs for clean-up after test is finished
        train_frac, val_frac = 0.6, 0.2
        end_train = int(num_frames * train_frac)
        end_val = int(num_frames * (val_frac + train_frac))
        train_fpath = data_dir / "train.csv"
        val_fpath = data_dir / "val.csv"
        train_df = pd.DataFrame([f"{frames_dir}/{idx}.png" for idx in range(end_train)], columns=["frame_filename"])
        val_df = pd.DataFrame([f"{frames_dir}/{idx}.png" for idx in range(end_train, end_val)], columns=["frame_filename"])
        train_df.to_csv(train_fpath, index=False)
        val_df.to_csv(val_fpath, index=False)

        # execute
        args = [
            "training",
            str(train_fpath),
            str(val_fpath),
            "--annotations-dir",
            annotations_dir,
            "--model-dir",
            model_dir,
        ]
        if model_filename is not None:
            args.extend(["--model-filename", model_filename])
        result = runner.invoke(app, args)
        assert result.exit_code == 0

        # assert model was stored
        _, _, files = next(os.walk(os.path.join(pytest.test_data_root_location, model_dir)))  # type: ignore
        assert len(files) == 2  # latest and best

    @pytest.mark.parametrize("base_accuracy, test_accuracy, verdict", [(0.5, 1.0, "better"), (1.0, 0.5, "worse")])
    def test_app_evaluation(
        self,
        model_fpath: Path,
        data_dir: Path,
        frames_dir: str,
        annotations_dir: str,
        model_dir: str,
        base_accuracy: float,
        test_accuracy: float,
        verdict: str,
    ):
        # setup
        test_fpath = data_dir / "test.csv"
        test_df = pd.DataFrame([f"{frames_dir}/0.png"], columns=["frame_filename"])
        test_df.to_csv(test_fpath, index=False)

        # execute
        args = [
            "evaluation",
            str(model_fpath.stem),
            str(model_fpath.stem),
            str(test_fpath),
            "--annotations-dir",
            annotations_dir,
            "--model-dir",
            model_dir,
        ]
        with patch("animal_classifier.training._matching_predictions", side_effect=[base_accuracy, test_accuracy]):
            result = runner.invoke(app, args)

        # assert
        assert result.exit_code == 0
        assert f"Test model is {verdict} than base model" in result.stdout

    @pytest.mark.parametrize("accuracy, verdict", [(0.6, "ready"), (0.4, "not ready")])
    def test_app_validation(
        self,
        model_fpath: Path,
        data_dir: Path,
        frames_dir: str,
        annotations_dir: str,
        model_dir: str,
        accuracy: float,
        verdict: str,
    ):
        # setup
        test_fpath = data_dir / "test.csv"
        test_df = pd.DataFrame([f"{frames_dir}/0.png"], columns=["frame_filename"])
        test_df.to_csv(test_fpath, index=False)

        # execute
        args = [
            "validation",
            str(model_fpath.stem),
            str(test_fpath),
            "--annotations-dir",
            annotations_dir,
            "--model-dir",
            model_dir,
            "--min-accuracy-validation",
            0.5,
        ]
        with patch("animal_classifier.training._matching_predictions", return_value=accuracy):
            result = runner.invoke(app, args)

        # assert
        assert result.exit_code == 0
        assert f"Model is {verdict} for deployment" in result.stdout

    def test_app_inference(self, frames_dir: str, model_dir: str, model_fpath: str):
        args = ["inference", "0.png", model_fpath.stem, "--frames-dir", frames_dir, "--model-dir", model_dir]
        result = runner.invoke(app, args)
        assert result.exit_code == 0
        assert "Image displays a dog" in result.stdout


class TestFullPipelineIntegration:
    def test_full_pipeline_integration(self, data_dir: Path, frames_dir: str, annotations_dir: str, model_dir: str):
        # setup
        train_fpath = data_dir / "train.csv"
        val_fpath = data_dir / "val.csv"
        test_fpath = data_dir / "test.csv"
        train_frac, val_frac, test_frac = 0.6, 0.2, 0.2
        base_model_filename = "base_model"

        # execute
        dataset_split(
            train_filepath=train_fpath,
            val_filepath=val_fpath,
            test_filepath=test_fpath,
            frames_dir=frames_dir,
            annotations_dir=annotations_dir,
            train_frac=train_frac,
            val_frac=val_frac,
            test_frac=test_frac,
        )
        training(
            train_filepath=train_fpath,
            val_filepath=val_fpath,
            model_filename=base_model_filename,
            annotations_dir=annotations_dir,
            model_dir=model_dir,
        )
        test_model_filename = training(
            train_filepath=train_fpath,
            val_filepath=val_fpath,
            annotations_dir=annotations_dir,
            model_dir=model_dir,
        )

        with patch("animal_classifier.training._matching_predictions", side_effect=[torch.tensor(v) for v in [0.4, 0.6, 0.6]]):
            is_better_model = evaluation(
                base_model_filename=f"{base_model_filename}.pth",
                test_model_filename=f"{test_model_filename}.pth",
                dataset_filepath=test_fpath,
                annotations_dir=annotations_dir,
                model_dir=model_dir,
            )
            is_valid_model = validation(
                model_filename=f"{test_model_filename}.pth",
                dataset_filepath=test_fpath,
                annotations_dir=annotations_dir,
                model_dir=model_dir,
                min_accuracy_validation=0.5,
            )

        # assert
        assert is_better_model
        assert is_valid_model

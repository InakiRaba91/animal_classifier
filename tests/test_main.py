import os
import shutil
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import torch
from typer.testing import CliRunner

from animal_classifier.__main__ import app

runner = CliRunner()


class TestCLI:
    @pytest.fixture
    def snapshot_fpaths(self, data_dir: Path) -> dict[str, Path]:
        return {
            "train": data_dir / "train.csv",
            "val": data_dir / "val.csv",
            "test": data_dir / "test.csv",
        }

    def test_app_dataset_split(self, snapshot_fpaths: dict[str, Path], frames_dir: str, annotations_dir: str, num_frames: int):
        train_frac, val_frac, test_frac = 0.6, 0.2, 0.2
        train_fpath, val_fpath, test_fpath = [snapshot_fpaths[key] for key in ["train", "val", "test"]]
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

    def test_app_dataset_split_with_prev_splits(
        self, snapshot_fpaths: dict[str, Path], frames_dir: str, annotations_dir: str, num_frames: int
    ):
        # setup
        train_fpath, val_fpath, test_fpath = [snapshot_fpaths[key] for key in ["train", "val", "test"]]
        train_frac, val_frac, test_frac = 0.6, 0.2, 0.2
        snapshot_filepaths = [train_fpath, val_fpath, test_fpath]
        prev_frame_filenames = [f"{frames_dir}/{i}.png" for i in range(num_frames, 2 * num_frames)]
        prev_train_filenames = prev_frame_filenames[: int(num_frames * train_frac)]
        prev_val_filenames = prev_frame_filenames[int(num_frames * train_frac) : int(num_frames * (train_frac + val_frac))]
        prev_test_filenames = prev_frame_filenames[int(num_frames * (train_frac + val_frac)) :]
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

    @pytest.mark.parametrize("model_filename", [None, "CatsAndDogsTest"])
    def test_app_training(
        self,
        num_frames: int,
        snapshot_fpaths: dict[str, Path],
        frames_dir: str,
        annotations_dir: str,
        model_dir: str,
        model_filename: str,
    ):
        # setup
        # We need file_keys as an arg so it calls the fixture to ensure test files are in the mocked bucket
        # as well as the dirs for clean-up after test is finished
        train_frac, val_frac = 0.6, 0.2
        train_fpath, val_fpath = [snapshot_fpaths[key] for key in ["train", "val"]]
        end_train = int(num_frames * train_frac)
        end_val = int(num_frames * (val_frac + train_frac))
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
            "--num-epochs",
            1,
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
        snapshot_fpaths: dict[str, Path],
        frames_dir: str,
        annotations_dir: str,
        model_dir: str,
        base_accuracy: float,
        test_accuracy: float,
        verdict: str,
    ):
        # setup
        test_df = pd.DataFrame([f"{frames_dir}/0.png"], columns=["frame_filename"])
        test_fpath = snapshot_fpaths["test"]
        test_df.to_csv(test_fpath, index=False)

        # execute
        args = [
            "evaluation",
            str(model_fpath.name),
            str(model_fpath.name),
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
        snapshot_fpaths: dict[str, Path],
        frames_dir: str,
        annotations_dir: str,
        model_dir: str,
        accuracy: float,
        verdict: str,
    ):
        # setup
        test_fpath = snapshot_fpaths["test"]
        test_df = pd.DataFrame([f"{frames_dir}/0.png"], columns=["frame_filename"])
        test_df.to_csv(test_fpath, index=False)

        # execute
        args = [
            "validation",
            str(model_fpath.name),
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
        args = ["inference", "0.png", model_fpath.name, "--frames-dir", frames_dir, "--model-dir", model_dir]
        result = runner.invoke(app, args)
        assert result.exit_code == 0
        assert "Image displays a dog" in result.stdout

    @pytest.mark.parametrize(
        "test_accuracy, exit_code",
        [
            (1.0, 0),  # valid model
            (0.0, 1),  # invalid model
        ],
    )
    def test_train_valid_model_first_time(
        self,
        data_dir: Path,
        snapshot_fpaths: dict[str, Path],
        model_dir: str,
        frames_dir: str,
        annotations_dir: str,
        test_accuracy: float,
        exit_code: int,
    ):
        train_frac, val_frac, test_frac = 0.6, 0.2, 0.2
        base_model_filename = "CatsAndDogsTest"
        min_accuracy_validation = 0.5
        expected_num_snapshots, expected_num_models = 0, 0
        val_accuracy_train = 1.0
        if test_accuracy > min_accuracy_validation:
            expected_num_snapshots, expected_num_models = 3, 2
        args = [
            "train-valid-model",
            base_model_filename,
            str(snapshot_fpaths["train"]),
            str(snapshot_fpaths["val"]),
            str(snapshot_fpaths["test"]),
            "--frames-dir",
            frames_dir,
            "--annotations-dir",
            annotations_dir,
            "--model-dir",
            model_dir,
            "--train-frac",
            train_frac,
            "--val-frac",
            val_frac,
            "--test-frac",
            test_frac,
            "--min-accuracy-validation",
            min_accuracy_validation,
            "--num-epochs",
            1,
        ]
        accuracies = [val_accuracy_train, test_accuracy]

        # execute
        with patch("animal_classifier.training._matching_predictions", side_effect=[torch.tensor(v) for v in accuracies]):
            result = runner.invoke(app, args)
        num_snapshots = len(list(Path(data_dir).glob("*.csv")))
        num_models = len(list(Path(model_dir).glob("*.pth")))

        # assert
        assert result.exit_code == exit_code
        assert num_snapshots == expected_num_snapshots
        assert num_models == expected_num_models

    @pytest.mark.parametrize("test_accuracy", [0.0, 1.0])
    @pytest.mark.parametrize("val_accuracy", [0.0, 1.0])
    def test_train_valid_model_second_time(
        self,
        data_dir: Path,
        snapshot_fpaths: dict[str, Path],
        model_dir: str,
        frames_dir: str,
        annotations_dir: str,
        test_accuracy: float,
        val_accuracy: int,
        num_frames: int,
        model_fpath: Path,  # needed so there's an existing model to compare against
    ):
        # setup
        train_fpath, val_fpath, test_fpath = [snapshot_fpaths[key] for key in ["train", "val", "test"]]
        train_frac, val_frac, test_frac = 0.6, 0.2, 0.2
        prev_frame_filenames = [f"{frames_dir}/{i}.png" for i in range(num_frames, 2 * num_frames)]
        prev_train_filenames = prev_frame_filenames[: int(num_frames * train_frac)]
        prev_val_filenames = prev_frame_filenames[int(num_frames * train_frac) : int(num_frames * (train_frac + val_frac))]
        prev_test_filenames = prev_frame_filenames[int(num_frames * (train_frac + val_frac)) :]
        set_filenames = [prev_train_filenames, prev_val_filenames, prev_test_filenames]
        prev_snapshot_filepaths = [train_fpath, val_fpath, test_fpath]
        for filenames, fpath in zip(set_filenames, prev_snapshot_filepaths):
            fpath = Path(fpath).parent / f"{Path(fpath).stem}_v1.csv"
            pd.DataFrame(filenames, columns=["frame_filename"]).to_csv(fpath, index=False)
        for idx in range(num_frames, 2 * num_frames):
            shutil.copy(f"{frames_dir}/0.png", f"{frames_dir}/{idx}.png")
            shutil.copy(f"{annotations_dir}/0.json", f"{annotations_dir}/{idx}.json")
        shutil.copy(model_fpath, model_dir / f"{model_fpath.stem}_latest.pth")

        base_model_filename = model_fpath.stem.split("_")[0]
        min_accuracy_validation = 0.5
        val_accuracy_train = 1.0
        ref_accuracy = 0.5
        expected_num_snapshots, expected_num_models = 3, 2
        exit_code = 1
        if (test_accuracy > min_accuracy_validation) and (val_accuracy > ref_accuracy):
            expected_num_snapshots, expected_num_models = 6, 4
            exit_code = 0
        args = [
            "train-valid-model",
            base_model_filename,
            str(train_fpath),
            str(val_fpath),
            str(test_fpath),
            "--frames-dir",
            frames_dir,
            "--annotations-dir",
            annotations_dir,
            "--model-dir",
            model_dir,
            "--train-frac",
            train_frac,
            "--val-frac",
            val_frac,
            "--test-frac",
            test_frac,
            "--min-accuracy-validation",
            min_accuracy_validation,
            "--num-epochs",
            1,
        ]
        accuracies = [val_accuracy_train, ref_accuracy, val_accuracy, test_accuracy]

        # execute
        with patch("animal_classifier.training._matching_predictions", side_effect=[torch.tensor(v) for v in accuracies]):
            result = runner.invoke(app, args)
        num_snapshots = len(list(Path(data_dir).glob("*.csv")))
        num_models = len(list(Path(model_dir).glob("*.pth")))

        # assert
        assert result.exit_code == exit_code
        assert num_snapshots == expected_num_snapshots
        assert num_models == expected_num_models

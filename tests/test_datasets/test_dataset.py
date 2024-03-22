from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from animal_classifier.datasets.dataset import AnimalDataset
from animal_classifier.utils.enums import AnimalLabel


class TestAnimalDataset:
    def test_dataset_length(self, frames_dir: str, annotations_dir: str, num_frames: int):
        dataset = AnimalDataset(frames_dir=frames_dir, annotations_dir=annotations_dir)
        assert len(dataset) == num_frames

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_generate_dataset(
        self,
        frames_dir: str,
        annotations_dir: str,
        frames: List[np.ndarray],
        labels: List[AnimalLabel],
        batch_size: int,
    ):
        # group expected data in batches
        batched_normalized_frames: List[torch.Tensor] = []
        batched_labels: List[torch.Tensor] = []
        for idx in range(0, len(frames), batch_size):
            tensor_normalized_frames = [
                AnimalDataset.transform_input_frame(frame).unsqueeze(0) for frame in frames[idx : (idx + batch_size)]
            ]
            tensor_labels = [torch.DoubleTensor([label.value]).unsqueeze(0) for label in labels[idx : (idx + batch_size)]]
            batched_normalized_frames.append(torch.cat(tensor_normalized_frames))
            batched_labels.append(torch.cat(tensor_labels))

        dataset = AnimalDataset(frames_dir=frames_dir, annotations_dir=annotations_dir)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for batch_idx, (normalized_frame, label) in enumerate(dataloader):
            assert (batched_normalized_frames[batch_idx] == normalized_frame).all()
            assert (batched_labels[batch_idx] == label).all()

    def test_split_datasets(self, frames_dir: str, annotations_dir: str, num_frames: int):
        train_frac, val_frac, test_frac = 0.6, 0.2, 0.2
        train, val, test = AnimalDataset.get_splits(
            frames_dir=frames_dir,
            annotations_dir=annotations_dir,
            train_frac=train_frac,
            val_frac=val_frac,
            test_frac=test_frac,
        )
        assert len(train._frame_filenames) == num_frames * train_frac
        assert len(val._frame_filenames) == num_frames * val_frac
        assert len(test._frame_filenames) == num_frames * test_frac

    def test_extend_datasets(self, frames_dir: str, annotations_dir: str, num_frames: int, data_dir: Path):
        # setup
        train_filepath = data_dir / "train.csv"
        val_filepath = data_dir / "val.csv"
        test_filepath = data_dir / "test.csv"
        snapshot_filepaths = [train_filepath, val_filepath, test_filepath]
        train_frac, val_frac = 0.6, 0.2
        prev_frame_filenames = [f"{frames_dir}/{i}.png" for i in range(num_frames, 2 * num_frames)]
        prev_train_filenames = prev_frame_filenames[: int(num_frames * train_frac)]
        prev_val_filenames = prev_frame_filenames[int(num_frames * train_frac) : int(num_frames * (train_frac + val_frac))]
        prev_test_filenames = prev_frame_filenames[int(num_frames * (train_frac + val_frac)) :]
        set_filenames = [prev_train_filenames, prev_val_filenames, prev_test_filenames]
        for filenames, fpath in zip(set_filenames, snapshot_filepaths):
            pd.DataFrame(filenames, columns=["frame_filename"]).to_csv(fpath, index=False)

        # execute
        train, val, test = AnimalDataset.extend_splits(
            frames_dir=frames_dir,
            annotations_dir=annotations_dir,
            train_fpath=train_filepath,
            val_fpath=val_filepath,
            test_fpath=test_filepath,
        )

        # assert
        assert len(train) + len(val) + len(test) == 2 * num_frames
        # val sets are the same
        assert len(val) == len(prev_val_filenames)
        # test sets are completely different but with equal length
        assert len(test) == len(prev_test_filenames)
        assert len(set(test._frame_filenames) & set(prev_test_filenames)) == 0
        # train sets include old train/test files but not new test/val files
        assert len(set(train._frame_filenames) & set(test._frame_filenames)) == 0
        assert len(set(train._frame_filenames) & set(val._frame_filenames)) == 0
        assert all(Path(f) in train._frame_filenames for f in prev_train_filenames + prev_test_filenames)

    def test_to_snapshot(self, frames_dir: str, annotations_dir: str, data_dir: Path):
        # setup
        fpath = data_dir / "snapshot.csv"
        dataset = AnimalDataset(frames_dir=frames_dir, annotations_dir=annotations_dir)

        # execute
        _ = dataset.to_snapshot(fpath=fpath)

        # assert
        # check csv exists
        assert fpath.exists()
        df = pd.read_csv(fpath)
        assert len(df) == len(dataset)

    def test_from_csv(self, frames_dir: str, annotations_dir: str, data_dir: Path):
        # setup
        fpath = data_dir / "snapshot.csv"
        df = pd.DataFrame({"frame_filename": [fp.as_posix() for fp in Path(frames_dir).glob("*.png")]})
        df.to_csv(fpath, index=False)
        dataset = AnimalDataset(frames_dir=frames_dir, annotations_dir=annotations_dir)

        # execute
        dataset_from_csv = AnimalDataset.from_snapshot(fpath=fpath)

        # assert
        assert len(dataset_from_csv) == len(dataset)

    def test_snapshot_conversion_revertible(self, frames_dir: str, annotations_dir: str, data_dir: Path):
        # setup
        fpath = data_dir / "snapshot.csv"
        dataset = AnimalDataset(frames_dir=frames_dir, annotations_dir=annotations_dir)

        # execute
        dataset.to_snapshot(fpath=fpath)
        dataset_from_csv = AnimalDataset.from_snapshot(fpath=fpath)

        # assert
        assert len(dataset_from_csv) == len(dataset)
        for frame_filename, frame_filename_from_csv in zip(dataset._frame_filenames, dataset_from_csv._frame_filenames):
            assert frame_filename == frame_filename_from_csv

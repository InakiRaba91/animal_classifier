from typing import List

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from animal_classifier.dataset import AnimalDataset
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

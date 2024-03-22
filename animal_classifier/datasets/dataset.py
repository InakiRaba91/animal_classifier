import json
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from numpy import ndarray
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from animal_classifier.cfg import cfg
from animal_classifier.utils.enums import AnimalLabel
from animal_classifier.utils.image_size import ImageSize


class AnimalDataset(Dataset):
    """Annimal dataset returning corresponding pairs of frame<->label

    Args:
        frames_dir: path to the directory containing the frames
        annotations_dir: path to the directory containing the annotations
        image_size: ImageSize for reshaping
        frame_filenames: list of frame filenames to use (if None, all frames in frames_dir will be used)
    """

    DEFAULT_IMAGE_SIZE = ImageSize(width=16, height=16)

    def __init__(
        self,
        frames_dir: str = cfg.FRAMES_DIR,
        annotations_dir: str = cfg.ANNOTATIONS_DIR,
        image_size: ImageSize = DEFAULT_IMAGE_SIZE,
        frame_filenames: Optional[Tuple[str, ...]] = None,
    ):
        self._annotations_dir = Path(annotations_dir)
        if frame_filenames is None:
            self._frame_filenames: List[Path] = sorted(Path(frames_dir).glob("*.png"))
        else:
            self._frame_filenames = [Path(frames_dir) / filename for filename in frame_filenames]
        self.image_size = image_size

    def __len__(self) -> int:
        """
        Computes the length of the dataset
        """
        return len(self._frame_filenames)

    def __getitem__(self, idx: int) -> Tuple[Tensor, ndarray]:
        """Reads the image&annotation files from the bucket and returns the normalized frame and
        the corresponding label (homography_matrix)
        Args:
            idx: integer indicating what item in the dataset to access
        Returns:
            dataset length given by the number of images in the bucker
        """
        # get keys
        image_filepath = self._frame_filenames[idx]
        ann_filepath = self._annotations_dir / self._frame_filenames[idx].name.replace(".png", ".json")
        frame = self._get_frame(image_filepath=image_filepath)
        label = self._get_label(ann_filepath=ann_filepath)
        X = self.transform_input_frame(frame, image_size=self.image_size)
        y = np.array([float(label)]).astype(np.float32)
        return X, y

    @staticmethod
    def _get_frame(image_filepath: Path) -> np.ndarray:
        """Read frame from disk

        Args:
            image_filepath: path to the image file

        Returns:
            frame: reads the png file into a numpy array (channel order: BGR)
        """
        return cv2.imread(image_filepath.as_posix())

    @staticmethod
    def _get_label(ann_filepath: Path) -> int:
        """Read label from disk

        Args:
            ann_filepath: path to the annotation file

        Returns:
            label: reads the json file with annotation and returns label (0/1 for cat dog)
        """
        with open(ann_filepath) as fp:
            j = json.load(fp=fp)
        animal: str = j["label"]
        return AnimalLabel.from_string(animal=animal).value

    @classmethod
    def transform_input_frame(cls, frame: np.ndarray, image_size: ImageSize = DEFAULT_IMAGE_SIZE) -> torch.Tensor:
        """Resizes image to default size, converts to gray scale, normalizes it, vectorizes it
        and converts it to tensor

        Args:
            frame: numpy array containing the image
            image_size: ImageSize for reshaping
        Returns:
            X: tensor [color_channels, width*height, 1], where color_channels=1 since we're converting to gray
        """
        # reshape to model input size
        frame = cv2.resize(frame, (image_size.width, image_size.height), interpolation=cv2.INTER_AREA)

        # convert to single channel
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # vectorize
        frame_vectorized = frame_gray.flatten()[None, :]

        # transform to tensor in range [0, 1]
        X = transforms.ToTensor()(frame_vectorized)

        # normalize
        X = transforms.Normalize(0.5, 0.5)(X)
        return X

    @classmethod
    def get_splits(
        cls,
        frames_dir: str = cfg.FRAMES_DIR,
        annotations_dir: str = cfg.ANNOTATIONS_DIR,
        image_size: ImageSize = DEFAULT_IMAGE_SIZE,
        train_frac: float = cfg.TRAIN_FRAC,
        val_frac: float = cfg.VAL_FRAC,
        test_frac: float = cfg.TEST_FRAC,
    ):
        """
        Wrap data into 3 different instances of CatsDogsDataset corresponding to the train/val/test splits

        Args:
            frames_dir: path to the directory containing the frames
            annotations_dir: path to the directory containing the annotations
            image_size: ImageSize for reshaping
            train_frac: float indicating the fraction of the split to use for training
            val_frac: float indicating the fraction of the split to use for validation
            test_frac: float indicating the fraction of the split to use for testing

        Returns:
            train/val/test CatsDogsDatasets
        """
        frame_filenames = list(Path(frames_dir).glob("*.png"))

        train_filenames, val_test_filenames = train_test_split(frame_filenames, train_size=train_frac)
        val_filenames, test_filenames = train_test_split(val_test_filenames, train_size=val_frac / (val_frac + test_frac))

        train = cls(
            frames_dir=frames_dir,
            annotations_dir=annotations_dir,
            image_size=image_size,
            frame_filenames=train_filenames,
        )
        val = cls(
            frames_dir=frames_dir,
            annotations_dir=annotations_dir,
            image_size=image_size,
            frame_filenames=val_filenames,
        )
        test = cls(
            frames_dir=frames_dir,
            annotations_dir=annotations_dir,
            image_size=image_size,
            frame_filenames=test_filenames,
        )
        return train, val, test

    @classmethod
    def extend_splits(
        cls,
        train_fpath: str,
        val_fpath: str,
        test_fpath: str,
        frames_dir: str = cfg.FRAMES_DIR,
        annotations_dir: str = cfg.ANNOTATIONS_DIR,
        image_size: ImageSize = DEFAULT_IMAGE_SIZE,
    ) -> Tuple["AnimalDataset", "AnimalDataset", "AnimalDataset"]:
        """
        Wrap data into 3 different instances of CatsDogsDataset corresponding to the train/val/test splits

        The current splits come from a set of items given by Tr + Vl + Ts. After the split, the dataset is extended
        to incorporate more items Ex, so it's not comprised of Tr + Vl + Ts + Ex.

        It extends the existing splits by using the following strategy:
        1. Vl'=Vl (for a fair comparison)
        2. Ts' = rand_sample(Ex, size=Ts)
        2. Tr' = Tr + Ts + (Ex - Ts')

        Args:
            train_fpath: path of file where existing train split snapshot was stored to
            val_fpath: path of file where existing val split snapshot was stored to
            test_fpath: path of file where existing test split snapshot was stored to
            frames_dir: path to the directory containing the frames
            annotations_dir: path to the directory containing the annotations
            image_size: ImageSize for reshaping

        Returns:
            train/val/test CatsDogsDatasets
        """
        frame_filenames = list(Path(frames_dir).glob("*.png"))
        prev_train_filenames = cls.from_snapshot(
            fpath=train_fpath, annotations_dir=annotations_dir, image_size=image_size
        )._frame_filenames
        prev_val_filenames = cls.from_snapshot(
            fpath=val_fpath, annotations_dir=annotations_dir, image_size=image_size
        )._frame_filenames
        prev_test_filenames = cls.from_snapshot(
            fpath=test_fpath, annotations_dir=annotations_dir, image_size=image_size
        )._frame_filenames
        prev_filenames = set(prev_train_filenames) | set(prev_val_filenames) | set(prev_test_filenames)
        ext_filenames = set(frame_filenames) - prev_filenames

        new_test_filenames = np.random.choice(list(ext_filenames), size=len(prev_test_filenames), replace=False)  # type: ignore
        new_train_filenames = list(ext_filenames - set(new_test_filenames)) + prev_test_filenames + prev_train_filenames

        train = cls(
            frames_dir=frames_dir,
            annotations_dir=annotations_dir,
            image_size=image_size,
            frame_filenames=tuple([f.as_posix() for f in new_train_filenames]),
        )
        val = cls(
            frames_dir=frames_dir,
            annotations_dir=annotations_dir,
            image_size=image_size,
            frame_filenames=tuple([f.as_posix() for f in prev_val_filenames]),
        )
        test = cls(
            frames_dir=frames_dir,
            annotations_dir=annotations_dir,
            image_size=image_size,
            frame_filenames=tuple([f.as_posix() for f in new_test_filenames]),
        )
        return train, val, test

    def to_snapshot(self, fpath: str):
        """Saves the dataset to a csv file

        Args:
            fpath: fpath to the csv file
        """
        df = pd.DataFrame(self._frame_filenames, columns=["frame_filename"])
        df.to_csv(fpath, index=False)

    @classmethod
    def from_snapshot(
        cls, fpath: str, annotations_dir: str = cfg.ANNOTATIONS_DIR, image_size: ImageSize = DEFAULT_IMAGE_SIZE
    ) -> "AnimalDataset":
        """Loads the dataset from a csv file

        Args:
            fpath: fpath to the csv file
        """
        # load list of frame filenames from a csv
        df = pd.read_csv(fpath)
        frame_filenames = df["frame_filename"].apply(Path)
        return cls(annotations_dir=annotations_dir, image_size=image_size, frame_filenames=frame_filenames)

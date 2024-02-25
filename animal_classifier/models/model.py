import os

import torch
import torch.nn as nn

from animal_classifier.cfg import cfg
from animal_classifier.models.state_info import StateInfo


class AnimalNet(nn.Module):
    """
    Model used to map an image to a cat/dog label
    """

    def __init__(self):
        super(AnimalNet, self).__init__()
        # fully connected layer
        self.fc1 = nn.Linear(256, 1)
        self.initialize_weights()

    def forward(self, x):
        # Pass data through fc1
        x = self.fc1(x)
        output = torch.sigmoid(x)
        return output

    def initialize_weights(self):
        """
        Initialize weights
        """
        torch.manual_seed(0)
        if cfg.USE_CUDA:
            torch.cuda.manual_seed(0)
        nn.init.kaiming_uniform_(self.fc1.weight.data)
        nn.init.constant_(self.fc1.bias.data, 0.5)

    def store(self, model_filename: str, state_info: StateInfo, model_dir: str = cfg.MODEL_DIR) -> str:
        """
        Store model weights to local file

        Args:
            model_filename: filename to store weights in
            state_info: StateInfo containing additional info to store together with weights (epoch, loss and optimizer state)
            model_dir: path to folder where model will be stored locally

        Returns:
            model filepath where it was stored
        """
        # store weights and state info to local file
        os.system(f"mkdir -p {model_dir}")
        model_filepath = f"{model_dir}/{model_filename}"
        torch.save(
            {
                "epoch": state_info.epoch,
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": state_info.optimizer_state_dict,
                "loss": state_info.loss,
            },
            f"{model_filepath}",
        )
        return model_filepath

    def load(self, model_filename: str, model_dir: str = cfg.MODEL_DIR) -> StateInfo:
        """
        Load model weights from local file

        Args:
            model_filename: filename to load weights from
            model_dir: path to folder where model will be loaded from locally

        Returns:
            state_info: StateInfo containing additional info stored together with weights (epoch, loss and optimizer state)
        """
        model_filepath = f"{model_dir}/{model_filename}"
        checkpoint = torch.load(model_filepath)
        self.load_state_dict(checkpoint["model_state_dict"])
        return StateInfo(
            epoch=checkpoint["epoch"],
            optimizer_state_dict=checkpoint["optimizer_state_dict"],
            loss=checkpoint["loss"],
        )

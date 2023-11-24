from typing import Dict, List

import numpy as np
import torch
from ts.torch_handler.vision_handler import VisionHandler

from animal_classifier.cfg import cfg
from animal_classifier.datasets.dataset import AnimalDataset
from animal_classifier.models.model import AnimalNet
from animal_classifier.utils.enums import AnimalLabel


class AnimalHandler(VisionHandler):
    """
    Handler to load torchscript or eager mode [state_dict] models
    Also, provides handle method per torch serve custom model specification
    """

    def initialize(self, context):
        """Initialize function loads the model.pt file and initialized the model object.
           First try to load torchscript else load eager mode state_dict based model.

        Args:
            context (context): It is a JSON Object containing information
            pertaining to the model artifacts parameters.
        """
        properties = context.system_properties
        self.map_location = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(
            self.map_location + ":" + str(properties.get("gpu_id")) if torch.cuda.is_available() else self.map_location
        )
        self.manifest = context.manifest

        model_dir = properties.get("model_dir")
        model_filename = self.manifest["model"]["serializedFile"]

        self.model = AnimalNet()
        self.state_info = self.model.load(model_filename=model_filename, model_dir=model_dir)
        self.model.eval()
        self.model.to(self.device)

        self.initialized = True
        self.image_processing = lambda x: AnimalDataset.transform_input_frame(np.array(x))

    def postprocess(self, data: torch.Tensor) -> List[Dict]:
        """
        Post-process method to convert the output scores to animal labels

        Args:
            data: with model predictions

        Returns:
            List[Dict] with animal labels
        """
        response = []
        for score in data:
            response.append({"animal": AnimalLabel.from_score(score=float(score), threshold=cfg.THRESHOLD).name.lower()})
        return response

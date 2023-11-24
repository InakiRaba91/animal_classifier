from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from animal_classifier.api.handler import AnimalHandler
from animal_classifier.models import AnimalNet, StateInfo


class TestAnimalHandler:
    @pytest.fixture
    def context(self, model_fpath: Path) -> Mock:
        context = Mock()
        context.system_properties = {"model_dir": model_fpath.parent, "gpu_id": 0}
        context.manifest = {"model": {"serializedFile": model_fpath.name}}
        return context

    @pytest.fixture
    def handler(self, context: Mock):
        handler = AnimalHandler()
        return handler

    def test_initialize(self, handler: AnimalHandler, context: Mock, state_info: StateInfo, model_fpath: Path):
        handler.initialize(context)

        assert handler.map_location == "cuda" if torch.cuda.is_available() else "cpu"
        assert isinstance(handler.device, torch.device)
        assert handler.manifest == {"model": {"serializedFile": model_fpath.name}}
        assert isinstance(handler.model, AnimalNet)
        assert handler.state_info == state_info
        assert handler.initialized

    def test_postprocess(self, handler: AnimalHandler):
        model_output = torch.tensor([[0.1], [0.9]])  # replace with your actual model output
        result = handler.postprocess(model_output)

        # replace with your actual postprocess output
        expected_output = [{"animal": "cat"}, {"animal": "dog"}]

        assert result == expected_output

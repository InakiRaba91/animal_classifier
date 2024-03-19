from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from animal_classifier.api.model.handler import AnimalHandler
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
        scores = [0.1, 0.9]
        model_output = torch.tensor([[scores[0]], [scores[1]]])  # replace with your actual model output
        response = handler.postprocess(model_output)

        # replace with your actual postprocess output
        expected_response = [{"animal": "cat", "score": scores[0]}, {"animal": "dog", "score": scores[1]}]

        # assert
        for result, expected_result in zip(response, expected_response):
            assert result["animal"] == expected_result["animal"]
            np.testing.assert_almost_equal(result["score"], expected_result["score"])

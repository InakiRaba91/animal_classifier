from typing import Dict

import torch

from animal_classifier.models import AnimalNet, StateInfo


def compare_state_dicts(state_dict1: Dict, state_dict2: Dict):
    assert state_dict1.keys() == state_dict2.keys()
    equal_weights = []
    for key in state_dict1.keys():
        equal_weights.append(torch.equal(state_dict1[key], state_dict2[key]))
    return all(equal_weights)


class TestAnimalNet:
    def test_model_forward_pass(self, model: AnimalNet, tol: float = 1e-6):
        torch.manual_seed(0)
        item = torch.rand((1, 1, 1, 256))
        output = model(item)
        expected_output = 0.997576
        assert abs(float(output) - expected_output) < tol

    def test_store(self, model: AnimalNet, model_fpath: str, state_info: StateInfo, model_dir: str):
        model_filepath = model.store(
            model_filename=model_fpath.stem,
            state_info=state_info,
            model_dir=model_dir,
        )
        checkpoint = torch.load(model_filepath)
        assert compare_state_dicts(state_dict1=checkpoint["model_state_dict"], state_dict2=model.state_dict())

    def test_load(self, model: AnimalNet, state_info: StateInfo, model_dir: str, model_fpath: str):
        state_info_loaded = model.load(model_filename=model_fpath.name, model_dir=model_dir)
        assert state_info_loaded == state_info
        checkpoint = torch.load(model_fpath)
        assert compare_state_dicts(state_dict1=checkpoint["model_state_dict"], state_dict2=model.state_dict())

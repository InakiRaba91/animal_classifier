import numpy as np
import pytest

from animal_classifier.utils.enums import AnimalLabel


class TestAnimalLabel:
    @pytest.mark.parametrize("animal", ["DOG", "dog", "CAT", "cat"])
    def test_from_string(self, animal: str):
        label = AnimalLabel.from_string(animal=animal)
        assert label.name == animal.upper()

    @pytest.mark.parametrize("score", list(np.linspace(0, 1, 10)))
    def test_from_score(self, score):
        threshold = 0.5
        label = AnimalLabel.from_score(score=score, threshold=threshold)
        expected_label = AnimalLabel.CAT if score < threshold else AnimalLabel.DOG
        assert label == expected_label

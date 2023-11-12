from enum import Enum


class AnimalLabel(Enum):
    CAT = 0
    DOG = 1

    @classmethod
    def from_string(cls, animal: str):
        return cls.__members__[animal.upper()]

    @classmethod
    def from_score(cls, score: float, threshold: float = 0.5):
        if score < threshold:
            return cls.CAT
        return cls.DOG

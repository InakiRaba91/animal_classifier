from pydantic import BaseModel, field_validator

from animal_classifier.utils.enums import AnimalLabel


class AnimalResponse(BaseModel):
    """
    A pydantic model used to validate images stored in base64.
    """

    animal: str
    score: float = 0
    version: int = 0

    @field_validator("animal")
    def check_valid_animal(cls, animal: str):
        """
        Validator to check that the input string is one of the labels in AnimalLabel
        """
        assert animal in list(AnimalLabel.__members__.keys())
        return animal

    @field_validator("score")
    def check_valid_animalscore(cls, score: float):
        """
        Validator to check that the score is between 0 and 1
        """
        assert 0 <= score <= 1
        return score

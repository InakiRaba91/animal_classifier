from pydantic import BaseModel, NonNegativeInt, model_validator


class Config(BaseModel):
    #: Traffic percentage for the current version
    traffic: float
    #: Current version of the model
    current_version: NonNegativeInt
    #: Previous version of the model
    previous_version: NonNegativeInt

    @model_validator(mode="after")
    def check_current_version_greater_equal_previous(self) -> "Config":
        if self.current_version < self.previous_version:
            raise ValueError(
                f"Current version {self.current_version} is not greater or equal to previous version {self.previous_version}"
            )
        return self

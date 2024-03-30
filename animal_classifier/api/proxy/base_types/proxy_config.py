from pydantic import (
    BaseModel,
    Field,
    NonNegativeInt,
    PrivateAttr,
    computed_field,
    model_validator,
)

from animal_classifier.cfg import cfg


class ProxyConfig(BaseModel):
    #: Current version of the model
    current_version: NonNegativeInt = Field(..., allow_mutation=False)  # type: ignore
    #: Previous version of the model
    previous_version: NonNegativeInt = Field(..., allow_mutation=False)  # type: ignore
    #: Index of the traffic percentage to use
    _idx_traffic: int = PrivateAttr(default=0)

    class Config:
        validate_assignment = True

    @model_validator(mode="after")  # type: ignore
    def check_current_version_greater_equal_previous(self):
        if self.current_version < self.previous_version:
            raise ValueError(
                f"Current version {self.current_version} is not greater or equal to previous version {self.previous_version}"
            )
        return self

    @computed_field
    def traffic(self) -> float:
        """
        Percentage of traffic to send to the current model version."""
        return cfg.TRAFFIC_STEPS[self._idx_traffic]

    @traffic.setter  # type: ignore
    def set_traffic(self, value: float) -> None:
        assert 0, "Cannot set traffic percentage directly, use `idx_traffic` instead"

    def increase_traffic(self) -> None:
        """Increase the traffic percentage by one step."""
        self._idx_traffic = min(self._idx_traffic + 1, len(cfg.TRAFFIC_STEPS) - 1)

    def decrease_traffic(self) -> None:
        """Decrease the traffic percentage by one step."""
        self._idx_traffic = max(self._idx_traffic - 1, 0)

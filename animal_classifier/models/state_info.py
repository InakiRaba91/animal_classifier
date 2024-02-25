from typing import Dict

from pydantic import BaseModel


class StateInfo(BaseModel):
    """
    Summary of state info during model training
    """

    epoch: int
    optimizer_state_dict: Dict
    loss: float

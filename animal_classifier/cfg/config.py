"""Config variables used internally within the service

  It will read environment variables from the .env file in the root
  directory. If it doesn't exist, it will get them from AWS. Otherwise
  they will be grabbed from the environment.

  Attributes:
      DATA_DIR: path to the folder where dataset is stored in.
      MODEL_DIR: path to the folder where trained models will be stored in.
      FRAMES_FOLDER: path to the directory containing the frames in S3
      ANNOTATIONS_FOLDER: path to the directory containing the annotations in S3
      MODEL_FOLDER: path to the directory contained weights for trained models in S3
      DEVICE: CPU/GPU
"""
from enum import Enum
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# get location of current file
ROOT_PATH = Path(__file__).parent.parent.parent.resolve()


class EnvState(Enum):
    DEV = "DEV"
    STAGING = "STAGING"
    PROD = "PROD"


class GlobalConfig(BaseSettings):
    """
    Global configuration, used by the dev, staging and prod.

    Attributes can be set using a .env file located in the package root directory or using env variables.
    Attributes are loaded using the following priority:
    Env variables > .env file > defaults
    """

    #: The env state (dev, staging or prod)
    ENV_STATE: EnvState = EnvState.DEV

    # Default settings
    # The filepaths will need to be updated when implemented as kubeflow components / pipelines.
    MODEL_DIR: str = (ROOT_PATH / "models/cats_and_dogs/").as_posix()
    ANNOTATIONS_DIR: str = (ROOT_PATH / "data/cats_and_dogs/annotations/").as_posix()
    FRAMES_DIR: str = (ROOT_PATH / "data/cats_and_dogs/frames/").as_posix()
    USE_CUDA: bool = False
    DEVICE: "str" = "cpu"
    TRAIN_FRAC: float = 0.8
    VAL_FRAC: float = 0.1
    TEST_FRAC: float = 0.1
    BATCH_SIZE: int = 1
    MIN_ACCURACY_VALIDATION: float = 0.0
    LEARNING_RATE: float = 0.0003
    WEIGHT_DECAY: float = 0.0
    NUM_EPOCHS: int = 3
    THRESHOLD: float = 0.5
    SUMMARY_FILE: str = (ROOT_PATH / "info/last_training.json").as_posix()

    # pydantic config
    model_config = SettingsConfigDict(
        env_file=".env",
        frozen=True,
        extra="allow",
    )


class DevConfig(GlobalConfig):
    """
    Dev Config
    For local development
    """

    @field_validator("ENV_STATE")
    def assert_correct_env_state(cls, v):
        assert v == EnvState.DEV
        return v


class StagingConfig(GlobalConfig):
    """Staging Configuration"""

    @field_validator("ENV_STATE")
    def assert_correct_env_state(cls, v):
        assert v == EnvState.STAGING
        return v


class ProdConfig(GlobalConfig):
    """Prod Configuration"""

    @field_validator("ENV_STATE")
    def assert_correct_env_state(cls, v):
        assert v == EnvState.PROD
        return v


class FactoryConfig:
    """Returns a config instance dependending on the ENV_STATE variable.

    Attributes can be set in the .env file, or as env variables.
    Attributes are loaded using the following priority:
        Env variables > .env file > defaults

    Example usage:
        ```bash
        export ENV_STATE=serving
        ```
        ```python
        cfg = FactoryConfig(GlobalConfig().ENV_STATE)()  # loads prod config
        ```
    """

    def __init__(self, env_state: EnvState):
        self.env_state = env_state

    def __call__(self):
        if self.env_state == EnvState.STAGING:
            return StagingConfig()
        elif self.env_state == EnvState.PROD:
            return ProdConfig()
        else:
            return DevConfig()

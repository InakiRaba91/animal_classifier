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
from typing import Optional

import torch
from pydantic import BaseSettings, Field, validator


class EnvState(Enum):
    DEV = "DEV"
    TEST = "TEST"
    PROD = "PROD"


class GlobalConfig(BaseSettings):
    """
    Global configuration, used by the dev, test and prod.

    Attributes can be set using a .env file located in the package root directory or using env variables.
    Attributes are loaded using the following priority:
    Env variables > .env file > defaults
    """

    #: The env state (dev, test or prod)
    ENV_STATE: EnvState = Field(EnvState.DEV, env="ENV_STATE")

    # Default settings
    # The filepaths will need to be updated when implemented as kubeflow components / pipelines.
    MODEL_DIR: str = "./models/cats_and_dogs/"
    ANNOTATIONS_DIR = "./data/cats_and_dogs/annotations/"
    FRAMES_DIR = "./data/cats_and_dogs/frames/"
    USE_CUDA: bool = False
    DEVICE: Optional[torch.device]
    TRAIN_FRAC: float = 0.8
    VAL_FRAC: float = 0.1
    TEST_FRAC: float = 0.1

    class Config:
        env_file: str = ".env"
        env_file_encoding: str = "utf-8"

    @validator("ENV_STATE")
    def assert_correct_env_state(cls, v):
        string_env_state = cls.__name__[:-6].upper()
        if string_env_state != "GLOBAL":
            assert v == EnvState(string_env_state)
        return v


class DevConfig(GlobalConfig):
    """Dev Configuration
    vars set in .env without prefix.
    """

    class Config:
        env_prefix = ""


class TestConfig(GlobalConfig):
    """Test Configuration
    Set using the TEST prefix.
    """

    class Config:
        env_prefix = "TEST_"


class ProdConfig(GlobalConfig):
    """Prod Configuration
    Set using the PROD prefix.
    """

    class Config:
        env_prefix = "PROD_"


class FactoryConfig:
    """Returns a TestConfig instance dependending on the ENV_STATE variable.

    Attributes can be set in the .env file, or as env variables.
    To set the test or prod TestConfig, you must define the vars with the prefix (followed by an underscore) TEST or PROD
    respectively.
    Setting the ENV_STATE to 'test', will load the TEST attributes (and similarly for prod).

    Attributes are loaded using the following priority:
        Env variables > .env file > defaults

    Example usage:
        ```bash
        export ENV_STATE=prod
        export PROD_MODEL=R50_FPN_3x
        ```
        ```python
        cfg = FactoryConfig(GlobalConfig().ENV_STATE)()  # loads prod TestConfig with model as ModelYaml.R50_FPN_3x
        ```
    """

    def __init__(self, env_state: EnvState):
        self.env_state = env_state

    def __call__(self):
        if self.env_state == EnvState.TEST:
            return TestConfig()
        elif self.env_state == EnvState.DEV:
            return DevConfig()
        else:
            return ProdConfig()

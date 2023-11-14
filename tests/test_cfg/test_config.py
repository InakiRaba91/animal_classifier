from typing import Union

import pytest
from pydantic import BaseModel, ValidationError

from animal_classifier.cfg.config import (
    DevConfig,
    EnvState,
    FactoryConfig,
    GlobalConfig,
    StagingConfig,
    ProdConfig,
)


# TOOLS
class TmpEnvVar(BaseModel):
    """
    Used for testing if changing an environment variable changes the corresponding attribute of the cfg
    """

    attribute_name: str
    key: str
    value: Union[int, float, str]


def validation_error_if_env_state_not_as_expected(cfg, monkeypatch):
    """
    Iterate through each available EnvState and set the ENV_STATE env var to be EnvState.value.
    If we try to initialise a cfg class with the incorrect env_state set, we should ge a ValidationError.
    e.g.,
    ENV_STATE = dev
    DevConfig() -> OK
    KubeflowConfig() -> ValidationError
    Args:
        cfg: an instance of the config to test
        monkeypatch: monkeypatch to update the ENV_STATE env var
    """
    for env_state in EnvState:
        if env_state == cfg.ENV_STATE:
            continue

        monkeypatch.setenv("ENV_STATE", env_state.value)
        with pytest.raises(ValidationError):
            cfg.__class__()


class TestGlobalConfig:
    @pytest.mark.parametrize(
        "env_state",
        [EnvState.DEV, EnvState.STAGING, EnvState.PROD],
    )
    def test_global_config(self, monkeypatch, env_state):
        """
        Check that altering the ENV_STATE environment variable affects the GlobalConfig ENV_STATE attribute
        """
        # SETUP
        monkeypatch.setenv("ENV_STATE", env_state.value)

        # EXECUTE
        cfg = GlobalConfig()

        # ASSERT
        assert cfg.ENV_STATE == env_state


@pytest.mark.parametrize(
    "env_state, env_config",
    [
        (EnvState.DEV, DevConfig),
        (EnvState.STAGING, StagingConfig),
        (EnvState.PROD, ProdConfig),
    ],
)
class TestEnvConfig:
    """
    Test the generic validators that are present for each {ENV_STATE}Config class.
    (1) test_prefix. Ensure that for changing an env var with a specified prefix has the desired effect on the config
        attributes.
    (2) test_assert_correct_env_state. Trial different env states and assert that a ValidationError is raised if the
        ENV_STATE does not match the config class. e.g. ENV_STATE = kubeflow, DevConfig() -> ValidationError
    """

    @pytest.fixture(autouse=True)
    def set_env_state(self, monkeypatch, env_state):
        """set env state to desired one for all tests in this class"""
        monkeypatch.setenv("ENV_STATE", env_state.value)

    def test_assert_correct_env_state(self, monkeypatch, env_config):
        """Env state must be set one"""
        validation_error_if_env_state_not_as_expected(env_config(), monkeypatch)


class TestFactoryConfig:
    @pytest.mark.parametrize(
        "env_state, expected_cfg_type",
        [
            (EnvState.DEV, DevConfig),
            (EnvState.STAGING, StagingConfig),
            (EnvState.PROD, ProdConfig),
        ],
    )
    def test_factory_config(self, monkeypatch, env_state, expected_cfg_type):
        """
        For each env state, the factory should return the expected config type
        """
        monkeypatch.setenv("ENV_STATE", env_state.value)
        cfg = FactoryConfig(GlobalConfig().ENV_STATE)()
        assert isinstance(cfg, expected_cfg_type)
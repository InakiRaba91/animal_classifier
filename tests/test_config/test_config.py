from typing import Union

import pytest
from pydantic import BaseModel, ValidationError

from animal_classifier.cfg.config import (
    DevConfig,
    EnvState,
    FactoryConfig,
    GlobalConfig,
    TestConfig,
)
from animal_classifier.utils.testing import does_not_raise


# TOOLS
class TmpEnvVar(BaseModel):
    """
    Used for testing if changing an environment variable changes the corresponding attribute of the cfg
    """

    attribute_name: str
    key: str
    value: Union[str, float, int]


@pytest.fixture
def set_config_env_vars(monkeypatch):
    """
    Set environment variables for different env states in order to test that the prefixes work as expected.
    For each env state we define a list of TmpEnvVar objects. For each object, we monkeypatch the environment
    to set the value TmpEnvVar.key = TmpEnvVar.value.

    During tests for each EnvState, we can assert that the expected attribute has the desired value. i.e.
    DevConfig.{TmpEnvVar.attribute_name} == TmpEnvVar.value
    """
    # define env vars
    prefix_test_cfg = {
        EnvState.DEV: [TmpEnvVar(attribute_name="DATA_DIR", key="DATA_DIR", value="env_data_dir")],
        EnvState.TEST: [TmpEnvVar(attribute_name="DATA_DIR", key="TEST_DATA_DIR", value="test_data_dir")],
    }

    # set env vars
    for env_state, env_var in prefix_test_cfg.items():
        for env_var_item in env_var:
            monkeypatch.setenv(env_var_item.key, env_var_item.value)
    return prefix_test_cfg


def assert_env_vars_altered_cfg(cfg, set_config_env_vars):
    """
    We expect that changing environment variables will alter the cfg.
    This function will assert that the altered environment variables set in set_config_env_vars match the attributes in
    the cfg instance.
    """
    for env_var_item in set_config_env_vars[cfg.ENV_STATE]:  # get the altered env vars for the given env state
        actual = getattr(cfg, env_var_item.attribute_name)  # the actual attribute value
        expected = type(actual)(env_var_item.value)  # type conversion for the set env var str
        assert actual == expected


def validation_error_if_env_state_not_as_expected(cfg, monkeypatch):
    """
    Iterate through each available EnvState and set the ENV_STATE env var to be EnvState.value.
    If we try to initialise a cfg class with the incorrect env_state set, we should ge a ValidationError.

    e.g.,
    ENV_STATE = test
    DevConfig() -> ValidationError
    TestConfig() -> OK

    Args:
        cfg: an instance of the test_config to test
        monkeypatch: monkeypatch to update the ENV_STATE env var
    """
    for env_state in EnvState:
        monkeypatch.setenv("ENV_STATE", env_state.value)
        with pytest.raises(ValidationError) if env_state != cfg.ENV_STATE else does_not_raise():
            cfg.__class__()


# TESTS
class TestGlobalConfig:
    @pytest.mark.parametrize(
        "env_state",
        [EnvState.DEV, EnvState.PROD, EnvState.TEST],
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
        (EnvState.TEST, TestConfig),
    ],
)
class TestEnvConfig:
    """
    Test the generic validators that are present for each {ENV_STATE}Config class.
    (1) test_prefix. Ensure that for changing an env var with a specified prefix has the desired effect on the test_config
        attributes.
    (2) test_assert_correct_env_state. Trial different env states and assert that a ValidationError is raised if the
        ENV_STATE does not match the test_config class. e.g. ENV_STATE = prod, TestConfig() -> ValidationError
    """

    @pytest.fixture(autouse=True)
    def set_env_state(self, monkeypatch, env_state):
        """set env state to desired one for all tests in this class"""
        monkeypatch.setenv("ENV_STATE", env_state.value)

    def test_prefix(self, set_config_env_vars, env_config):
        """Setting the env var with the prefix associated to the type of env should work.
        We assert that the environment variables set in set_config_env_vars math the value of the attribute that it is
        expected to change.
        """
        assert_env_vars_altered_cfg(cfg=env_config(), set_config_env_vars=set_config_env_vars)

    def test_assert_correct_env_state(self, monkeypatch, env_config):
        """Env state must be set one"""
        validation_error_if_env_state_not_as_expected(env_config(), monkeypatch)


class TestFactoryConfig:
    @pytest.mark.parametrize("env_state, expected_cfg_type", [(EnvState.DEV, DevConfig), (EnvState.TEST, TestConfig)])
    def test_factory_config(self, monkeypatch, env_state, expected_cfg_type):
        """
        For each env state, the factory should return the expected test_config type
        """
        monkeypatch.setenv("ENV_STATE", env_state.value)
        cfg = FactoryConfig(GlobalConfig().ENV_STATE)()
        assert isinstance(cfg, expected_cfg_type)

import pytest

from animal_classifier.api.proxy.base_types.proxy_config import ProxyConfig
from animal_classifier.cfg import cfg


class TestConfig:
    @pytest.mark.parametrize("current_version, previous_version", [(1, 1), (2, 1)])
    def test_config_instantiation(self, current_version, previous_version):
        config = ProxyConfig(current_version=current_version, previous_version=previous_version)
        assert config.current_version == current_version
        assert config.previous_version == previous_version
        assert config.traffic == cfg.TRAFFIC_STEPS[0]

    def test_config_version_validation_fails(self):
        with pytest.raises(ValueError):
            ProxyConfig(current_version=1, previous_version=2)

    def test_cannot_set_traffic_directly(self):
        config = ProxyConfig(current_version=2, previous_version=1)
        with pytest.raises(AssertionError):
            config.traffic = 0.5

    def test_increase_and_decrease_traffic(self):
        config = ProxyConfig(current_version=2, previous_version=1)
        config.increase_traffic()
        assert config.traffic == cfg.TRAFFIC_STEPS[1]
        config.decrease_traffic()
        assert config.traffic == cfg.TRAFFIC_STEPS[0]

from unittest import mock
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from animal_classifier.api.proxy.service import ProxyConfig, app
from animal_classifier.api.proxy.utils.process_request import (
    get_model_version,
    process_request,
)
from animal_classifier.cfg.config import DevConfig

client = TestClient(app)


@pytest.mark.parametrize("traffic", [0.1, 0.9])
class TestHandler:
    def test_get_model_version_current(self, traffic: float):
        # setup
        previous_version, current_version = 1, 2
        new_cfg = DevConfig(TRAFFIC_STEPS=(traffic,))

        # execute
        current_versions, previous_versions = 0, 0
        with patch("animal_classifier.api.proxy.base_types.proxy_config.cfg", new_cfg):
            config = ProxyConfig(current_version=current_version, previous_version=previous_version)
            for _ in range(10):
                version = get_model_version(config)
                if version == current_version:
                    current_versions += 1
                else:
                    previous_versions += 1

        # assert
        if traffic < 0.5:
            assert current_versions < previous_versions
        else:
            assert current_versions > previous_versions

    @mock.patch("requests.post", return_value=mock.Mock(json=mock.Mock(return_value={"animal": "dog", "score": 1.0})))
    def test_process_request(self, mocked_post, traffic: float):
        # setup
        base_64_str = ""
        previous_version, current_version = 1, 2
        new_cfg = DevConfig(TRAFFIC_STEPS=(traffic,))

        # execute
        current_versions, previous_versions = 0, 0
        with patch("animal_classifier.api.proxy.base_types.proxy_config.cfg", new_cfg):
            config = ProxyConfig(current_version=current_version, previous_version=previous_version)
            for _ in range(10):
                response = process_request(base_64_str=base_64_str, config=config)
                if response.version == current_version:
                    current_versions += 1
                else:
                    previous_versions += 1

        # assert
        if traffic < 0.5:
            assert current_versions < previous_versions
        else:
            assert current_versions > previous_versions

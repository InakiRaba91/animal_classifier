import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from animal_classifier.api.proxy.service import CONFIG_FILE, Config, app

client = TestClient(app)


@pytest.fixture
def config() -> Config:
    return Config(traffic=0.9, current_version=3, previous_version=2)


@pytest.fixture
def config_file(tmpdir_factory, config: Config) -> Path:
    config_folder = tmpdir_factory.mktemp("config")
    config_file = config_folder / "config.json"
    with open(config_file, "w") as f:
        json.dump(config.model_dump(), f)
    return config_file


def test_healthz_endpoint():
    # setup
    response_health = {}

    # execute
    response = client.get("/health")

    # assert
    assert response.status_code == 200
    assert response.json() == response_health


def test_root_endpoint():
    # setup
    response_service = {"message": "Welcome to the Animal Classifier API!"}

    # execute
    response = client.get("/")

    # assert
    assert response.status_code == 200
    assert response.json() == response_service


def test_get_config():
    # setup
    with open(CONFIG_FILE) as f:
        data = f.read()
    config = Config.model_validate_json(data)

    # execute
    response = client.get("/api/get-config")

    # assert
    assert response.status_code == 200
    assert response.json() == config.model_dump()


def test_update_config(config: Config, config_file: Path):
    # execute
    with patch("animal_classifier.api.proxy.service.CONFIG_FILE", str(config_file)):
        response_update = client.put("/api/update-config")
        response_get = client.get("/api/get-config")

    # assert
    assert response_update.status_code == 200
    assert response_get.status_code == 200
    assert response_get.json() == config.model_dump()

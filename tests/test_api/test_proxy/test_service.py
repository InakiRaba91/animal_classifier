import json
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from animal_classifier.api.proxy.base_types.animal_response import AnimalResponse
from animal_classifier.api.proxy.base_types.base64_image import Base64ImageType
from animal_classifier.api.proxy.service import CONFIG_FILE, ProxyConfig, app

client = TestClient(app)


@pytest.fixture
def config():
    with open(CONFIG_FILE) as f:
        data = f.read()
    return ProxyConfig.model_validate_json(data)


@pytest.fixture
def new_prediction_summary(config):
    return {
        str(config.current_version): {"correct": 0, "total": 0},
        str(config.previous_version): {"correct": 0, "total": 0},
    }


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


def test_get_config(config):
    # execute
    response = client.get("/api/get-config")

    # assert
    assert response.status_code == 200
    assert response.json() == config.model_dump()


def test_get_prediction_summary(new_prediction_summary):
    # execute
    response = client.get("/api/get-prediction-summary")

    # assert
    assert response.status_code == 200
    assert response.json() == new_prediction_summary


def test_update_prediction_summary(config):
    # setup
    prediction_summary_after_manual = {
        str(config.previous_version): {"correct": 0, "total": 0},
        str(config.current_version): {"correct": 0, "total": 1},
    }
    prediction_summary_after_auto = {
        str(config.previous_version): {"correct": 0, "total": 0},
        str(config.current_version): {"correct": 1, "total": 2},
    }

    # execute
    response = client.post(f"/api/update-prediction-summary?version={config.current_version}&label_source=MANUAL")
    response_get_prediction_summary1 = client.get("/api/get-prediction-summary")
    response = client.post(f"/api/update-prediction-summary?version={config.current_version}&label_source=AUTO")
    response_get_prediction_summary2 = client.get("/api/get-prediction-summary")

    # assert
    assert response.status_code == 200
    assert response_get_prediction_summary1.status_code == 200
    assert response_get_prediction_summary1.json() == prediction_summary_after_manual
    assert response_get_prediction_summary2.status_code == 200
    assert response_get_prediction_summary2.json() == prediction_summary_after_auto


def test_reload_config(tmpdir_factory, config, new_prediction_summary):
    # setup
    new_config = ProxyConfig(traffic=0.9, current_version=3, previous_version=2)
    config_folder = tmpdir_factory.mktemp("config")
    config_file = config_folder / "config.json"
    with open(config_file, "w") as f:
        json.dump(new_config.model_dump(), f)
    new_prediction_summary = {
        str(new_config.current_version): {"correct": 0, "total": 0},
        str(new_config.previous_version): {"correct": 0, "total": 0},
    }

    # execute
    response_update1 = client.post(f"/api/update-prediction-summary?version={config.current_version}&label_source=MANUAL")
    response_update2 = client.post(f"/api/update-prediction-summary?version={config.current_version}&label_source=AUTO")
    with patch("animal_classifier.api.proxy.service.CONFIG_FILE", str(config_file)):
        response_reload = client.put("/api/reload-config")
        response_get_config = client.get("/api/get-config")
        response_get_prediction_summary = client.get("/api/get-prediction-summary")

    # assert
    assert response_update1.status_code == 200
    assert response_update2.status_code == 200
    assert response_reload.status_code == 200
    assert response_get_config.status_code == 200
    assert response_get_prediction_summary.status_code == 200
    assert response_get_config.json() == new_config.model_dump()
    assert response_get_prediction_summary.json() == new_prediction_summary


@patch("requests.post", return_value=Mock(json=Mock(return_value={"animal": "dog", "score": 1.0})))
def test_predict(mocked_post):
    # setup
    image = Base64ImageType(base_64_str="img")

    # execute
    response = client.post("/predict", json=image.model_dump())

    # assert
    assert response.status_code == 200
    assert mocked_post.call_count == 1
    assert AnimalResponse(**response.json())


@patch("requests.post", return_value=Mock(json=Mock(return_value={"animal": "dog", "score": 1.0})))
def test_upload(mocked_post):
    # setup
    image = Base64ImageType(base_64_str="img")

    # execute
    response = client.post("/upload", files={"file": ("test_image.jpg", image.base_64_str, "image/jpg")})

    # assert
    assert response.status_code == 200
    assert mocked_post.call_count == 1
    assert AnimalResponse(**response.json())

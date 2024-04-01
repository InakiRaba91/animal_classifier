import json
from importlib import reload
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

import animal_classifier.api.proxy.service as service
from animal_classifier.api.proxy.base_types.animal_response import AnimalResponse
from animal_classifier.api.proxy.base_types.base64_image import Base64ImageType
from animal_classifier.api.proxy.service import CONFIG_FILE, ProxyConfig, app

client = TestClient(app)


def get_mocked_config_file(current_version, previous_version, tmpdir_factory):
    config = ProxyConfig(current_version=current_version, previous_version=previous_version)
    config_folder = tmpdir_factory.mktemp("config")
    config_file = config_folder / "config.json"
    with open(config_file, "w") as f:
        json.dump(config.model_dump(), f)
    return config, config_file


def get_initial_prediction_summary(current_version, previous_version):
    return {
        str(current_version): {"correct": 0, "total": 0},
        str(previous_version): {"correct": 0, "total": 0},
    }


@pytest.fixture
def base_config() -> ProxyConfig:
    with open(CONFIG_FILE) as f:
        data = f.read()
    return ProxyConfig.model_validate_json(data)


@pytest.fixture
def base_prediction_summary(base_config: ProxyConfig):
    return get_initial_prediction_summary(
        current_version=base_config.current_version, previous_version=base_config.previous_version
    )


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


def test_get_config(base_config):
    # setup
    reload(service)
    client = TestClient(service.app)

    # execute
    response = client.get("/api/get-config")

    # assert
    assert response.status_code == 200
    assert response.json() == base_config.model_dump()


def test_get_prediction_summary(base_prediction_summary):
    # setup
    reload(service)
    client = TestClient(service.app)

    # execute
    response = client.get("/api/get-prediction-summary")

    # assert
    assert response.status_code == 200
    assert response.json() == base_prediction_summary


def test_reload_config(base_config: ProxyConfig, tmpdir_factory):
    # setup
    reload(service)
    client = TestClient(service.app)
    previous_version, current_version = base_config.current_version + 1, base_config.current_version + 2
    new_config, new_config_file = get_mocked_config_file(
        current_version=current_version, previous_version=previous_version, tmpdir_factory=tmpdir_factory
    )
    new_prediction_summary = get_initial_prediction_summary(current_version=current_version, previous_version=previous_version)

    # execute
    with patch("animal_classifier.api.proxy.service.CONFIG_FILE", str(new_config_file)):
        with patch.object(
            ProxyConfig, "model_validate_json", wraps=ProxyConfig.model_validate_json
        ) as mocked_model_validate_json:
            response_reload = client.put("/api/reload-config")
    response_get_config = client.get("/api/get-config")
    response_get_prediction_summary = client.get("/api/get-prediction-summary")

    # assert
    assert mocked_model_validate_json.call_count == 1
    assert json.loads(mocked_model_validate_json.call_args[0][0]) == new_config.model_dump()
    assert response_reload.status_code == 200
    assert response_get_config.status_code == 200
    assert response_get_prediction_summary.status_code == 200
    assert response_get_config.json() == new_config.model_dump()
    assert response_get_prediction_summary.json() == new_prediction_summary


def test_reload_config_after_updates(tmpdir_factory):
    # setup
    reload(service)
    client = TestClient(service.app)
    previous_version, current_version, new_version = 1, 2, 3
    config, config_file = get_mocked_config_file(
        current_version=current_version, previous_version=previous_version, tmpdir_factory=tmpdir_factory
    )
    new_config, new_config_file = get_mocked_config_file(
        current_version=new_version, previous_version=current_version, tmpdir_factory=tmpdir_factory
    )
    new_prediction_summary = get_initial_prediction_summary(current_version=new_version, previous_version=current_version)

    # execute
    with patch("animal_classifier.api.proxy.service.CONFIG_FILE", str(config_file)):
        response_reload1 = client.put("/api/reload-config")
        response_update1 = client.post(f"/api/update-prediction-summary?version={config.current_version}&label_source=MANUAL")
        response_update2 = client.post(f"/api/update-prediction-summary?version={config.current_version}&label_source=AUTO")
    with patch("animal_classifier.api.proxy.service.CONFIG_FILE", str(new_config_file)):
        response_reload2 = client.put("/api/reload-config")
        response_get_config = client.get("/api/get-config")
        response_get_prediction_summary = client.get("/api/get-prediction-summary")

    # assert
    assert response_reload1.status_code == 200
    assert response_update1.status_code == 200
    assert response_update2.status_code == 200
    assert response_reload2.status_code == 200
    assert response_get_config.status_code == 200
    assert response_get_prediction_summary.status_code == 200
    assert response_get_config.json() == new_config.model_dump()
    assert response_get_prediction_summary.json() == new_prediction_summary


def test_update_prediction_summary_fails(base_config):
    # setup
    reload(service)
    client = TestClient(service.app)
    new_version = base_config.current_version + 1

    # execute
    response = client.post(f"/api/update-prediction-summary?version={new_version}&label_source=MANUAL")

    # assert
    assert response.status_code == 422


def test_update_prediction_summary(base_config: ProxyConfig):
    # setup
    reload(service)
    client = TestClient(service.app)

    prediction_summary_after_manual = {
        str(base_config.previous_version): {"correct": 0, "total": 0},
        str(base_config.current_version): {"correct": 0, "total": 1},
    }
    prediction_summary_after_auto = {
        str(base_config.previous_version): {"correct": 0, "total": 0},
        str(base_config.current_version): {"correct": 1, "total": 2},
    }

    # execute
    with patch("animal_classifier.api.proxy.service.update_traffic", return_value=base_config) as mocked_update_traffic:
        response = client.post(f"/api/update-prediction-summary?version={base_config.current_version}&label_source=MANUAL")
        response_get_prediction_summary1 = client.get("/api/get-prediction-summary")
        response = client.post(f"/api/update-prediction-summary?version={base_config.current_version}&label_source=AUTO")
        response_get_prediction_summary2 = client.get("/api/get-prediction-summary")

    # assert
    assert response_get_prediction_summary1.status_code == 200
    assert response_get_prediction_summary1.json() == prediction_summary_after_manual
    assert response.status_code == 200
    assert response_get_prediction_summary2.status_code == 200
    assert response_get_prediction_summary2.json() == prediction_summary_after_auto
    assert mocked_update_traffic.call_count == 2


@patch("requests.post", return_value=Mock(json=Mock(return_value={"animal": "dog", "score": 1.0})))
def test_predict(mocked_post):
    # setup
    reload(service)
    client = TestClient(service.app)
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
    reload(service)
    client = TestClient(service.app)
    image = Base64ImageType(base_64_str="img")

    # execute
    response = client.post("/upload", files={"file": ("test_image.jpg", image.base_64_str, "image/jpg")})

    # assert
    assert response.status_code == 200
    assert mocked_post.call_count == 1
    assert AnimalResponse(**response.json())

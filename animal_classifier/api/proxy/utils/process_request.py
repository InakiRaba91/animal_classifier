import json
import random

import requests

from animal_classifier.api.proxy.base_types.animal_response import AnimalResponse
from animal_classifier.api.proxy.base_types.proxy_config import ProxyConfig


def get_model_version(config: ProxyConfig) -> int:
    """
    Get the version of the model to use for prediction.

    Args:
        config: The configuration object.

    Returns:
        The version of the model to use for prediction.
    """
    if random.random() < config.traffic:  # type: ignore
        return config.current_version
    return config.previous_version


def process_request(base_64_str: str, config: ProxyConfig) -> AnimalResponse:
    """
    Predict the animal in the image and return the prediction.

    Args:
        base_64_str: The base64 encoded image.

    Returns:
        The prediction of the animal in the image.
    """
    version = get_model_version(config)
    data = json.dumps(base_64_str)
    response = requests.post(
        f"http://torchserve:8080/predictions/animal/{version}.0", data=data, headers={"Content-Type": "application/json"}
    )
    prediction = response.json()
    return AnimalResponse(animal=prediction["animal"].upper(), score=prediction["score"], version=version)

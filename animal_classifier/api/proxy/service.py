import base64
import json

import requests

# from animal_classifier.utils.enums import AnimalLabel
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi_health import health

from animal_classifier.api.proxy.base_types.animal_response import AnimalResponse
from animal_classifier.api.proxy.base_types.base64_image import Base64ImageType
from animal_classifier.api.proxy.base_types.config import Config

# Load config on start-up
CONFIG_FILE = "config/config.json"
with open(CONFIG_FILE) as f:
    data = f.read()
config = Config.model_validate_json(data)


# Launch FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_api_route("/health", health([]), description="Used simply to ensure the API is up and responding")


# Exception handling
@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    """Handle all exceptions by returning a JSON response with the exception message."""
    return JSONResponse(
        status_code=500,
        content={"message": str(exc).replace('"', "'")},
    )


@app.get("/")
def root():
    """Root endpoint that returns a welcome message."""
    return {"message": "Welcome to the Animal Classifier API!"}


@app.get("/api/get-config", include_in_schema=False)
def get_config() -> Config:
    """Return the current configuration."""
    return config


@app.put("/api/update-config", include_in_schema=False)
def update_config():
    """Update the configuration."""
    global config
    with open(CONFIG_FILE) as f:
        data = f.read()
    config = Config.model_validate_json(data)


@app.post("/predict")
async def predict(image: Base64ImageType) -> AnimalResponse:
    """
    Predict the animal in the image and return the prediction.

    Args:
        image: The image in base64 format.

    Returns:
        The prediction of the animal in the image.
    """
    data = json.dumps(image.base_64_str)
    response = requests.post(
        "http://torchserve:8080/predictions/animal", data=data, headers={"Content-Type": "application/json"}
    )
    prediction = response.json()
    return AnimalResponse(animal=prediction["animal"].upper(), score=prediction["score"], version=config.current_version)


@app.post("/upload")
async def upload(file: UploadFile = File(...)) -> AnimalResponse:
    """
    Predict the animal in the image and return the prediction.

    Args:
        file: The image file.

    Returns:
        The prediction of the animal in the image.
    """
    contents = await file.read()
    base_64_str = base64.b64encode(contents).decode("utf-8")
    data = json.dumps(base_64_str)
    response = requests.post(
        "http://torchserve:8080/predictions/animal", data=data, headers={"Content-Type": "application/json"}
    )
    prediction = response.json()  # adjust this line based on the actual response structure
    return AnimalResponse(animal=prediction["animal"].upper(), score=prediction["score"], version=config.current_version)

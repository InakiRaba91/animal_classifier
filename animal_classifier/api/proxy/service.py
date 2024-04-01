import base64

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi_health import health

from animal_classifier.api.proxy.base_types.animal_response import AnimalResponse
from animal_classifier.api.proxy.base_types.base64_image import Base64ImageType
from animal_classifier.api.proxy.base_types.label_source import LabelSource
from animal_classifier.api.proxy.base_types.proxy_config import ProxyConfig
from animal_classifier.api.proxy.utils import process_request, update_traffic

# Load config on start-up
CONFIG_FILE = "config/config.json"
with open(CONFIG_FILE) as f:
    data = f.read()
config = ProxyConfig.model_validate_json(data)
prediction_summary = {
    config.current_version: {"correct": 0, "total": 0},
    config.previous_version: {"correct": 0, "total": 0},
}

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
def get_config() -> ProxyConfig:
    """Return the current configuration."""
    return config


@app.get("/api/get-prediction-summary", include_in_schema=False)
def get_prediction_summary():
    """Return the prediction summary."""
    global prediction_summary
    return prediction_summary


@app.post("/api/update-prediction-summary", include_in_schema=False)
def update_prediction_summary(version: int, label_source: LabelSource):
    """Update the prediction summary."""
    global prediction_summary
    if version not in prediction_summary:
        raise HTTPException(
            status_code=422,
            detail=f"Version {version} is currently not deployed. Deployed versions are {list(prediction_summary.keys())}.",
        )
    prediction_summary[version]["total"] += 1
    if label_source == LabelSource.AUTO:
        prediction_summary[version]["correct"] += 1

    # update traffic if necessary
    global config
    config = update_traffic(config=config, prediction_summary=prediction_summary)


@app.put("/api/reload-config", include_in_schema=False)
def reload_config():
    """Update the configuration."""
    global config
    with open(CONFIG_FILE) as f:
        data = f.read()
    config = ProxyConfig.model_validate_json(data)
    global prediction_summary
    prediction_summary = {
        config.current_version: {"correct": 0, "total": 0},
        config.previous_version: {"correct": 0, "total": 0},
    }


@app.post("/predict")
def predict(image: Base64ImageType) -> AnimalResponse:
    """
    Predict the animal in the image and return the prediction.

    Args:
        image: The image in base64 format.

    Returns:
        The prediction of the animal in the image.
    """
    global config
    return process_request(base_64_str=image.base_64_str, config=config)


@app.post("/upload")
async def upload(file: UploadFile = File(...)) -> AnimalResponse:
    """
    Predict the animal in the image and return the prediction.

    Args:
        file: The image file.

    Returns:
        The prediction of the animal in the image.
    """
    global config
    contents = await file.read()
    base_64_str = base64.b64encode(contents).decode("utf-8")
    return process_request(base_64_str=base_64_str, config=config)

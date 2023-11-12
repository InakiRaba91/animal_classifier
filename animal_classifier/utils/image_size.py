from pydantic import BaseModel
from pydantic.types import PositiveInt


class ImageSize(BaseModel):
    width: PositiveInt
    height: PositiveInt

    class Config:
        frozen = True


BASE_IMAGE_SIZE = ImageSize(width=1280, height=720)

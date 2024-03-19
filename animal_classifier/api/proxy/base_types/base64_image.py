from pydantic import BaseModel, field_validator


class Base64ImageType(BaseModel):
    """
    A pydantic model used to validate images stored in base64.
    """

    base_64_str: str

    @field_validator("base_64_str")
    def check_input_string_is_not_empty(cls, b64_str: str):
        """
        Validator to check that the input string is not empty.

        Args:
            b64_str (str): Input string to object.
        """
        assert len(b64_str) != 0, "Cannot have empty input string for B64 Image."
        return b64_str

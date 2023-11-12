# import filecmp
# import os
# from typing import Tuple

# import pytest
# from typer.testing import CliRunner

# from animal_classifier.cfg import cfg
# from animal_classifier.cli import app

# runner = CliRunner()


# class TestCLI:
#     def test_app_training(self, file_keys: Tuple[str, ...], data_dir: str, model_dir: str, s3_client: BaseS3Client):
#         # We need file_keys as an arg so it calls the fixture to ensure test files are in the mocked bucket
#         # as well as the dirs for clean-up after test is finished
#         result = runner.invoke(app, ["training"])
#         assert result.exit_code == 0

#         # check logging worked fine
#         assert "Loss on train dataset:" in result.stdout
#         assert "Loss on val dataset:" in result.stdout
#         assert "Loss on test dataset:" in result.stdout

#         # test weights were stored fine in S3
#         _, _, files = next(os.walk(os.path.join(pytest.test_data_root_location, model_dir)))  # type: ignore
#         model_filename = [file for file in files if not file.endswith("_latest")][0]
#         uploaded_filename_model = f"{model_dir}/{model_filename}"
#         downloaded_filename_model = f"{uploaded_filename_model}_downloaded"
#         model_key = os.path.join(cfg.MODEL_FOLDER, model_filename)
#         s3_client.download_file(key=model_key, filename=downloaded_filename_model)
#         assert filecmp.cmp(uploaded_filename_model, downloaded_filename_model)

#     def test_app_inference(self, file_keys: Tuple[str, ...], model_filename: str, model_filekey_s3: str):
#         # We need file_keys as an arg so it calls the fixture to ensure test files are in the mocked bucket
#         # as well as the dirs for clean-up after test is finished
#         result = runner.invoke(app, ["inference", f"{file_keys[0]}.jpg", model_filename])
#         assert result.exit_code == 0
#         assert "Image displays a cat" in result.stdout

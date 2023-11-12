# import filecmp
# from typing import Dict

# import torch

# from animal_classifier.cfg import cfg
# from animal_classifier.model import CatsDogsNet, StateInfo


# def compare_state_dicts(state_dict1: Dict, state_dict2: Dict):
#     assert state_dict1.keys() == state_dict2.keys()
#     equal_weights = []
#     for key in state_dict1.keys():
#         equal_weights.append(torch.equal(state_dict1[key], state_dict2[key]))
#     return all(equal_weights)


# class TestCatsDogsNet:
#     def test_model_forward_pass(self, model: CatsDogsNet, tol: float = 1e-6):
#         torch.manual_seed(0)
#         item = torch.rand((1, 1, 1, 256))
#         output = model(item)
#         expected_output = 0.997576
#         assert abs(float(output) - expected_output) < tol

#     def test_store_locally(self, model: CatsDogsNet, model_filename: str, state_info: StateInfo, model_dir: str):
#         model_filepath = model.store_locally(
#             model_filename=model_filename,
#             state_info=state_info,
#             model_dir=model_dir,
#         )
#         checkpoint = torch.load(model_filepath)
#         assert compare_state_dicts(state_dict1=checkpoint["model_state_dict"], state_dict2=model.state_dict())

#     def test_load_locally(
#         self,
#         model: CatsDogsNet,
#         model_filename: str,
#         state_info: StateInfo,
#         model_dir: str,
#         model_filepath_stored: str,
#     ):
#         state_info_loaded = model.load_locally(model_filename=model_filename, model_dir=model_dir)
#         assert state_info_loaded == state_info
#         checkpoint = torch.load(model_filepath_stored)
#         assert compare_state_dicts(state_dict1=checkpoint["model_state_dict"], state_dict2=model.state_dict())

#     def test_store_s3(
#         self,
#         model: CatsDogsNet,
#         state_info: StateInfo,
#         model_dir: str,
#         model_filename: str,
#         s3_client: BaseS3Client,
#     ):
#         model_prefix = cfg.MODEL_FOLDER
#         model_key = model.store_s3(
#             model_filename=model_filename,
#             state_info=state_info,
#             bucket=cfg.S3_BUCKET,
#             model_dir=model_dir,
#             model_prefix=model_prefix,
#         )
#         uploaded_filename_model = f"{model_dir}/{model_filename}"
#         downloaded_filename_model = f"{uploaded_filename_model}_downloaded"
#         s3_client.download_file(key=model_key, filename=downloaded_filename_model)
#         assert filecmp.cmp(uploaded_filename_model, downloaded_filename_model)

#     def test_load_s3(
#         self,
#         model: CatsDogsNet,
#         model_dir: str,
#         state_info: StateInfo,
#         model_filename: str,
#         s3_client: BaseS3Client,
#         model_filekey_s3: str,  # we need it to ensure weights are in mocked bucket
#         model_filepath_stored: str,
#     ):
#         state_info_loaded = model.load_s3(
#             model_filename=model_filename,
#             bucket=cfg.S3_BUCKET,
#             model_dir=model_dir,
#             model_prefix=cfg.MODEL_FOLDER,
#         )
#         assert state_info_loaded == state_info
#         checkpoint = torch.load(model_filepath_stored)
#         assert compare_state_dicts(state_dict1=checkpoint["model_state_dict"], state_dict2=model.state_dict())

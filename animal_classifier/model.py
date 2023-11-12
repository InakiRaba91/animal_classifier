# import os
# from typing import Dict

# import torch
# import torch.nn as nn
# from pydantic import BaseModel

# from animal_classifier.cfg import cfg


# class StateInfo(BaseModel):
#     """
#     Summary of state info during model training
#     """

#     epoch: int
#     optimizer_state_dict: Dict
#     loss: float


# class CatsDogsNet(nn.Module):
#     """
#     Model used to map an image to a cat/dog label
#     """

#     def __init__(self):
#         super(CatsDogsNet, self).__init__()
#         # fully connected layer
#         self.fc1 = nn.Linear(256, 1)
#         self.initialize_weights()

#     def forward(self, x):
#         # Pass data through fc1
#         x = self.fc1(x)
#         output = torch.sigmoid(x)
#         return output

#     def initialize_weights(self):
#         """
#         Initialize weights
#         """
#         torch.manual_seed(0)
#         if cfg.USE_CUDA:
#             torch.cuda.manual_seed(0)
#         nn.init.kaiming_uniform_(self.fc1.weight.data)
#         nn.init.constant_(self.fc1.bias.data, 0.5)

#     def store_locally(
#         self,
#         model_filename: str,
#         state_info: StateInfo,
#         model_dir: str = cfg.MODEL_DIR,
#     ) -> str:
#         """
#         Store model weights to local file

#         Args:
#             model_filename: filename to store weights in
#             state_info: StateInfo containing additional info to store together with weights (epoch, loss and optimizer state)
#             model_dir: path to folder where model will be stored locally

#         Returns:
#             model filepath where it was stored
#         """
#         # store weights and state info to local file
#         os.system(f"mkdir -p {model_dir}")
#         model_filepath = f"{model_dir}/{model_filename}"
#         torch.save(
#             {
#                 "epoch": state_info.epoch,
#                 "model_state_dict": self.state_dict(),
#                 "optimizer_state_dict": state_info.optimizer_state_dict,
#                 "loss": state_info.loss,
#             },
#             f"{model_filepath}",
#         )
#         return model_filepath

#     def load_locally(
#         self,
#         model_filename: str,
#         model_dir: str = cfg.MODEL_DIR,
#     ) -> StateInfo:
#         """
#         Load model weights from local file

#         Args:
#             model_filename: filename to load weights from
#             model_dir: path to folder where model will be loaded from locally

#         Returns:
#             state_info: StateInfo containing additional info stored together with weights (epoch, loss and optimizer state)
#         """
#         model_filepath = f"{model_dir}/{model_filename}"
#         checkpoint = torch.load(model_filepath)
#         self.load_state_dict(checkpoint["model_state_dict"])
#         return StateInfo(
#             epoch=checkpoint["epoch"],
#             optimizer_state_dict=checkpoint["optimizer_state_dict"],
#             loss=checkpoint["loss"],
#         )

#     def store_s3(
#         self,
#         model_filename: str,
#         state_info: StateInfo,
#         bucket: str = cfg.S3_BUCKET,
#         model_dir: str = cfg.MODEL_DIR,
#         model_prefix: str = cfg.MODEL_FOLDER,
#     ) -> str:
#         """
#         Store model weights in S3

#         Args:
#             model_filename: filename to store weights in
#             state_info: StateInfo containing additional info to store together with weights (epoch, loss and optimizer state)
#             bucket: unique name of the bucket in S3 containing the frames and annotations
#             model_dir: path to folder where model will be stored locally to cache them
#             model_prefix: path to the directory contained weights for trained models in S3

#         Returns:
#             model_key where the file was stored in S3 bucket
#         """
#         # store weights and state info to local file to cache it
#         model_filepath = self.store_locally(
#             model_filename=model_filename,
#             state_info=state_info,
#             model_dir=model_dir,
#         )

#         # upload local file to s3
#         s3_client = BaseS3Client(bucket=bucket)
#         model_key = os.path.join(model_prefix, model_filename)
#         s3_client.upload_file(key=model_key, filename=model_filepath)
#         return model_key

#     def load_s3(
#         self,
#         model_filename: str,
#         bucket: str = cfg.S3_BUCKET,
#         model_dir: str = cfg.MODEL_DIR,
#         model_prefix: str = cfg.MODEL_FOLDER,
#     ) -> StateInfo:
#         """
#         Loads model weights from S3

#         Args:
#             model_filename: filename to store weights in
#             bucket: unique name of the bucket in S3 containing the frames and annotations
#             model_dir: path to folder where model will be stored locally to cache them
#             model_prefix: path to the directory contained weights for trained models in S3
#         """
#         s3_client = BaseS3Client(bucket=bucket)
#         model_key = os.path.join(model_prefix, model_filename)
#         model_filepath = f"{model_dir}/{model_filename}"
#         s3_client.download_file(key=model_key, filename=model_filepath)
#         return self.load_locally(model_filename=model_filename, model_dir=model_dir)

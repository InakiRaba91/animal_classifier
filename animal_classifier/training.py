# import time
# from typing import Tuple

# import numpy as np
# import torch
# from loguru import logger
# from torch.nn import Module
# from torch.optim import Optimizer
# from torch.utils.data import DataLoader

# from animal_classifier.cfg import cfg
# from animal_classifier.dataset import CatsDogsDataset
# from animal_classifier.model import CatsDogsNet, StateInfo

# np.set_printoptions(linewidth=200, suppress=True)


# def model_evaluation(model: CatsDogsNet, loss_function: Module, loader: DataLoader) -> float:
#     """
#     Module to train a model on a given dataset optimizing the given loss function

#     Args:
#         loader: DataLoader on which evaluation will be done
#         model: CatsDogsNet model used to map the image to the cat/dog label
#         loss_function: loss function

#     Returns:
#         loss on eval dataset
#     """
#     with torch.no_grad():
#         running_loss = 0.0
#         for _, (data, targets) in enumerate(loader):
#             data, targets = data.to(cfg.DEVICE), targets.to(cfg.DEVICE)
#             outputs = model(data)
#             loss = loss_function(outputs.squeeze(), targets.squeeze())
#             summed_batch_loss = loss.item() * data.size(0)
#             running_loss += summed_batch_loss
#     eval_loss = running_loss / len(loader.sampler)  # type: ignore
#     return eval_loss


# def model_training(
#     train_dataset: CatsDogsDataset,
#     val_dataset: CatsDogsDataset,
#     model: CatsDogsNet,
#     loss_function: Module,
#     optimizer: Optimizer,
#     batch_size: int = 1,
#     num_epochs: int = 100,
#     model_dir: str = cfg.MODEL_DIR,
#     bucket: str = cfg.S3_BUCKET,
#     model_prefix: str = cfg.MODEL_FOLDER,
# ) -> Tuple[str, float, float, StateInfo]:
#     """
#     Module to train a model on a given dataset optimizing the given loss function

#     Args:
#         train_dataset: CatsDogsDataset used for training
#         val_dataset: CatsDogsDataset used for validation
#         model: CatsDogsNet model used to map the image to the cat/dog label
#         loss_function: loss function
#         optimizer: optimizer used to determine optimal values for model weights
#         batch_size: batch size in datasets
#         num_epochs: number of epochs to train for
#         model_dir: path to folder where model will be stored locally to cache them
#         bucket: unique name of the bucket in S3 containing the frames and annotations
#         model_prefix: path to the directory contained weights for trained models in S3

#     Returns:
#         model_filename: name of stored model to use for inference
#         train_loss: loss on training dataset after training is done
#         best_loss: loss on val dataset after training is done
#         state_info: StateInfo with summary of where trained finished on
#     """
#     # log used device
#     logger.info(f"Training using device {cfg.DEVICE}")

#     # get loaders
#     torch.manual_seed(0)
#     if cfg.USE_CUDA:
#         torch.cuda.manual_seed(0)
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

#     # loop through epochs to optimize the loss function
#     best_loss = np.inf
#     model_filename = f"{model.__class__.__name__}_{time.strftime('%Y%m%d-%H%M%S')}"
#     for epoch in range(num_epochs):
#         running_loss = 0.0

#         # loop through items in train dataset
#         for _, (data, targets) in enumerate(train_loader):
#             data, targets = data.to(cfg.DEVICE), targets.to(cfg.DEVICE)
#             optimizer.zero_grad()
#             outputs = model(data)
#             loss = loss_function(outputs.squeeze(), targets.squeeze())
#             loss.backward()
#             optimizer.step()

#             summed_batch_loss = loss.item() * data.size(0)
#             running_loss += summed_batch_loss
#         train_loss = running_loss / len(train_loader.sampler)  # type: ignore

#         state_info = StateInfo(epoch=epoch, optimizer_state_dict=optimizer.state_dict(), loss=train_loss)
#         model.store_s3(
#             model_filename=f"{model_filename}_latest",
#             state_info=state_info,
#             model_dir=model_dir,
#             bucket=bucket,
#             model_prefix=model_prefix,
#         )

#         val_loss = model_evaluation(model=model, loss_function=loss_function, loader=val_loader)
#         logger.info(f"[{epoch}] loss:{train_loss}\t val-loss: {val_loss}")
#         if val_loss < best_loss:
#             best_loss = val_loss
#             logger.info("best!")
#             model.store_s3(
#                 model_filename=model_filename,
#                 state_info=state_info,
#                 model_dir=model_dir,
#                 bucket=bucket,
#                 model_prefix=model_prefix,
#             )

#     return model_filename, train_loss, best_loss, state_info

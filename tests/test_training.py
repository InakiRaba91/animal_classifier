# import os
# from typing import Tuple

# import pytest
# import torch
# from torch.utils.data import DataLoader

# from animal_classifier.dataset import CatsDogsDataset
# from animal_classifier.loss import CatsDogsLoss
# from animal_classifier.model import CatsDogsNet
# from animal_classifier.training import model_evaluation, model_training


# class TestTraining:
#     @pytest.mark.parametrize("batch_size", [1, 2])
#     @pytest.mark.parametrize("shuffle", [True, False])
#     def test_model_evaluation(
#         self,
#         file_keys: Tuple[str, ...],
#         data_dir: str,
#         batch_size: int,
#         shuffle: bool,
#         tol: float = 1e-6,
#     ):
#         model = CatsDogsNet()
#         loss_function = CatsDogsLoss()
#         dataset = CatsDogsDataset(file_keys=file_keys, data_dir=data_dir)
#         dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
#         loss = model_evaluation(model=model, loss_function=loss_function, loader=dataloader)
#         expected_loss = 0.890426
#         assert abs(loss - expected_loss) < tol

#     @pytest.mark.parametrize(
#         "batch_size, expected_train_loss, expected_best_loss", [(1, 0.732888, 0.655502), (2, 0.749940, 0.656081)]
#     )
#     def test_model_training(
#         self,
#         file_keys: Tuple[str, ...],
#         data_dir: str,
#         model_dir: str,
#         batch_size: int,
#         expected_train_loss: float,
#         expected_best_loss: float,
#         tol: float = 1e-6,
#     ):
#         # We need file_keys as an arg so it calls the fixture to ensure test files are in the mocked bucket
#         train_dataset, val_dataset, _ = CatsDogsDataset.get_splits(data_dir=data_dir)
#         model = CatsDogsNet()
#         loss_function = CatsDogsLoss()
#         optimizer = torch.optim.Adam(model.parameters())
#         _, train_loss, best_loss, _ = model_training(
#             train_dataset=train_dataset,
#             val_dataset=val_dataset,
#             model=model,
#             loss_function=loss_function,
#             optimizer=optimizer,
#             num_epochs=2,
#             batch_size=batch_size,
#             model_dir=model_dir,
#         )
#         assert abs(train_loss - expected_train_loss) < tol
#         assert abs(best_loss - expected_best_loss) < tol
#         _, _, files = next(os.walk(os.path.join(pytest.test_data_root_location, model_dir)))  # type: ignore
#         num_models_saved = len(files)
#         assert num_models_saved == 2  # latest and best

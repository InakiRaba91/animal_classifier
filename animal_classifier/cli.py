# import os

# import torch
# import typer
# from torch.utils.data import DataLoader

# from animal_classifier.cfg import cfg
# from animal_classifier.dataset import CatsDogsDataset
# from animal_classifier.loss import CatsDogsLoss
# from animal_classifier.model import CatsDogsNet
# from animal_classifier.training import model_evaluation, model_training
# from animal_classifier.utils.enums import AnimalLabel

# app = typer.Typer()


# @app.command()
# def training(
#     data_dir: str = cfg.DATA_DIR,
#     model_dir: str = cfg.MODEL_DIR,
#     batch_size: int = 1,
#     train_frac: float = 0.8,
#     val_frac: float = 0.1,
#     test_frac: float = 0.1,
#     lr: float = 0.0003,
#     weight_decay: float = 0.0,
#     num_epochs: int = 3,
# ):
#     """
#     Pipeline to train cats/dogs classifier

#     Args:
#         data_dir: path to folder where files will be stored locally to cache them
#         model_dir: path to folder where model will be stored locally to cache them
#         batch_size: batch size in datasets
#         train_frac: float indicating the fraction of the split to use for training
#         val_frac: float indicating the fraction of the split to use for validation
#         test_frac: float indicating the fraction of the split to use for testing
#         lr: learning rate
#         weight_decay: weight decay (L2 penalty)
#         num_epochs: number of epochs to train for
#     """
#     assert (train_frac + val_frac + test_frac) == 1, "Train/val/test fractions must add up to 1"
#     # generate datasets and split them
#     train_dataset, val_dataset, test_dataset = CatsDogsDataset.get_splits(
#         data_dir=data_dir, train_frac=train_frac, val_frac=val_frac, test_frac=test_frac
#     )

#     # train model
#     model = CatsDogsNet()
#     loss_function = CatsDogsLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
#     model_filename, train_loss, val_loss, state_info = model_training(
#         train_dataset=train_dataset,
#         val_dataset=val_dataset,
#         model=model,
#         loss_function=loss_function,
#         optimizer=optimizer,
#         num_epochs=num_epochs,
#         batch_size=batch_size,
#         model_dir=model_dir,
#     )
#     # this could be sent to prometheus or tensorboard for logging purposes
#     typer.echo(f"Model {model_filename} trained for {num_epochs} epochs")
#     typer.echo("----------------------------------------------------------")
#     typer.echo(f"Loss on train dataset: {train_loss}")
#     typer.echo(f"Loss on val dataset: {val_loss}")

#     # test best
#     model_filepath = f"{model_dir}/{model_filename}"
#     checkpoint = torch.load(model_filepath)
#     model.load_state_dict(checkpoint["model_state_dict"])
#     model.eval()
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
#     test_loss = model_evaluation(model=model, loss_function=loss_function, loader=test_loader)
#     typer.echo(f"Loss on test dataset: {test_loss}")
#     typer.echo("----------------------------------------------------------")


# @app.command()
# def inference(
#     frame_filename: str,
#     model_filename: str,
#     data_dir: str = cfg.DATA_DIR,
#     model_dir: str = cfg.MODEL_DIR,
#     bucket: str = cfg.S3_BUCKET,
#     frames_prefix: str = cfg.FRAMES_FOLDER,
#     model_prefix: str = cfg.MODEL_FOLDER,
#     threshold: float = 0.5,
# ) -> AnimalLabel:
#     """
#     Pipeline to run inference for cats/dogs classifier

#     Note:
#         Running pipelines for inference is not ideal. We would like to expose the model with KFServe
#         directly on Kubeflow with associated transforms and explain components.

#     Args:
#         frame_filename: name of frame to process (stored in s3 bucket)
#         model_filename: name of stored model to use for inference
#         data_dir: path to folder where files will be stored locally to cache them
#         model_dir: path to folder where model will be stored locally to cache them
#         bucket: unique name of the bucket in S3 containing the frames and annotations
#         frames_prefix: path to the directory containing the frames in S3
#         model_prefix: path to the directory containing the models in S3
#         threshold: threshold used for assigning cat/dog label based on confidence score
#     """
#     # load model
#     model = CatsDogsNet()
#     model.load_s3(model_filename=model_filename, bucket=cfg.S3_BUCKET, model_dir=model_dir, model_prefix=model_prefix)
#     model.eval()
#     model.to(cfg.DEVICE)

#     # get image
#     s3_client = BaseS3Client(bucket=bucket)
#     image_key = os.path.join(frames_prefix, frame_filename)
#     frame = CatsDogsDataset._get_frame(s3_client=s3_client, data_dir=data_dir, image_key=image_key)
#     X = CatsDogsDataset.transform_input_frame(frame, image_size=CatsDogsDataset.DEFAULT_IMAGE_SIZE)
#     with torch.no_grad():
#         score = model(X.unsqueeze(0).to(cfg.DEVICE)).item()
#         animal_label = AnimalLabel.from_score(score=score, threshold=threshold)
#         typer.echo(f"Image displays a {animal_label.name.lower()}")
#         return animal_label


# if __name__ == "__main__":
#     app()

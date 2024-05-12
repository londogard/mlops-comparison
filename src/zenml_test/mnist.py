from typing import Annotated, Tuple
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import logging
import zenml
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker


# Define LightningModule
class SimpleNN(pl.LightningModule):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = nn.functional.cross_entropy(output, target)
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        return self(batch[0])

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)


# Load MNIST dataset
@zenml.step
def load_mnist() -> Tuple[
    Annotated[torch.utils.data.DataLoader, "train_loader"],
    Annotated[torch.utils.data.DataLoader, "test_loader"],
]:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    train_dataset = datasets.MNIST(
        root="./data", train=True, transform=transform, download=True
    )
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=128, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=128, shuffle=False
    )
    return train_loader, test_loader


@zenml.step
def train_model(
    train_loader: torch.utils.data.DataLoader, test_loader: torch.utils.data.DataLoader
):
    import mlflow

    mlflow.autolog()
    # Initialize the LightningModule
    model = SimpleNN()
    params_dictionary = {"epochs": 3}
    mlflow.log_params(params_dictionary)
    # Initialize a trainer
    trainer = pl.Trainer(max_epochs=params_dictionary["epochs"])

    # Train the model
    trainer.fit(model, train_loader)

    # Evaluate the model
    trainer.test(dataloaders=test_loader)


@zenml.pipeline
def train_pipeline():
    logging.info("Getting data...")
    train_loader, test_loader = load_mnist()
    logging.info("Training model...")
    train_model(train_loader, test_loader)
    logging.info("Done!")


if __name__ == "__main__":
    train_pipeline()

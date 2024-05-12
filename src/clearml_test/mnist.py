import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from clearml import Task

# Initialize ClearML task
task = Task.init(
    project_name="MNIST Digit Recognition",
    task_name="Simple NN model with PyTorch Lightning",
    task_type=Task.TaskTypes.training,
    output_uri=None,
)


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


params_dictionary = {"epochs": 3}
task.connect(params_dictionary)
# Load MNIST dataset
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

# Initialize the LightningModule
model = SimpleNN()

# Initialize a trainer
trainer = pl.Trainer(max_epochs=params_dictionary["epochs"])

# Train the model
trainer.fit(model, train_loader)

# Evaluate the model
trainer.test(dataloaders=test_loader)

# Report metrics to ClearML
metrics = trainer.callback_metrics
task.get_logger().report_scalar(
    "test_loss", "loss", iteration=1, value=metrics["test_loss"]
)
task.get_logger().report_scalar(
    "test_accuracy", "accuracy", iteration=1, value=metrics["test_accuracy"]
)

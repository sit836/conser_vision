import os

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
import torchvision.models as models
import torch.optim as optim
from image_dataset import ImagesDataset

from constants import IN_PATH, SPECIES_LABELS
from utils import try_gpu


def plot_training_loss(training_loss):
    plt.figure(figsize=(10, 5))
    training_loss.plot(alpha=0.2, label="loss")

    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.legend(loc=0)
    plt.show()


train_features = pd.read_csv(os.path.join(IN_PATH, "train_features.csv"), index_col="id")
train_labels = pd.read_csv(os.path.join(IN_PATH, "train_labels.csv"), index_col="id")

frac = 0.5
y = train_labels.sample(frac=frac, random_state=1)
x = train_features.loc[y.index].filepath.to_frame()

# note that we are casting the species labels to an indicator/dummy matrix
x_train, x_eval, y_train, y_eval = train_test_split(
    x, y, stratify=y, test_size=0.25
)
print(f'x_train.shape, x_eval.shape: {x_train.shape, x_eval.shape}')

train_dataset = ImagesDataset(x_train, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=32)

model = models.resnet50(pretrained=True)
# model.fc = nn.Sequential(
#     nn.Linear(2048, 100),  # dense layer takes a 2048-dim input and outputs 100-dim
#     nn.ReLU(inplace=True),  # ReLU activation introduces non-linearity
#     nn.Dropout(0.1),  # common technique to mitigate overfitting
#     nn.Linear(
#         100, 8
#     ),  # final dense layer outputs 8-dim corresponding to our target classes
# )
model.fc = nn.Sequential(
    nn.Linear(2048, 2),  # dense layer takes a 2048-dim input and outputs 100-dim
    nn.ReLU(inplace=True),  # ReLU activation introduces non-linearity
    nn.Dropout(0.1),  # common technique to mitigate overfitting
    nn.Linear(
        2, 8
    ),  # final dense layer outputs 8-dim corresponding to our target classes
)
model = model.to(device=try_gpu())

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

num_epochs = 1
training_loss = []

for epoch in range(1, num_epochs + 1):
    print(f"Starting epoch {epoch}")

    # iterate through the dataloader batches. tqdm keeps track of progress.
    for batch_n, batch in tqdm(
            enumerate(train_dataloader), total=len(train_dataloader)
    ):
        # 1) zero out the parameter gradients so that gradients from previous batches are not used in this step
        optimizer.zero_grad()

        # 2) run the foward step on this batch of images
        outputs = model(batch["image"])

        # 3) compute the loss
        loss = criterion(outputs, batch["label"])
        # let's keep track of the loss by epoch and batch
        training_loss.append(float(loss))

        # 4) compute our gradients
        loss.backward()
        # update our weights
        optimizer.step()

training_loss = pd.Series(training_loss)

plot_training_loss(training_loss)
quit()

torch.save(model, "model/model.pth")
loaded_model = torch.load("model/model.pth")

eval_dataset = ImagesDataset(x_eval, y_eval)
eval_dataloader = DataLoader(eval_dataset, batch_size=32)

preds_collector = []

# put the model in eval mode so we don't update any parameters
model.eval()

# we aren't updating our weights so no need to calculate gradients
with torch.no_grad():
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        # 1) run the forward step
        logits = model.forward(batch["image"])
        # 2) apply softmax so that model outputs are in range [0,1]
        preds = nn.functional.softmax(logits, dim=1)
        # 3) store this batch's predictions in df
        # note that PyTorch Tensors need to first be detached from their computational graph before converting to numpy arrays
        preds_df = pd.DataFrame(
            preds.cpu().detach().numpy(),
            index=batch["image_id"],
            columns=SPECIES_LABELS,
        )
        preds_collector.append(preds_df)
        loss = criterion(preds, batch["label"])

eval_preds_df = pd.concat(preds_collector)

eval_predictions = eval_preds_df.idxmax(axis=1)
eval_true = y_eval.idxmax(axis=1)
correct = (eval_predictions == eval_true).sum()
accuracy = correct / len(eval_predictions)
print(f'accuracy: {accuracy}')

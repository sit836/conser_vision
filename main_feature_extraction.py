import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torchvision.models as models
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from constants import IN_PATH, SPECIES_LABELS
from image_dataset import ImagesDataset
from submit import make_submission
from utils import try_gpu, transform_train, transform_test


def get_net():
    net = models.resnet50(pretrained=True)
    # net = models.resnet152(pretrained=True)
    # net = models.efficientnet_b7(pretrained=True)

    for param in net.parameters():
        param.requires_grad = False

    # Overwrite the last fully connected layer.
    # By default, tensor has requires_grad=True, so we do not need to unfreeze fc layer again.
    net.fc = nn.Linear(2048, len(SPECIES_LABELS))

    for param in net.layer4.parameters():
        param.requires_grad = True

    net = net.to(device=try_gpu())

    return net


def get_accuracy(pred_list, y):
    pred_df = pd.concat(pred_list)
    predictions = pred_df.idxmax(axis=1)
    eval_true = y.idxmax(axis=1)
    correct = (predictions == eval_true).sum()
    accuracy = correct / len(predictions)
    return accuracy


def evaluate_loss(net, val_dataloader):
    val_loss_list = []
    val_pred_list = []

    # put the model in eval mode so we don't update any parameters
    net.eval()

    # we aren't updating our weights so no need to calculate gradients
    with torch.no_grad():
        for batch in tqdm(val_dataloader, total=len(val_dataloader)):
            # 1) run the forward step
            logits = net.forward(batch["image"])
            # 2) apply softmax so that model outputs are in range [0,1]
            preds = nn.functional.softmax(logits, dim=1)

            pred_val = pd.DataFrame(
                preds.cpu().detach().numpy(),
                index=batch["image_id"],
                columns=SPECIES_LABELS,
            )
            val_pred_list.append(pred_val)
            val_loss_list.append(criterion(preds, batch["label"]).item())
    return val_loss_list, val_pred_list


def make_plot(train_loss_epoch_arr, val_loss_epoch_arr, train_acc_epoch_arr, val_acc_epoch_arr):
    fontsize = 15
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(train_loss_epoch_arr, label="train")
    ax1.plot(val_loss_epoch_arr, label="val")
    ax1.set_xlabel("Epoch", fontsize=fontsize)
    ax1.set_ylabel("Loss", fontsize=fontsize)
    ax1.legend(loc=0, fontsize=fontsize)

    ax2.plot(train_acc_epoch_arr, label="train")
    ax2.plot(val_acc_epoch_arr, label="val")
    ax2.set_xlabel("Epoch", fontsize=fontsize)
    ax2.set_ylabel("Acc", fontsize=fontsize)
    ax2.legend(loc=0, fontsize=fontsize)
    plt.show()
    # plt.savefig(f'plot/num_node_{num_node}.png')
    # plt.close()


batch_size = 64
num_epochs = 10

lr = 1e-3
lr_period, lr_decay = 2, 0.9
wd = 0.01

train_features = pd.read_csv(os.path.join(IN_PATH, "train_features.csv"), index_col="id")
train_labels = pd.read_csv(os.path.join(IN_PATH, "train_labels.csv"), index_col="id")

# TODO: for test purpose
# frac = 0.5
# y = train_labels.sample(frac=frac, random_state=1)
# x = train_features.loc[y.index].filepath.to_frame()

y = train_labels
x = train_features.filepath.to_frame()

# note that we are casting the species labels to an indicator/dummy matrix
x_train, x_eval, y_train, y_eval = train_test_split(
    x, y, stratify=y, test_size=0.27
)
print(f'x_train.shape, x_eval.shape: {x_train.shape, x_eval.shape}')

train_dataset = ImagesDataset(transform_train, x_train, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
val_dataset = ImagesDataset(transform_test, x_eval, y_eval)
val_dataloader = DataLoader(val_dataset, batch_size=64)

net = get_net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
# optimizer = optim.SGD((param for param in net.parameters() if param.requires_grad), lr=lr, momentum=0.9)
# optimizer = optim.Adam(model.parameters(), lr=1e-4)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_period, lr_decay)

train_loss_epoch_arr = np.zeros(num_epochs)
val_loss_epoch_arr = np.zeros(num_epochs)
train_acc_epoch_arr = np.zeros(num_epochs)
val_acc_epoch_arr = np.zeros(num_epochs)

for epoch in range(1, num_epochs + 1):
    train_loss_list = []
    train_pred_list = []

    # iterate through the dataloader batches. tqdm keeps track of progress.
    for batch_n, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        # 1) zero out the parameter gradients so that gradients from previous batches are not used in this step
        optimizer.zero_grad()

        # 2) run the foward step on this batch of images
        outputs = net(batch["image"])

        # 3) compute the loss
        loss = criterion(outputs, batch["label"])
        # let's keep track of the loss by epoch and batch
        pred_train = pd.DataFrame(
            outputs.cpu().detach().numpy(),
            index=batch["image_id"],
            columns=SPECIES_LABELS,
        )
        train_pred_list.append(pred_train)
        train_loss_list.append(loss.item())

        # 4) compute our gradients
        loss.backward()
        # update our weights
        optimizer.step()

    scheduler.step()

    val_loss_list, val_pred_list = evaluate_loss(net, val_dataloader)

    train_loss = np.mean(train_loss_list)
    val_loss = np.mean(val_loss_list)
    train_acc = get_accuracy(train_pred_list, y_train)
    val_acc = get_accuracy(val_pred_list, y_eval)

    train_loss_epoch_arr[epoch - 1] = train_loss
    val_loss_epoch_arr[epoch - 1] = val_loss
    train_acc_epoch_arr[epoch - 1] = train_acc
    val_acc_epoch_arr[epoch - 1] = val_acc

    print(
        f'\nEpoch {epoch}\ntrain loss: {round(train_loss, 2)}, '
        f'train acc: {round(train_acc, 2)}\nval loss: {round(val_loss, 2)}, '
        f'val acc: {round(val_acc, 2)}')

    # plt.plot(train_loss_list)
    # plt.show()

make_plot(train_loss_epoch_arr, val_loss_epoch_arr, train_acc_epoch_arr, val_acc_epoch_arr)

# TODO: train model on the full dataset

# make_submission(net)

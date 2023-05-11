import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.models as models
from torch import nn
from torchvision import transforms
from tqdm import tqdm

from constants import RGB_MEAN, RGB_STD, IMAGE_SIZE
from constants import SPECIES_LABELS
from utils import get_accuracy, try_gpu

criterion = nn.CrossEntropyLoss()


def get_net(net_name, feature_extraction=False):
    if net_name == 'resnet18':
        net = models.resnet18(pretrained=True)
    elif net_name == 'resnet34':
        net = models.resnet34(pretrained=True)
    elif net_name == 'resnet50':
        net = models.resnet50(pretrained=True)
    elif net_name == 'resnet152':
        net = models.resnet152(pretrained=True)
    elif net_name == 'efficientnet_b3':
        net = models.efficientnet_b3(pretrained=True)
    else:
        raise Exception(f'net_name is not a valid {net_name}.')

    if feature_extraction:
        for param in net.parameters():
            param.requires_grad = False

    # Overwrite the last fully connected layer.
    # By default, tensor has requires_grad=True, so we do not need to unfreeze fc layer again.
    net.fc = nn.Linear(2048, len(SPECIES_LABELS))
    # net.classifier[1] = nn.Linear(1536, len(SPECIES_LABELS))

    if feature_extraction:
        for param in net.layer4.parameters():
            param.requires_grad = True

    # net.classifier[1] = nn.Linear(1536, len(SPECIES_LABELS))
    # for param in net.features.parameters():
    #     param.requires_grad = True

    net = net.to(device=try_gpu()).half()

    return net


def train_batch_loop(net, train_dataloader, optimizer):
    train_loss_list = []
    train_pred_list = []

    # iterate through the dataloader batches. tqdm keeps track of progress.
    for batch_n, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        # 1) zero out the parameter gradients so that gradients from previous batches are not used in this step
        optimizer.zero_grad()

        # 2) run the foward step on this batch of images
        outputs = net(batch["image"].half())

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
    return train_loss_list, train_pred_list


def valid_batch_loop(net, val_dataloader):
    val_loss_list = []
    val_pred_list = []

    # put the model in eval mode so we don't update any parameters
    net.eval()

    # we aren't updating our weights so no need to calculate gradients
    with torch.no_grad():
        for batch in tqdm(val_dataloader, total=len(val_dataloader)):
            # 1) run the forward step
            logits = net.forward(batch["image"].half())
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


def fit(net, optimizer, y_train, y_eval, train_dataloader, val_dataloader, num_epochs, lr_period, lr_decay):
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_period, lr_decay)
    train_loss_epoch_arr = np.zeros(num_epochs)
    val_loss_epoch_arr = np.zeros(num_epochs)
    train_acc_epoch_arr = np.zeros(num_epochs)
    val_acc_epoch_arr = np.zeros(num_epochs)

    for epoch in range(1, num_epochs + 1):
        train_loss_list, train_pred_list = train_batch_loop(net, train_dataloader, optimizer)
        train_loss = np.mean(train_loss_list)
        train_acc = get_accuracy(train_pred_list, y_train)
        train_loss_epoch_arr[epoch - 1] = train_loss
        train_acc_epoch_arr[epoch - 1] = train_acc
        scheduler.step()

        print(
            f'\nEpoch {epoch}\ntrain loss: {round(train_loss, 2)}, '
            f'train acc: {round(train_acc, 2)}\n')

        if val_dataloader is not None:
            val_loss_list, val_pred_list = valid_batch_loop(net, val_dataloader)
            val_loss = np.mean(val_loss_list)
            val_acc = get_accuracy(val_pred_list, y_eval)
            val_loss_epoch_arr[epoch - 1] = val_loss
            val_acc_epoch_arr[epoch - 1] = val_acc

            print(
                f'val loss: {round(val_loss, 2)}, '
                f'val acc: {round(val_acc, 2)}')

        # plt.plot(train_loss_list)
        # plt.show()
    return train_loss_epoch_arr, val_loss_epoch_arr, train_acc_epoch_arr, val_acc_epoch_arr


transform_train = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.08, 1.0),
                                 ratio=(3.0 / 4.0, 4.0 / 3.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.ToTensor(),
    transforms.Normalize(RGB_MEAN, RGB_STD)])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(RGB_MEAN, RGB_STD)])

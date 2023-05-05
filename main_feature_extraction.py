import os

import pandas as pd
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from constants import IN_PATH
from image_dataset import ImagesDataset
from model import get_net, fit, transform_train, transform_test
from utils import make_plot


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
full_dataset = ImagesDataset(transform_train, x, y)
full_dataloader = DataLoader(full_dataset, batch_size=64)

net = get_net()
optimizer = optim.SGD([p for p in net.parameters() if p.requires_grad], lr=lr, momentum=0.9, weight_decay=wd)
# optimizer = optim.Adam(model.parameters(), lr=1e-4)

train_loss_epoch_arr, val_loss_epoch_arr, train_acc_epoch_arr, val_acc_epoch_arr = fit(net, optimizer, y_train, y_eval, train_dataloader,
                                                                                       val_dataloader, num_epochs, lr_period, lr_decay)
make_plot(train_loss_epoch_arr, val_loss_epoch_arr, train_acc_epoch_arr, val_acc_epoch_arr)

# fit(net, optimizer, y, None, full_dataloader, None, num_epochs, lr_period, lr_decay)
# make_submission(net)

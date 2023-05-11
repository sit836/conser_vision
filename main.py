import os

import pandas as pd
import torch.optim as optim
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import DataLoader

from constants import IN_PATH
from image_dataset import ImagesDataset
from model import get_net, fit, transform_train, transform_test
from utils import make_plot
from submit import make_submission

batch_size = 32
num_epochs = 10

lr = 5e-4
lr_period, lr_decay = 100, 0.9
wd = 0.0
net_name = "resnet152"
feature_extraction = False

train_features = pd.read_csv(os.path.join(IN_PATH, "train_features.csv"), index_col="id")
train_labels = pd.read_csv(os.path.join(IN_PATH, "train_labels.csv"), index_col="id")

y = train_labels
x = train_features.filepath.to_frame()

gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=12)
train_ind, eval_ind = next(gss.split(x, y, groups=train_features.site))
x_train, x_eval, y_train, y_eval = x.iloc[train_ind], x.iloc[eval_ind], y.iloc[train_ind], y.iloc[eval_ind]
print(f'x_train.shape, x_eval.shape: {x_train.shape, x_eval.shape}')
# print(y_train.sum())
# print(y_eval.sum())

train_dataset = ImagesDataset(transform_train, x_train, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
val_dataset = ImagesDataset(transform_test, x_eval, y_eval)
val_dataloader = DataLoader(val_dataset, batch_size=64)
full_dataset = ImagesDataset(transform_train, x, y)
full_dataloader = DataLoader(full_dataset, batch_size=64)

net = get_net(net_name, feature_extraction=feature_extraction)
optimizer = optim.SGD([p for p in net.parameters() if p.requires_grad], lr=lr, momentum=0.9, weight_decay=wd)
# optimizer = optim.Adam([p for p in net.parameters() if p.requires_grad], lr=lr, weight_decay=wd)

train_loss_epoch_arr, val_loss_epoch_arr, train_acc_epoch_arr, val_acc_epoch_arr = fit(net, optimizer, y_train, y_eval,
                                                                                       train_dataloader,
                                                                                       val_dataloader, num_epochs,
                                                                                       lr_period, lr_decay)
make_plot(train_loss_epoch_arr, val_loss_epoch_arr, train_acc_epoch_arr, val_acc_epoch_arr, lr, num_epochs, lr_period,
          lr_decay, wd, net_name, feature_extraction)

# fit(net, optimizer, y, None, full_dataloader, None, num_epochs, lr_period, lr_decay)
# make_submission(net)

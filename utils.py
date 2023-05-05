import matplotlib.pyplot as plt
import pandas as pd
import torch


def cpu():
    return torch.device('cpu')


def gpu(i=0):
    return torch.device(f'cuda:{i}')


def num_gpus():
    return torch.cuda.device_count()


def try_gpu(i=0):
    if num_gpus() >= i + 1:
        return gpu(i)
    return cpu()


def get_accuracy(pred_list, y):
    pred_df = pd.concat(pred_list)
    predictions = pred_df.idxmax(axis=1)
    eval_true = y.idxmax(axis=1)
    correct = (predictions == eval_true).sum()
    accuracy = correct / len(predictions)
    return accuracy


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

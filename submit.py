import os

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from constants import IN_PATH, OUT_PATH, SPECIES_LABELS
from image_dataset import ImagesDataset
from model import transform_test


def make_submission(model):
    print("Make a submission...")
    test_features = pd.read_csv(os.path.join(IN_PATH, "test_features.csv"), index_col="id")
    test_dataset = ImagesDataset(transform_test, test_features.filepath.to_frame())
    test_dataloader = DataLoader(test_dataset, batch_size=64)

    preds_collector = []

    # put the model in eval mode so we don't update any parameters
    model.eval()

    # we aren't updating our weights so no need to calculate gradients
    with torch.no_grad():
        for batch in tqdm(test_dataloader, total=len(test_dataloader)):
            # run the forward step
            logits = model.forward(batch["image"].half())
            # apply softmax so that model outputs are in range [0,1]
            preds = nn.functional.softmax(logits, dim=1)
            # store this batch's predictions in df
            # note that PyTorch Tensors need to first be detached from their computational graph before converting to numpy arrays
            preds_df = pd.DataFrame(
                preds.cpu().detach().numpy(),
                index=batch["image_id"],
                columns=SPECIES_LABELS,
            )
            preds_collector.append(preds_df)

    submission_df = pd.concat(preds_collector)
    submission_df.to_csv(os.path.join(OUT_PATH, "submission_df.csv"))

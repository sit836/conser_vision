import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from constants import IN_PATH
from utils import try_gpu


class ImagesDataset(Dataset):
    """Reads in an image, transforms pixel values, and serves
    a dictionary containing the image id, image tensors, and label.
    """

    def __init__(self, x_df, y_df=None):
        self.data = x_df
        self.label = y_df
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

    def __getitem__(self, index):
        image = Image.open(os.path.join(IN_PATH, self.data.iloc[index]["filepath"])).convert("RGB")
        image = self.transform(image).to(device=try_gpu())
        image_id = self.data.index[index]

        # if we don't have labels (e.g. for test set) just return the image and image id
        if self.label is None:
            sample = {"image_id": image_id, "image": image}
        else:
            label = torch.tensor(self.label.iloc[index].values,
                                 dtype=torch.float).to(device=try_gpu())
            sample = {"image_id": image_id, "image": image, "label": label}
        return sample

    def __len__(self):
        return len(self.data)

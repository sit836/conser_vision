import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from constants import IN_PATH, RGB_MEAN, RGB_STD
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

                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),

                transforms.ToTensor(),
                transforms.Normalize(
                    mean=RGB_MEAN, std=RGB_STD
                ),
            ]
        )

        # self.transform = transforms.Compose([
        #     # Randomly crop the image to obtain an image with an area of 0.08 to 1 of
        #     # the original area and height-to-width ratio between 3/4 and 4/3. Then,
        #     # scale the image to create a new 224 x 224 image
        #     transforms.RandomResizedCrop(224, scale=(0.08, 1.0),
        #                                  ratio=(3.0 / 4.0, 4.0 / 3.0)),
        #     transforms.RandomHorizontalFlip(),
        #     # Randomly change the brightness, contrast, and saturation
        #     transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        #     # Add random noise
        #     transforms.ToTensor(),
        #     # Standardize each channel of the image
        #     transforms.Normalize(RGB_MEAN, RGB_STD)])

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

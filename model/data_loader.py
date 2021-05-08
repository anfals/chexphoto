import os

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
import math

# borrowed from http://pytorch.org/tutorials/advanced/neural_style_tutorial.html
# and http://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# define a training image loader that specifies transforms on images. See documentation for more details.

# Resize to 320x320 and match DenseNet-121 requirements
preprocess = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# train_transformer = transforms.Compose([
#     transforms.Resize(64),  # resize the image to 64x64 (remove if images are already 64x64)
#     transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
#     transforms.ToTensor()])  # transform it into a torch tensor
#
# # loader for evaluation, no horizontal flip
# eval_transformer = transforms.Compose([
#     transforms.Resize(64),  # resize the image to 64x64 (remove if images are already 64x64)
#     transforms.ToTensor()])  # transform it into a torch tensor


class CheXPertDataset(Dataset):
    path = "CheXpert-v1.0-small"
    observations = ["Pleural Effusion", "Edema", "Atelectasis", "Consolidation", "Cardiomegaly"]

    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """
    def __init__(self, data_df, transform):
        """
        Store the filenames of the jpgs to use. Specifies transforms to apply on images.

        Args:
            data_dir: (string) directory containing the dataset
            transform: (torchvision.transforms) transformation to apply on image
        """
        self.filenames = []
        self.labels = []

        for index, row in data_df.iterrows():
            file_path = row["Path"]
            # TODO: verify this path stuff later
            file_path = os.path.relpath(file_path, CheXPertDataset.path)
            file_path = os.path.join("data", file_path)

            if os.path.exists(file_path):
                self.filenames.append(file_path)
                label = torch.zeros((5,))
                for index, observation in enumerate(CheXPertDataset.observations):
                    o_result = row[observation]
                    # We are going with the U-Ones approach
                    if math.isnan(o_result) or o_result == 0:
                        label[index] = 0
                    else:
                        label[index] = 1
                self.labels.append(label)
            else:
                print(f'File was not found at path ${file_path}')

        self.transform = transform

    def __len__(self):
        # return size of dataset
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        Fetch index idx image and labels from dataset. Perform transforms on image.

        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]

        Returns:
            image: (Tensor) transformed image
            label: (int) corresponding label of image
        """
        image = Image.open(self.filenames[idx])  # PIL image
        image = self.transform(image)
        return image, self.labels[idx]


def fetch_dataloader(types, data_dir, params):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}

    # load the pandas dataframe for the csv files describing the datasets
    train_csv = os.path.join(data_dir, "train.csv")
    val_csv = os.path.join(data_dir, "valid.csv")
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(val_csv)

    # Shuffle the training one up, with a random state to keep things reproducible
    train_df = train_df.sample(frac=1, random_state=2)

    # Split train into our training and validation sets
    num_train = int(train_df.shape[0] * 0.9)
    train_df, val_df = train_df.iloc[:num_train, :], train_df.iloc[num_train:, :]

    dfs = [train_df, val_df, test_df]

    for index, split in enumerate(['train', 'val', 'test']):
        if split in types:
            df = dfs[index]
            dl = DataLoader(CheXPertDataset(df, preprocess), batch_size=params.batch_size, shuffle=split == 'train',
                            num_workers=params.num_workers,
                            pin_memory=params.cuda)
            dataloaders[split] = dl

    return dataloaders

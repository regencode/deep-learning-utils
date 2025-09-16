from numpy.core.multiarray import ndarray
import torch, torchvision
import torch.utils.data
import typing, torchtyping
import csv
import tifffile
from typing import Any, Callable
import torch.nn.functional as F
import numpy as np
import os


Callback = Callable[[Any], Any]

class SequentialPipeline():
    def __init__(self,  action_list: tuple[Callback]):
        self.action_list = action_list

    def __call__(self, *data: Any) -> tuple[Any, ...]:
        # Apply transforms sequentially
        # Make sure each action can be executed as a function
        for action in self.action_list:
            data = action(data)
        return data


class InMemoryDataset(torch.utils.data.Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor,
                 transforms: SequentialPipeline | None = None, 
                 X_transforms: SequentialPipeline | None = None, 
                 y_transforms: SequentialPipeline | None = None, 
                 pair_transforms: SequentialPipeline | None = None):

        self.X = X
        self.y = y
        self.transforms = transforms
        self.X_transforms=  X_transforms
        self.y_transforms = y_transforms
        self.pair_transforms = pair_transforms

        if self.X_transforms is None: # default behavior is self.transforms
            self.X_transforms = transforms
        if self.y_transforms is None:
            self.y_transforms = transforms

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        x_data = self.X[i]
        y_data = self.y[i]
        
        x_data = self.X_transforms(x_data) if self.X_transforms is not None else x_data
        y_data = self.y_transforms(y_data) if self.y_transforms is not None else y_data
        if self.pair_transforms is not None:
            x_data, y_data = self.pair_transforms(x_data, y_data)

        # Convert numpy arrays to tensors
        x_data = torch.from_numpy(x_data).float()
        y_data = torch.from_numpy(y_data).long() # Labels are likely integers
        return x_data, y_data

    def __len__(self) -> int :
        return len(self.X)


def load_csv_data(root_path: os.PathLike, 
                  input_folder_path: os.PathLike[str], 
                  label_folder_path: os.PathLike[str]) -> tuple[list[str], list[str]]:
    # Return (X, y) tuple where X and y are list of img and their label in .tiff format
    X, y = [], []
    with open(root_path, newline='') as csv_file:
        reader = csv.reader(csv_file)
        for input, label in reader:
            input = os.path.join(input_folder_path, input)
            label = os.path.join(label_folder_path, label)
            X.append(input)
            y.append(label)
    return X, y

def read_from_tiff(tiff_path: str) -> np.ndarray:
    img = tifffile.imread(tiff_path)
    return img

def identity(data: Any) -> Any: # for empty pipeline
    return data

class OneHotEncoder:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def __call__(self, label: torch.LongTensor) -> torch.Tensor:
        return F.one_hot(label, num_classes=self.num_classes)


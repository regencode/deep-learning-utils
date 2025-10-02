import numpy
import torch, torchvision
import torch.utils.data
import typing, torchtyping
import csv
import tifffile
from typing import Any, Callable, List
import torch.nn.functional as F
import numpy as np
import os


Callback = Callable[[Any], Any]
Path = os.PathLike

class SequentialPipeline():
    '''
         Apply transforms sequentially to one data,
         where each transform may result in n data,
         finally outputting one or more data. (1 to n>=1)
         Make sure each action can be executed as a function
         *args will be ignored for the sake of GroupedOperation
    '''
    def __init__(self,  action_list: tuple[Callback]):
        self.action_list = action_list

    def __call__(self, data: Any, *args) -> tuple[Any, ...]:
        if not isinstance(data, tuple):
            data = (data,)  # wrap into tuple
        for action in self.action_list:
            data = action(data)
        return data

class OneToManyOperation():
    '''
    Apply a list of actions (n actions) to a single data (1 data),
    returning multiple data (n data)
    *args is discarded.
    '''
    def __init__(self, each_action):
        self.each_action = each_action

    def __call__(self, data, *args):
        new_data = []
        for action in self.each_action:
            new_data.append(action(data))
        return new_data

class GroupedOperation():
    '''
    Apply the same action (1 action) to a group of data (n_data),
    returning mutiple data (n_data)
    __call__ is variadic.
    '''
    def __init__(self, action: Callback):
        self.action = action

    def __call__(self, *data: Any):
        data = self.action(data)
        return data

class InMemoryDataset(torch.utils.data.Dataset):
    def __init__(self, X: List[Path], y: List[Path]):
        self.X = X
        self.y = y

    def __getitem__(self, i: int) -> tuple[Path, Path]:
        return self.X[i], self.y[i]

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

class TiffReader:
    def __init__(self, path_prefix: str = ""):
        self.path_prefix = path_prefix
    def __call__(self, tiff_path: str) -> np.ndarray:
        if self.path_prefix is not None:
            tiff_path = os.path.join(self.path_prefix, tiff_path)
        img = tifffile.imread(tiff_path)
        return img

def identity(data: Any) -> Any: # for empty pipeline
    return data

class OneHotEncoder:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def __call__(self, label: torch.LongTensor) -> torch.Tensor:
        return F.one_hot(label, num_classes=self.num_classes)


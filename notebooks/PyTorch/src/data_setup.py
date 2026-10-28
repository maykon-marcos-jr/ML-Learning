"""
Contains functionality for creating PyTorch DataLoaders for
image classification data.
"""

import os
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from pathlib import Path
from typing import Tuple, Dict, List

NUM_WORKERS = os.cpu_count()

if not NUM_WORKERS:
    NUM_WORKERS = 1

def find_classes(Dir: str|Path) -> Tuple[List[str], Dict[str, int]]:
    """
    Finds the class folder names in a target directory.

    Assumes target directory is in standard image classification format.

    Args:
        directory (str): target directory to load classnames from.

    Returns:
        Tuple[List[str], Dict[str, int]]: (list_of_class_names, dict(class_name: idx...))

    Example:
        find_classes("food_images/train")
        >>> (["class_1", "class_2"], {"class_1": 0, ...})
    """
    # 1. Get the class names by scanning the target directory
    classes = sorted(entry.name for entry in os.scandir(Dir) if entry.is_dir())

    # 2. Raise an error if class names not found
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {Dir}.")

    # 3. Create a dictionary of index labels (computers prefer numerical rather than string labels)
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def create_img_dataloaders(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int=NUM_WORKERS
):
  """Creates training and testing DataLoaders.

  Takes in a training directory and testing directory path and turns
  them into PyTorch Datasets and then into PyTorch DataLoaders.

  Args:
    train_dir: Path to training directory.
    test_dir: Path to testing directory.
    transform: torchvision transforms to perform on training and testing data.
    batch_size: Number of samples per batch in each of the DataLoaders.
    num_workers: An integer for number of workers per DataLoader.

  Returns:
    A tuple of (train_dataloader, test_dataloader, class_names).
    Where class_names is a list of the target classes.
    Example usage:
      train_dataloader, test_dataloader, class_names = \
        = create_img_dataloaders(train_dir=path/to/train_dir,
                             test_dir=path/to/test_dir,
                             transform=some_transform,
                             batch_size=32,
                             num_workers=4)
  """
  # Use ImageFolder to create dataset(s)
  train_data = ImageFolder(
    root=train_dir,
    transform=transform, # transform for the input
    target_transform=None, # transformer for the output
  )
  test_data = ImageFolder(
        root=test_dir,
        transform=transform, # transform for the input
        target_transform=None, # transformer for the output
    )

  # Get class names
  class_names = train_data.classes

  # Turn images into data loaders
  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      # Keeps more memory in the GPU
      # Reducing the transfer time
      pin_memory=True,
  )
  test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False, # don't need to shuffle test data
      num_workers=num_workers,
      # Keeps more memory in the GPU
      # Reducing the transfer time
      pin_memory=True,
  )

  return train_dataloader, test_dataloader, class_names


class CustomDataSet(Dataset):
    """
    - Subclass torch.utils.data.Dataset.
    - Initialize our subclass with a target_dir parameter and transform parameter.
    - Create several attributes for paths, transform, classes and class_to_idx.
    - Create a function to load data from file and return them.
    - Overwrite the __len__ method of torch.utils.data.Dataset to return the number of samples in the Dataset, allowing calling len(Dataset).
    - Overwrite the __getitem__ method of torch.utils.data.Dataset to return a single sample from the Dataset.
    """
    def __init__(self,
                target_dir: str|Path,
                data_ext:str="jpg",
                dtype=Image,
                transformer:transforms.Compose|None = None
        ):
        super().__init__()
        # Get all image paths
        self.paths = list(Path(target_dir).glob(f"*/*.{data_ext}"))
        # Setup transforms
        if transformer is None:
            transformer = transforms.Compose([(transforms.ToTensor())])
        self.transform = transformer
        # Create classes and class_to_idx attributes
        self.classes, self.class_to_idx = find_classes(target_dir)

        if dtype == Image:
            self.open_method = Image.open
        else:
            raise TypeError("Data Type not supported")


    def load_data(self, index: int) -> torch.Tensor:
        data_path = self.paths[index]
        data = self.open_method(data_path)
        return self.transform(data) # type: ignore

    # Overwrite the __len__() method
    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths)

    # Overwrite the __getitem__() method (required for subclasses of torch.utils.data.Dataset)
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        "Returns one sample of data, data and label (X, y)."
        data = self.load_data(index)
        class_name  = self.paths[index].parent.name
        class_idx = self.class_to_idx[class_name]

        # return data, label (X, y)
        return data, class_idx

from pathlib import Path
from datasets import DatasetDict, Dataset
from PIL import Image, ImageOps
import numpy as np

class LazyLoadingImageDataset(Dataset):
    def __init__(self, dataset: Dataset, size: tuple = (224, 224), padding: bool = False):
        self.dataset: Dataset = dataset
        self.size: tuple = size
        self.padding_transformation: bool = padding

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Handle batch indices (list of indices)
        if isinstance(idx, list):
            items = [self._get_single_item(i) for i in idx]
            # Convert list of dicts to dict of lists
            batch = {}
            for key in items[0].keys():
                batch[key] = [item[key] for item in items]
            return batch
        
        return self._get_single_item(idx)
    
    def _get_single_item(self, idx):
        row = self.dataset[idx]
        image_path = row["image"]
        image = Image.open(image_path)
        
        if image.mode != "RGB":
            image = image.convert("RGB")

        if not self.padding_transformation:
            image = image.resize(self.size, Image.BILINEAR)
            row["image"] = np.array(image)
            return row
        
        # If padding transformation, do the following

        ## Get aspect ratio of the image and the target image's aspect ratio
        image_aspect_ratio = image.width / image.height
        target_aspect_ratio = self.size[0] / self.size[1]
        if image_aspect_ratio == target_aspect_ratio:
            image = image.resize(self.size, Image.BILINEAR)
            row["image"] = np.array(image)
            return row
        
        longer_dimension = 0 if image.width > image.height else 1
        ## Scale the image up/down so that it's longer dimension is the same as the target size in that dimension, maintaining aspect ratio
        if longer_dimension == 0:
            new_width = self.size[0]
            new_height = int(image.height * (new_width / image.width))
        else:
            new_height = self.size[1]
            new_width = int(image.width * (new_height / image.height))
        image = image.resize((new_width, new_height), Image.BILINEAR)
        ## Pad the shorter dimension so that the final image is the target size, select padding from the image edges, pad evenly left and right/ top and bottom
        padding_left = (self.size[0] - new_width) // 2
        padding_right = self.size[0] - new_width - padding_left
        padding_top = (self.size[1] - new_height) // 2
        padding_bottom = self.size[1] - new_height - padding_top
        image = ImageOps.expand(image, border=(padding_left, padding_top, padding_right, padding_bottom), fill=0)
        row["image"] = np.array(image)
        return row
    
    def __repr__(self):
        return self.dataset.__repr__()

    def __getattr__(self, name):
        # Delegate all other attribute access to the wrapped dataset
        return getattr(self.dataset, name)
    
def convert_to_lazy_loading_dataset(dataset: DatasetDict, size: tuple = (512, 512)) -> DatasetDict:
    dataset["train"] = LazyLoadingImageDataset(dataset["train"], size=size)
    dataset["validation"] = LazyLoadingImageDataset(dataset["validation"], size=size)
    dataset["test"] = LazyLoadingImageDataset(dataset["test"], size=size)
    return dataset

def load_and_split_dataset(labels: list[str] | None, dataset_folder_path: Path | str, split_ratio: tuple[float] = (0.7, 0.2, 0.1),
                           split_seed: int = 42, sample_seed: int = 42) -> DatasetDict:
    
    datasetDict_per_label: DatasetDict = load_dataset_from_source(labels, dataset_folder_path)
    downsample_target: int = get_min_count_in_labels(datasetDict_per_label)
    datasetDict_per_label_downsampled: DatasetDict = downsample_dataset(datasetDict_per_label, sample_size=downsample_target, sample_seed=sample_seed)
    split_datasetDict = split_dataset(datasetDict_per_label_downsampled, split_ratio=split_ratio, split_seed=split_seed)
    return split_datasetDict
    

def downsample_dataset(dataset: DatasetDict, sample_size: int, sample_seed: int) -> DatasetDict:
    for label, dataset_values in dataset.items():
        if len(dataset_values) == sample_size:
            continue
        values_shuffled = dataset_values.shuffle(seed=sample_seed)
        dataset[label] = values_shuffled.select(range(sample_size))

    return dataset

def split_dataset(dataset: DatasetDict, split_ratio: tuple[float], split_seed: int) -> DatasetDict:
    train_list = []
    val_list = []
    test_list = []
    
    for label, dataset_values in dataset.items():
        n = len(dataset_values)
        # Calculate exact sample counts to avoid rounding errors
        train_count = int(n * split_ratio[0])
        val_count = int(n * split_ratio[1])
        test_count = n - train_count - val_count  # Remaining samples go to test
        
        # Shuffle once and use select with exact indices
        shuffled = dataset_values.shuffle(seed=split_seed)
        
        train_split = shuffled.select(range(0, train_count)).add_column("label", [label] * train_count)
        val_split = shuffled.select(range(train_count, train_count + val_count)).add_column("label", [label] * val_count)
        test_split = shuffled.select(range(train_count + val_count, n)).add_column("label", [label] * test_count)
        
        train_list.append(train_split)
        val_list.append(val_split)
        test_list.append(test_split)

    # Flatten the list of lists of dicts into list of dicts
    train_combined = []
    validation_combined = []
    test_combined = []

    for train, val, test in zip(train_list, val_list, test_list):
        train_combined.extend(train.to_list())
        validation_combined.extend(val.to_list())
        test_combined.extend(test.to_list())

    # Load the combined lists into a DatasetDict
    final_datasetDict = DatasetDict({
        "train": Dataset.from_list(train_combined).shuffle(seed=split_seed),
        "validation": Dataset.from_list(validation_combined).shuffle(seed=split_seed),
        "test": Dataset.from_list(test_combined).shuffle(seed=split_seed)
    })


    return final_datasetDict


def get_min_count_in_labels(dataset: DatasetDict) -> int:
    min = float("inf")
    min_label = None
    for label, dataset_values in dataset.items():
        count = len(dataset_values)
        if count < min:
            min = count
            min_label = label

    print(f"Minimum count in labels is {min} for label {min_label}, downsampling other classes to this count...")
    return min

def load_dataset_from_source(labels: list[str] | None, dataset_folder_path: Path | str) -> DatasetDict:
    dataset_source_folderpath: Path = Path(dataset_folder_path) if isinstance(dataset_folder_path, str) else dataset_folder_path
    print(f"Loading dataset from source at {dataset_source_folderpath}...")

    datasetDict_per_label = DatasetDict()

    for folder in dataset_source_folderpath.iterdir(): # Loop over the labels, they're folders
        if labels is not None and folder.name not in labels:
            continue # Skip the labels we don't want, if none, we load all labels

        datasetDict_per_label[folder.name] = Dataset.from_dict({
            "image": [str(image_path) for image_path in folder.iterdir()]
        })

    return datasetDict_per_label

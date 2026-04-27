from pathlib import Path
from datasets import DatasetDict, Dataset
from PIL import Image, ImageOps, ImageChops
import numpy as np

class LazyLoadingImageDataset(Dataset):
    def __init__(self, dataset: Dataset, size: tuple = (224, 224), padding: bool = False):
        self.dataset: Dataset = dataset
        self.size: tuple = size
        self.padding_transformation: bool = padding

    def __len__(self):
        return len(self.dataset)
    
    def apply_augmentation(self, row):
        image = Image.open(row["image_with_augmentation_params"]["image"])
        params = row["image_with_augmentation_params"]

        if params["rotation"] != 0:
            image = image.rotate(params["rotation"], resample=Image.BILINEAR)

        if params["flipping"]:
            image = ImageOps.mirror(image)

        if params["inverting"]:
            image = ImageOps.invert(image)

        if params["vertical_shift"] != 0 or params["horizontal_shift"] != 0:
            shift_x = int(params["horizontal_shift"] * image.width)
            shift_y = int(params["vertical_shift"] * image.height)
            image = ImageChops.offset(image, shift_x, shift_y)

        return image
    
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

        image = self.apply_augmentation(row)

        
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
                           split_seed: int = 42, sample_seed: int = 42, augmentation_params: dict | None = None) -> DatasetDict:
    
    datasetDict_per_label: DatasetDict = load_dataset_from_source(labels, dataset_folder_path)
    augmented_datasetDict_per_label: DatasetDict = augment_dataset(datasetDict_per_label, augmentation_params) if augmentation_params is not None else datasetDict_per_label
    downsample_target: int = get_min_count_in_labels(augmented_datasetDict_per_label)
    datasetDict_per_label_downsampled: DatasetDict = downsample_dataset(augmented_datasetDict_per_label, sample_size=downsample_target, sample_seed=sample_seed)
    split_datasetDict = split_dataset(datasetDict_per_label_downsampled, split_ratio=split_ratio, split_seed=split_seed)
    return split_datasetDict
    
def augment_dataset(dataset: DatasetDict, augmentation_params: dict) -> DatasetDict:
    target_count = augmentation_params.get("target_count_per_label") # Required
    selection_seed = augmentation_params.get("selection_seed", 42) # Optional, default 42
    generation_seed = augmentation_params.get("generation_seed", 42) # Optional, default 42
    rotation_min = augmentation_params.get("rotation_min", 0) # Optional, default 0 degrees
    rotation_max = augmentation_params.get("rotation_max", 0) # Optional, default 0 degrees
    flipping_chance = augmentation_params.get("flipping_chance", 0) # Optional, default 0
    inverting = augmentation_params.get("inverting", False) # Optional, default False
    shift_range = augmentation_params.get("shift_range", 0) # Optional, default 0, is a 0-1 float representing max allowable shift in both dimensions

    dataset_entries_lists = {label: dataset_values.to_list() for label, dataset_values in dataset.items()}

    # convert to new type
    for label, item_list in dataset_entries_lists.items():
        dataset_entries_lists[label] = [{
            "image_with_augmentation_params": {
                "image": item["image"],
                "rotation": 0,
                "flipping": False,
                "inverting": False,
                "vertical_shift": 0,
                "horizontal_shift": 0
            }
        } for item in item_list]

    # add the augmentation data
    for label, item_list in dataset_entries_lists.items():
        if len(item_list) >= target_count:
            continue # Do not augment, we have enough


        np.random.seed(selection_seed)
        augmentation_count = target_count - len(item_list)
        selected_items_idx = []
        for i in range(augmentation_count // len(item_list)):
            idx = np.random.choice(len(item_list), size=len(item_list), replace=False)
            selected_items_idx.extend(idx)
        if augmentation_count % len(item_list) > 0:
            idx = np.random.choice(len(item_list), size=augmentation_count % len(item_list), replace=False)
            selected_items_idx.extend(idx)


        selected_items_idx = np.random.choice(len(item_list), size=augmentation_count, replace=True)
        np.random.seed(generation_seed)

        rotation_signs = np.random.choice([-1, 1], size=augmentation_count)

        flipping_factor = 1 if np.random.rand() < flipping_chance else 0
        if label == "Chest":
            flipping_factor = 0

        augmented_items = [{
            "image_with_augmentation_params": {
                "image": item_list[idx]["image_with_augmentation_params"]["image"],
                "rotation": np.random.uniform(rotation_min, rotation_max) * rotation_signs[i],
                "flipping": flipping_factor == 1,
                "inverting": inverting and np.random.choice([True, False]),
                "vertical_shift": np.random.uniform(-shift_range, shift_range),
                "horizontal_shift": np.random.uniform(-shift_range, shift_range)
            }
        } for idx in selected_items_idx]

        dataset_entries_lists[label].extend(augmented_items)

    # convert back to DatasetDict
    dataset = DatasetDict()
    for label, item_list in dataset_entries_lists.items():
        # print(F"sample augmented item for label {label}: {item_list[len(item_list)-1]}") # Print an example of the augmented data format
        # print(f"Total items for label {label} after augmentation: {len(item_list)}")
        dataset[label] = Dataset.from_list(item_list)


    return dataset

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
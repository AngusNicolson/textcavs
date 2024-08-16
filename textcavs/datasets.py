

from typing import Callable, Optional, Union, Sequence
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms as T

from textcavs.utils import MIMIC_ROOT


def get_mimic_dataset(split, transform=None):
    data_dir = f"{MIMIC_ROOT}"
    labels_to_use = ('No Finding', 'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity', 'Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices')
    dataset = MimicDataset(data_dir, split=split, transform=transform, labels_to_use=labels_to_use)
    return dataset


class MimicDataset(Dataset):
    def __init__(
            self,
            data_dir: Union[str, Path],
            split: str = "train",
            labels_name: str = "chexpert",
            labels_to_use: Sequence = ('Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity', 'Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices'),
            transform: Optional[Callable] = None,
            use_resized: bool = True,
            uncertain_impute=0,
    ):
        self.data_dir = Path(data_dir)
        self.labels_to_use = labels_to_use
        self.transform = transform
        self.use_resized = use_resized
        self.split = split
        self.labels_name = labels_name
        self.uncertain_impute = uncertain_impute

        split_path = self.data_dir / "mimic-cxr-2.0.0-split.csv.gz"
        split_df = pd.read_csv(split_path)
        # They use validate instead of val
        # But I want to use val in the rest of the project
        if split == "val":
            split_df = split_df.loc[split_df.split == "validate"]
        else:
            split_df = split_df.loc[split_df.split == split]

        if split == "test":
            labels_path = self.data_dir / "mimic-cxr-2.1.0-test-set-labeled.csv"
        else:
            if labels_name == "negbio":
                labels_path = self.data_dir / "mimic-cxr-2.0.0-negbio.csv.gz"
            elif labels_name == "chexpert":
                labels_path = self.data_dir / "mimic-cxr-2.0.0-chexpert.csv.gz"
            else:
                raise ValueError(f"{labels_name} not recognised, must be one of negbio or chexpert")

        labels_df = pd.read_csv(labels_path)
        labels_df = labels_df.loc[labels_df.study_id.isin(split_df.study_id)]
        labels_df = labels_df.set_index("study_id")
        split_df = split_df.set_index("dicom_id")
        self.split_df = split_df
        self.labels_df = labels_df

        self.image_paths = self.get_image_paths_(use_resized, split)

        labels_arr = self.get_image_labels(self.labels_to_use)
        self.labels = labels_arr

    def get_image_labels(self, labels):
        img_dicom_ids = [path.stem for path in self.image_paths]
        img_study_ids = self.split_df.loc[img_dicom_ids].study_id.values
        labels_arr = self.labels_df.loc[img_study_ids, labels].values
        if self.uncertain_impute == 0:
            labels_arr = labels_arr == 1
        elif self.uncertain_impute == 1:
            labels_arr = (labels_arr == 1) | (labels_arr == -1)
        else:
            raise ValueError(f"self.uncertain_impute must be 0 or 1. Value: {self.uncertain_impute}")
        labels_arr = torch.from_numpy(labels_arr).float()
        return labels_arr

    def get_image_paths_(self, use_resized, split):
        if use_resized:
            image_paths = self.get_resized_image_paths(split)
        else:
            image_paths = self.get_base_image_paths()
        return image_paths

    def get_resized_image_paths(self, split):
        image_dir = self.data_dir / f"resized_files/{split}"
        if not image_dir.exists():
            raise FileNotFoundError(f"Directory {image_dir} does not exist!")
        image_paths = image_dir.glob("*")

        # Only use images we have labels for
        label_study_ids = self.labels_df.index.unique()
        dicom_ids = self.split_df.loc[
            self.split_df.study_id.isin(label_study_ids)].index.values
        dicom_ids = set(dicom_ids)
        image_paths = [
            path for path in image_paths if path.stem in dicom_ids
        ]
        return image_paths

    def get_base_image_paths(self):
        image_filenames_path = self.data_dir / "IMAGE_FILENAMES"
        with open(image_filenames_path, "r") as fp:
            image_paths = fp.read().split("\n")
        # remove empty line(s) at end of file
        while image_paths[-1] == "":
            image_paths = image_paths[:-1]

        # Only use images we have labels for
        label_study_ids = self.labels_df.index.unique()
        image_paths = [self.data_dir / path for path in image_paths if
                       int(path.split("/")[-2][1:]) in label_study_ids]
        return image_paths

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path)
        img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        labels = self.labels[idx]

        return img, labels

    def __len__(self):
        return len(self.image_paths)


class BiasedMimicDataset(MimicDataset):
    def __init__(self, label_subset, bias_target="Pleural Effusion", bias_source="Support Devices", **kwargs):
        super().__init__(**kwargs)
        #self.labels_df, self.labels, self.image_paths
        self.labels_to_use = label_subset

        # Remove rows which contain the target class and not the bias class
        rows_to_remove = (self.labels_df.loc[:, bias_target] == 1) & ~(self.labels_df.loc[:, bias_source] == 1)
        self.labels_df = self.labels_df.loc[~rows_to_remove]

        # Remove rows which contain no labels for label_subset
        no_label_rows = (~(self.labels_df.loc[:, label_subset] == 1)).all(axis=1)
        self.labels_df = self.labels_df.loc[~no_label_rows]
        self.image_paths = self.get_image_paths_(
            self.use_resized,
            self.split
        )
        self.labels = self.get_image_labels(self.labels_to_use)


def main():
    transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])

    dataset = BiasedMimicDataset(
        data_dir="data/mimic-cxr-jpg/2.0.0",
        transform=transform,
        labels_to_use=['No Finding', 'Atelectasis', 'Cardiomegaly', 'Edema', 'Pleural Effusion'],
        label_subset=['No Finding', 'Atelectasis', 'Cardiomegaly', 'Edema', 'Pleural Effusion'],
        labels_name="chexpert",
        bias_source='Support Devices',
        bias_target='Pleural Effusion',
    )
    img, labels = dataset[0]
    print(img.shape, labels)

    dataset = MimicDataset("data/mimic-cxr-jpg/2.0.0", transform=transform)
    img, labels = dataset[0]
    print(img.shape, labels)
    img, labels = dataset[5]
    print(img.shape, labels)

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    for imgs, labels in dataloader:
        break
    print(imgs.shape, labels.shape)

    print("Done!")


if __name__ == "__main__":
    main()

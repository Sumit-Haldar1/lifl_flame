import logging
from typing import Any, Dict

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import NaturalIdPartitioner

logger = logging.getLogger(__name__)

class FEMNIST(Dataset):
    def __init__(self, data_dir, meta_dir, partition_id, dataset: str = "train", transform=None):
        self.transform = transform

        fds = FederatedDataset(
            dataset="flwrlabs/femnist",
            partitioners={"train": NaturalIdPartitioner(partition_by="writer_id")},
        )

        if dataset != "train":
            logger.warning("Only 'train' split exists; using 'train' instead.")

        ds = fds.load_partition(partition_id=partition_id, split="train")
        self.ds = ds

        if self.transform is None:
            self.transform = transforms.ToTensor()

        logger.info("Loaded FEMNIST partition_id=%d with %d samples", partition_id, len(self.ds))

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        example: Dict[str, Any] = self.ds[idx]

        img = example["image"]
        label = int(example.get("character", example.get("label", example.get("y", example.get("target")))))

        img = self.transform(img)

        # Ensure 3 channels (ResNet expects RGB)
        if isinstance(img, torch.Tensor):
            if img.ndim == 3 and img.shape[0] == 1:
                img = img.expand(3, img.shape[1], img.shape[2])
            elif img.ndim == 2:
                img = img.unsqueeze(0).expand(3, img.shape[0], img.shape[1])

        return img, label

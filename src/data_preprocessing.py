import deeplake
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

class PlantVillageDataset(Dataset):
    def __init__(self, subset='without-augmentation', transform=None):
        self.subset = subset
        self.transform = transform
        self.ds = deeplake.load(f'hub://activeloop/plantvillage-{subset}')
        self.classes = self.get_class_labels()
        self.label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted(set(self.ds.labels[:].numpy().flatten())))}

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        image = sample.tensors['images'].numpy()
        label = sample.tensors['labels'].numpy()
        
        # Ensure image is in the correct format (H x W x C)
        if image.ndim == 2:
            image = np.expand_dims(image, axis=2)
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        elif image.shape[2] != 3:
            raise ValueError(f"Unexpected number of channels: {image.shape[2]}")
        
        # Ensure image is uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        # Convert to PIL Image
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        # Map the old label to the new label
        new_label = self.label_mapping[label.item()]
        label = torch.tensor(new_label, dtype=torch.int64)
        
        return image, label

    def get_class_labels(self):
        return sorted(list(set(self.ds.labels[:].numpy().flatten())))

    def get_unique_labels(self):
        return set(self.label_mapping.values())
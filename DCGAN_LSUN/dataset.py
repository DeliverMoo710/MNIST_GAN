import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class LSUNDataset(Dataset):
    def __init__(self, dataset,
                batch_size: int = 32,
                transform = None,
                im_size: int = 64):
        self.ds = dataset
        self.batch_size = batch_size
        
        # Default transforms
        self.transform = transform or transforms.Compose([
            transforms.Resize((im_size, im_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        img = self.ds[idx]['pixel_values']
        return self.transform(img)

    def get_dataloader(self, shuffle: bool = True) -> DataLoader:
        return DataLoader(self, batch_size=self.batch_size, shuffle=shuffle)
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms

# dataset module
class CIFARDataset(Dataset):
    '''
    downloads dataset, performs splitting and transformation, and returns dataloaders
    '''
    def __init__(self, root = './data', download = True, transform = None):
        # download mnist dataset
        self.cifar = CIFAR10(root = root, download = download)

        # default transformation if no specific transformation is provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768))
            ])
        else:
            self.transform = transform

        self.indices = list(range(len(self.cifar)))

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        img, _ = self.cifar[self.indices[idx]]
    
        if self.transform:
            img = self.transform(img)

        return img
    
    def get_dataloader(self, batch_size = batch_size, shuffle = True):
        return DataLoader(self, batch_size = batch_size, shuffle = shuffle)
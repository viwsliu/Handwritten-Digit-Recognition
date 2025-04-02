from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(batch_size=64):
    transform = transforms.ToTensor()
    train_loader = DataLoader(
        datasets.MNIST(root='./data', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        datasets.MNIST(root='./data', train=False, download=True, transform=transform),
        batch_size=1000, shuffle=False
    )
    return train_loader, test_loader
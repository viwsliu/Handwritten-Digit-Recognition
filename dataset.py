from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(batch_size=64): # creates two PyTorch DataLoaders: one for training, one for testing. each step feeds 64 images to the model
    transform = transforms.ToTensor() # converts images from PIL format to a PyTorch tensor [0-1 floats] from 0-255 pixel values
    train_loader = DataLoader(
        datasets.MNIST(root='./data', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True
    )
    # downloads training split of MNIST (60,000 images), applies the transform to each image
    # wraps dataset in a DataLoader that shuffles and batches it

    test_loader = DataLoader(
        datasets.MNIST(root='./data', train=False, download=True, transform=transform),
        batch_size=1000, shuffle=False
    )
    # loads the test split, uses a bigger batch size (1000) for faster evaluation
    # does not shuffle, so predictions align with original test labels
    
    return train_loader, test_loader
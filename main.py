import torch
from model import DigitNet
from dataset import get_data_loaders
from train import train
from test import test
import matplotlib.pyplot as plt #for plotting predictions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # picks GPU if available
train_loader, test_loader = get_data_loaders() # loads MNIST training and test data
model = DigitNet().to(device) # creates a new DigitNet model and moves it to GPU/CPU
train(model, train_loader, device) 
test(model, test_loader, device) #tests, prints accuracy

def show_predictions(model, test_loader, device):
    model.eval()
    examples = enumerate(test_loader)
    _, (example_data, example_targets) = next(examples)
    with torch.no_grad(): # predicts labels for the batch of test images
        preds = model(example_data.to(device)).argmax(dim=1) # argmax(dim=1) selects the digit with the highest score
    plt.figure(figsize=(10, 4)) # displays 6 test images, shows the true and predicted label for each one, saves the plot as output.png
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.imshow(example_data[i][0], cmap='gray')
        plt.title(f"True: {example_targets[i]}\nPred: {preds[i].item()}")
        plt.axis("off")
    plt.tight_layout()
    # plt.show()
    plt.savefig("output.png")

torch.save(model.state_dict(), "model.pth")
show_predictions(model, test_loader, device)
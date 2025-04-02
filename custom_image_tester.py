import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from model import DigitNet  # import your model definition

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DigitNet().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(),             # ensure 1 channel
    transforms.Resize((28, 28)),        # resize to MNIST format
    transforms.ToTensor(),              # convert to [0,1]
    transforms.Normalize((0.1307,), (0.3081,))  # same normalization as MNIST
])

image_folder = "./my_digits"

with torch.no_grad():
    for file in os.listdir(image_folder):
        if file.endswith(".png") or file.endswith(".jpg"):
            img_path = os.path.join(image_folder, file)
            image = Image.open(img_path)
            image = transform(image).unsqueeze(0).to(device)  # add batch dim
            output = model(image)
            pred = output.argmax(dim=1).item()
            print(f"{file}: Predicted → {pred}")
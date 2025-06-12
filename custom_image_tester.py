import torch
import torchvision.transforms as transforms #provides common datasets, model architectures, and image transformations specifically for computer vision tasks
from PIL import Image # Python Imaging Library (PIL)
import os
from model import DigitNet  # import your model definition

# print(torch.cuda.is_available()) #check if Pytorch detects CUDA, uses cpu otherwise
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DigitNet().to(device) # instantiates CNN model
model.load_state_dict(torch.load("model.pth", map_location=device)) # loads pretrained weights from model.pth
model.eval() # switches model to inference mode (prediction only)
 
transform = transforms.Compose([
    transforms.Grayscale(), # converts image to grayscale
    transforms.Resize((28, 28)), # resize to MNIST format
    transforms.ToTensor(), # convert to tensor float values [0,1]
    transforms.Normalize((0.5,), (0.5,))
    # transforms.Normalize((0.1307,), (0.3081,))  # same normalization as MNIST
])

image_folder = "./uploaded_images/"

with torch.no_grad():
    for file in os.listdir(image_folder):
        if file.endswith(".png") or file.endswith(".jpg"):
            img_path = os.path.join(image_folder, file)
            image = Image.open(img_path)
            image = transform(image).unsqueeze(0).to(device)  # add batch dim
            output = model(image)
            pred = output.argmax(dim=1).item()
            print(f"{file}: Predicted → {pred}")

#loops thru each image in folder,
# load + preprocess image
# add batch dimensions
# passes it through the model and prints the predicted
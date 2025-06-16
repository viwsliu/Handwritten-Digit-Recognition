import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import csv
from model import DigitNet  # import your model

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
model = DigitNet().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# Define preprocessing (same shape and scale as MNIST)
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # match MNIST normalization
])

# Paths
image_folder = "./uploaded_images/"
label_csv = "./uploaded_images/answers.csv"

# Load labels from CSV
label_dict = {}
with open(label_csv, mode='r') as file:
    reader = csv.reader(file)
    for row in reader:
        filename, label = row
        label_dict[filename.strip()] = int(label.strip())

# Evaluate predictions
correct = 0
total = 0

with torch.no_grad():
    for filename, true_label in label_dict.items():
        img_path = os.path.join(image_folder, filename)
        if not os.path.exists(img_path):
            print(f"Missing image: {filename}")
            continue

        image = Image.open(img_path)
        image = transform(image).unsqueeze(0).to(device)

        output = model(image)
        pred = output.argmax(dim=1).item()

        print(f"{filename}: Predicted → {pred}, Actual → {true_label}")

        if pred == true_label:
            correct += 1
        total += 1

# Print summary accuracy
if total > 0:
    accuracy = 100.0 * correct / total
    print(f"\nAccuracy: {correct}/{total} correct ({accuracy:.2f}%)")
else:
    print("No valid images found.")

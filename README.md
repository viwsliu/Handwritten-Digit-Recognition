# Handwritten-Digit-Recognition
## Description:
This project uses a Convolutional Neural Network (CNN) built with PyTorch to recognize handwritten digits from the MNIST dataset. After training, it can also predict digits from custom images provided by the user. Note that <code>main.py</code> will run indefinitely for training unless interrupted by user. model.pth is saved after every cycle.

## Files in Directory:

<code>main.py</code> - Runs the full training, evaluation, and saving pipeline using model and data

<code>model.py</code> - Defines the architecture of the CNN (DigitNet)

<code>dataset.py</code> - Loads and preprocesses the MNIST dataset using PyTorch's DataLoader

<code>train.py</code> - Contains the training loop that updates model weights using backpropagation.

<code>test.py</code> - Evaluates the trained model's accuracy on the test dataset

## Setup:
Create a py environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install dependencies - requirements.txt: 
```bash
pip install -r requirements.txt
```

Run:
```bash
python ./main.py
```

To stop the python environment:
```bash
deactivate
```

## How it Works
### Step 1: Run main.py
```bash
python3 ./main.py
```

### Step 2: Setup
```bash
from model import DigitNet
from dataset import get_data_loaders
from train import train
from test import test
```
Running main.py will initialize by pulling in model structure from 'model.py', data from 'dataset.py', training logic from 'train.py', and testing logic from 'test.py'.

### Step 3: Configure Device and Model
```bash
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DigitNet().to(device)
```
It will build the CNN model (DigitNet) from 'model.py', and move the model to the appropriate device.

### Step 4: Load Model
```bash
if os.path.exists("model.pth"):
  model.load_state_dict(torch.load("model.pth"))
```
If a previously saved, trained model is found in the directory, it will be used. A new model will be created otherwise.

### Step 5: Load Data
```bash
train_loader, test_loader = get_data_loaders()
```
Data will be pulled from 'dataset.py' using 'torchvision.datasets' and applies basic transforms. It then returns 'DataLoader' for training and testing.

### Step 6: Training
```bash
train(model, train_loader, device)
```
'train()' is then called from 'train.py', which loops training data for multiple epochs.
For each pass:s
 - Forward pass, making predictions
 - Calculate Loss
 - Backpropoagation, computing gradients
 - Optimizer, updating weights

### Step 7: Testing
Model switches to evaluation mode, looping through the test data, making predictions and comparing to actual labels. Calculates and returns accuracy.

### Step 8: Saving Model
```bash
torch.save(model.state_dict(), "model.pth")
```
Trained model weights are serialized to a <code>.pth</code> file

## Additional Info
A few topics I was able to learn about when working on this project:
Convolutional layers
Pooling layers
Forward Pass
Loss Function
Backpropagation
Training Loop (main.py)
Inference (Evaluation)
Model Persistence
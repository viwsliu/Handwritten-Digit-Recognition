# Handwritten-Digit-Recognition
## Description:
This project uses a Convolutional Neural Network (CNN) built with PyTorch to recognize handwritten digits from the MNIST dataset. After training, it can also predict digits from custom images provided by the user.

## Note:
Main.py will run indefinitely for training unless interrupted by user. model.pth is saved after every cycle.

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


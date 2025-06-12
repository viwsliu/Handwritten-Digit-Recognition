import torch.nn as nn #used for building neural networks

class DigitNet(nn.Module): 
    def __init__(self): # inherits from nn.Module, which is base class for all PyTorch models
        super(DigitNet, self).__init__() # initializes parent class
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), # first conv. layer. one input channel -> 16 output channels. kernal size 3x3, padding keeps output size same (28x28)
            nn.ReLU(), # activation function: keeps positives, zeroes out negatives
            nn.MaxPool2d(2), # downsamples by a factor of 2; output becomes 14×14
            nn.Conv2d(16, 32, 3, padding=1), # 16 input channels -> 32 output channels
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(), # converts [batch_size, 32, 7, 7] into [batch_size, 1568]
            nn.Linear(32 * 7 * 7, 128), # fully connected layer: 1568 inputs -> 128 neurons
            nn.ReLU(),
            nn.Linear(128, 10) # final output 10 neurons (for digits 0–9)
        )

    def forward(self, x):
        return self.net(x)
    

# note to self (summary of file):

# extracts features using 2 convolution + pooling layers
# flattens the result
# classifies using 2 fully connected layers
# outputs 10 scores (1 per digit)
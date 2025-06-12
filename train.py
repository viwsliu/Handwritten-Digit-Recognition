import torch
import torch.nn as nn # defines loss functions
import torch.optim as optim # provides optimizers like Adam or SGD

def train(model, train_loader, device):
    model.train() # switches model to training mode
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    #Adam: advanced optimizer
    #lr = learning rate
    criterion = nn.CrossEntropyLoss() 
    # CrossEntropyLoss: common loss function for classification (used when predicting classes like digits)
    for epoch in range(5): #per 5 training cycles, calculates total errors made
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device) # move input (x) and labels (y) to the GPU or CPU
            optimizer.zero_grad() # clear gradients from the previous step
            output = model(x) # forward pass: compute model predictions
            loss = criterion(output, y) # compute loss between predictions and true labels
            loss.backward() # backwards pass: calculate gradients
            optimizer.step() # update model weights using the optimizer
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
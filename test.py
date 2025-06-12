import torch

def test(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad(): # disables gradient tracking; speeds up evaluation
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            output = model(x) # gets predicted logits for each image
            pred = output.argmax(dim=1) # picks the index with the highest score
            correct += (pred == y).sum().item() # counts how many were correct
            total += y.size(0) # adds the number of test samples in the batch
    print(f"Test Accuracy: {correct / total * 100:.2f}%")
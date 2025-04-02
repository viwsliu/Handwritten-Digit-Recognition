import struct
import numpy as np
import matplotlib.pyplot as plt

def read_images(path):
    with open(path, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, rows, cols)
    return images

def read_labels(path):
    with open(path, 'rb') as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

def show_images(images, labels, count=6):
    plt.figure(figsize=(10, 4))
    for i in range(count):
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(f"Label: {labels[i]}")
        plt.axis('off')
    plt.tight_layout()
    # plt.show()
    plt.savefig("output.png")

if __name__ == "__main__":
    image_path = "data/MNIST/raw/train-images-idx3-ubyte"
    label_path = "data/MNIST/raw/train-labels-idx1-ubyte"

    images = read_images(image_path)
    labels = read_labels(label_path)

    show_images(images, labels)
import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset

class MNISTDataset(torchvision.datasets.MNIST):
    def __init__(self, transform):
        super().__init__(
            "./dataset/mnist",
            train = True,
            download = True,
            transform = transform
        )

    def __getitem__(self, item):
        return super().__getitem__(item)[0], super().__getitem__(item)[1] 
        
class PixelClassificationData(Dataset):
    def __init__(self, pixels, labels):
        self.pixels = pixels
        self.labels = labels
        
    def __len__(self):
        return self.pixels.shape[0]
    
    def __getitem__(self, idx):
        return self.pixels[idx], self.labels[idx]
    
# def visualize_dataset(dataset, idx: int):
#     img, labels = dataset[idx]
#     img = img.numpy().transpose(1,2,0)
    
#     plt.figure(figsize=(10, 5))

#     plt.subplot(1, 2, 1)
#     plt.imshow(img)
#     plt.title('Image')

#     plt.subplot(1, 2, 2)
#     plt.imshow(labels)
#     plt.title('Semantic Segmentation Masks')
    
#     plt.show()

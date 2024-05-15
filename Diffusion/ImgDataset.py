import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import pandas as pd
from PIL import Image
import h5py
import os
import random
from tqdm import tqdm
import time
import datetime
import torch.optim as optim
from torch.nn import init

class ImageDataset(Dataset):
    def __init__(self, root_dir, split = 'train'):
        super(ImageDataset, self).__init__()
        
        self.root_dir = root_dir
        self.split = split
        self.transform = transforms.Compose([
                                            transforms.Resize((128, 128), interpolation=transforms.InterpolationMode.BILINEAR),  # Resize the image to a common size
                                            transforms.ToTensor(),           # Convert the image to a PyTorch tensor
                                            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))  # Normalize pixel values
                                            ])
        self.image_names = os.listdir(os.path.join(root_dir,self.split))
        
    def __getitem__(self, index):
        image_name = self.image_names[index]
        img_path = os.path.join(self.root_dir, self.split ,image_name)
#         img = cv2.imread(img_path)
        img = Image.open(img_path).convert("RGB")
#         plt.imshow(img)
#         print(img.shape)
#         plt.show()
        img = self.transform(img)
        
        return img

    def __len__(self):
        return len(self.image_names)
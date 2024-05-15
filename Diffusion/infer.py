# %% [code]
import sys
import os
import random
import json
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
import h5py
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import matplotlib.pyplot as plt
from torch.nn import init
import torch.nn.functional as F
import cv2
import shutil
import math
import time
import argparse
from args import args
from unet_dissusion import DiffusionModel
from slot_autoencoder import SlotAttentionAutoEncoder
# %% [code]

input_dir = args.input_dir
model_path = args.model_path
part = args.part
output_dir = args.output_dir

# %% [code]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% [code]
class PARTNET(Dataset):
    def __init__(self, root_dir):
#         super(ImageDataset, self)._init_()
        
        self.root_dir = root_dir
#         self.split = split
        self.transform = transforms.Compose([
                                            transforms.CenterCrop(128),  # Resize the image to a common size
                                            transforms.ToTensor(),           # Convert the image to a PyTorch tensor
                                            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))  # Normalize pixel values
                                            ])
        self.image_names = os.listdir(root_dir)
        
    def __getitem__(self, index):
        image_name = self.image_names[index]
        img_path = os.path.join(self.root_dir, image_name)
#         img = cv2.imread(img_path)
        img = Image.open(img_path).convert("RGB")
#         plt.imshow(img)
#         print(img.shape)
#         plt.show()
        img = self.transform(img)
        sample = {'image_name': image_name, 'image': img}
        return sample

    def __len__(self):
        return len(self.image_names)

# %% [code]
test_dataset = PARTNET(input_dir)
test_loader = DataLoader(test_dataset, batch_size=20,num_workers = 4, shuffle=True)

# %% [code]

# %% [code]

    
from vae import VAE
import torch
vae = VAE().to(device)
# load the c h e c k p o i n t
# ( replace with your c h e c k p o i n t path )
ckpt = torch.load (args.vae_ckpt)
vae.load_state_dict(ckpt)
for param in vae.parameters():
    param.requires_grad = False
    

# %% [code]
if part == 1:
    print('Part 1')
    resolution = (128,128)
    model = SlotAttentionAutoEncoder(resolution, 11, 2, 128).to(device)
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    denormalize = transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])
#         img = denormalize(comp_img)
    with torch.no_grad():
        model.eval()
        for i, sample in enumerate(test_loader):
#             print('Batch:', i)
            image = sample['image'].to(device)
            image_names = sample['image_name']
            B = image.shape[0]
            recon_combined, recons, masks, _ , slots = model(image)
#             print(recons.shape)
#             print(masks.shape)
#             print(masks)
            for j in range(B):
                recombined_image = denormalize(recon_combined[j].cpu())
                recombined_image = transforms.ToPILImage()(recombined_image)
                recombined_image.save(os.path.join(output_dir, image_names[j]))
#                 cv2.imwrite(os.path.join(output_dir, image_names[j]), recon_combined[j].permute(1,2,0).cpu().numpy())
                for s in range(11):
                    slot_name = image_names[j][:-4] + f'_{s}.jpg'
#                     slot_img = denormalize(masks[j][s].cpu().permute(2,0,1))
                    slot_img = masks[j][s].permute(2,0,1)*255
                    slot_img = transforms.ToPILImage()(slot_img)
                    slot_img.save(os.path.join(output_dir, slot_name))
#                     cv2.imwrite(os.path.join(output_dir, slot_name), recons[j][s].cpu().numpy())

            del image, recon_combined, recons, masks, _, slots
#             if i>3:
#                 break

# %% [code]
if part == 2:
    model = DiffusionModel().to(device)
#     if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    beta_start = 0.0015
    beta_end = 0.0195
    betas= torch.linspace(beta_start, beta_end, 1000)
    sqrt_betas = torch.sqrt(betas)
    alphas = 1.0 - betas
    sqrt_alphas = torch.sqrt(alphas)
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    denormalize = transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])
    
    with torch.no_grad():
        model.eval()
        for i, sample in enumerate(test_loader):
#             print('Batch:', i)
            image = sample['image'].to(device)
            image_names = sample['image_name']
            B = image.shape[0]
            recon_combined, recons, masks, _ , slots = model.module.slot_attn(image)
            x = torch.randn((B, 3, 32, 32), device=device)
            for t in range(999, -1, -1):
                if t > 0:
                    z = torch.randn((B, 3, 32, 32), device=device)
                else:
                    z = torch.zeros((B, 3, 32, 32), device=device)
                    
                t_emb = model.module.time_embedding(torch.tensor([t], device=device))
                unet_out = model.module.unet(x, t_emb, slots)
                x = ((x - (((1 - alphas[t]) * unet_out) / (sqrt_one_minus_alphas_cumprod[t]))) / (sqrt_alphas[t])) + (sqrt_betas[t] * z)
            rec = vae.decode(x)
            for j in range(B):
                recombined_image = denormalize(rec[j].cpu())
                recombined_image = transforms.ToPILImage()(recombined_image)
                recombined_image.save(os.path.join(output_dir, image_names[j]))
                
                for s in range(11):
                    slot_name = image_names[j][:-4] + f'_{s}.jpg'
#                     slot_img = denormalize(slots[j][s].cpu().permute(2,0,1))
                    slot_img = slots[j][s].view(8,8)
#                     slot_img = masks[j][s].permute(2,0,1)*255
                    slot_img = transforms.ToPILImage()(slot_img)
                    slot_img = slot_img.resize((128, 128))
                    slot_img.save(os.path.join(output_dir, slot_name))
                
            
#             if i > 3:
#                 break
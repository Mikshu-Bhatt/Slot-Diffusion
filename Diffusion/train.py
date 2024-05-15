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
import datetime
import argparse
from args import args
from unet_diffusion import DiffusionModel
import pytorch_warmup as warmup
from slot_autoencoder import SlotAttentionAutoEncoder
from vae import VAE
from ImgDataset import ImageDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = ImageDataset(args.input_dir,'train')
val_dataset = ImageDataset(args.input_dir,'val')

batch_size = args.batch_size
train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers = 4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers = 4, shuffle=True)

model = DiffusionModel().to(device)

if torch.cuda.device_count() > 1:
    print('multiple devices')
    model = nn.DataParallel(model).to(device)
# model.load_state_dict(torch.load('./tmp/model6.ckpt')['model_state_dict'])

criterion = nn.MSELoss()

params = [{'params': model.parameters()}]

# train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=opt.batch_size,
#                         shuffle=True, num_workers=opt.num_workers)

optimizer = torch.optim.Adam(params, lr=1e-3,weight_decay = 1e-5)
num_steps = len(train_loader) * 40
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 25, verbose=False)
warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=2500)
iters = len(train_loader)

epochs = 150
i = 0
start = time.time()
train_loss_lst = []
val_loss_lst = []
best_val_loss = np.inf
with torch.autograd.set_detect_anomaly(True):
    for epoch in range(epochs):
        print(f'epoch is {epoch}')
        start = time.time()
        model.train()
        train_loss = 0
        val_loss = 0
        
        for idx, img in enumerate(train_loader):
            i += 1
            if idx == 50:
                print(f'time taken is {round((time.time() - start)/60,2)}')
    #         print(f'batch loading time {time.time() - start}')
            image = img.to(device)
    #         print(f' image shape {image.shape}')
#             t = torch.tensor([int(random.uniform(0, 1000))]).to(device)
            t = int(random.uniform(0, 1000))
            t_tensor = torch.tensor([t]).to(device)
#             start_f = time.time()
            z, noise = model(image, t, t_tensor)
#             print ("Forward Time: {}".format(datetime.timedelta(seconds=time.time() - start_f)))

    #             break
#             print('z shape:', z.shape)
#             print('noise shape:', noise.shape)
            loss = criterion(z, noise)
            if (idx+1)%625 == 0:
                print(f"Batch: {idx+1}, Loss: {loss}")
            train_loss += loss.item()

            del z, noise, image

            optimizer.zero_grad()
#             start_back = time.time()
            loss.backward()
#             print ("backward Time: {}".format(datetime.timedelta(seconds=time.time() - start_back)))
            optimizer.step()
            with warmup_scheduler.dampening():
                lr_scheduler.step(epoch + idx / iters)
#         scheduler.step()
        train_loss /= len(train_loader)
        train_loss_lst.append(train_loss)
        
        model.eval()
        with torch.no_grad():
            for idx, img in enumerate(val_loader):
                image = img.to(device)
                t = int(random.uniform(0, 1000))
                t_tensor = torch.tensor([t]).to(device)
                z, noise = model(image, t, t_tensor)
                loss = criterion(z, noise)
                val_loss += loss.item()
                del z, noise, image

            val_loss /= len(val_loader)
            val_loss_lst.append(val_loss)
            
        print ("Epoch: {},Train Loss: {}, Val Loss: {}, Time: {}".format(epoch, train_loss, val_loss,
            datetime.timedelta(seconds=time.time() - start)))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({'model_state_dict': model.state_dict(),}, args.model_path)

item = {'train loss' : train_loss_lst, 'val loss' : val_loss_lst}
df = pd.DataFrame(item)
df.to_csv(args.csv_path)
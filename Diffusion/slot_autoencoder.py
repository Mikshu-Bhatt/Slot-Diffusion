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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def layer_normalization(dim):
        return nn.BatchNorm2d(dim)

class SlotAttention(nn.Module):
    def __init__(self, num_slots, input_dim, num_iterations=3, eps=1e-8, intermediate_dim=128):
        super(SlotAttention, self).__init__()
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.stability_term = eps
        self.scale = input_dim ** -0.5

        # Slot initialization parameters
        self.slot_mean = nn.Parameter(torch.randn(1, 1, input_dim))
        self.slot_log_std = nn.Parameter(torch.zeros(1, 1, input_dim))
        init.xavier_uniform_(self.slot_log_std)

        # Transformations for query, key, and value
        self.query_transform = nn.Linear(input_dim, input_dim)
        self.key_transform = nn.Linear(input_dim, input_dim)
        self.value_transform = nn.Linear(input_dim, input_dim)

        # GRU for updating slots
        self.gru_update = nn.GRUCell(input_dim, input_dim)

        # Feedforward layers for slots
        expanded_dim = max(input_dim, intermediate_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(input_dim, expanded_dim),
            nn.ReLU(inplace=True),
            nn.Linear(expanded_dim, input_dim)
        )

        # Normalization layers
        self.input_norm = nn.LayerNorm(input_dim)
        self.slot_norm = nn.LayerNorm(input_dim)
        self.pre_ff_norm = nn.LayerNorm(input_dim)

    def forward(self, inputs, num_slots_override=None):
        batch_size, num_elements, dim = inputs.shape
        num_slots = num_slots_override if num_slots_override is not None else self.num_slots
        
        mean = self.slot_mean.expand(batch_size, num_slots, -1)
        std_dev = self.slot_log_std.exp().expand(batch_size, num_slots, -1)

        slots = mean + std_dev * torch.randn(mean.shape, device=inputs.device, dtype=inputs.dtype)

        inputs = self.input_norm(inputs)
        keys = self.key_transform(inputs)
        values = self.value_transform(inputs)

        for _ in range(self.num_iterations):
            previous_slots = slots

            slots = self.slot_norm(slots)
            queries = self.query_transform(slots)

            dot_products = torch.einsum('bmd,bnd->bmn', queries, keys) * self.scale
            attention = dot_products.softmax(dim=1) + self.stability_term

            attention = attention / attention.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bnd,bmn->bmd', values, attention)

            slots = self.gru_update(
                updates.reshape(-1, dim),
                previous_slots.reshape(-1, dim)
            )

            slots = slots.reshape(batch_size, num_slots, dim)
            slots = slots + self.feedforward(self.pre_ff_norm(slots))

        return slots

    

def build_grid(dimensions):
    """Generates a spatial grid with normalized coordinates and their complementary values."""
    axes = [np.linspace(0.0, 1.0, num=d) for d in dimensions]
    meshgrid = np.meshgrid(*axes, sparse=False, indexing="ij")
    meshgrid = np.stack(meshgrid, axis=-1).reshape(dimensions[0], dimensions[1], -1)
    meshgrid = np.concatenate([meshgrid, 1 - meshgrid], axis=-1)
    return torch.from_numpy(meshgrid.astype(np.float32)).to(device)

class SoftPositionEmbed(nn.Module):
    def __init__(self, feature_size, grid_size):

        super(SoftPositionEmbed, self).__init__()
        self.positional_embedding = nn.Linear(4, feature_size, bias=True)
        self.spatial_grid = build_grid(grid_size)

    def forward(self, features):
        embedded_grid = self.positional_embedding(self.spatial_grid)
        return features + embedded_grid

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.resid_layer = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_channels)])
    
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        for layer in self.resid_layer:
            x = layer(x)

        x += identity
        x = self.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self, block, layers):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.layers = nn.ModuleList([
        nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
         nn.BatchNorm2d(16),
         nn.ReLU(inplace=True),

         self.make_layer(block, 64, layers[0], stride=2)])

    def make_layer(self, block, out_channels, blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
        return x

def ResNet18():
    return ResNet(ResBlock, [2])


class Encoder(nn.Module):
    def __init__(self, resolution, hid_dim):
        super().__init__()
        self.resnet = ResNet18()
        self.encoder_pos = SoftPositionEmbed(64, (64,64))

    def forward(self, x):
#         print(f'encoder ip shape {x.shape}')
        x = self.resnet(x)
#         print(f'time after resnet {time.time() - start}')
#         print(f' shape after resnet {x.shape}')
        x = x.permute(0,2,3,1)
        x = self.encoder_pos(x)
#         print(f'encoder after pos emb shape {x.shape}')
        x = torch.flatten(x, 1, 2)
#         print(f'encoder op shape {x.shape}')
        return x

class Decoder(nn.Module):
    def __init__(self, hid_dim, resolution):
        super().__init__()
        self.deconv = nn.ModuleList([nn.ConvTranspose2d(64, 64, 5, stride=(2, 2), padding=2, output_padding=1)
       ,nn.ConvTranspose2d(64, 64, 5, stride=(2, 2), padding=2, output_padding=1),
        nn.ReLU(inplace=True)
        ,nn.ConvTranspose2d(64, 64, 5, stride=(2, 2), padding=2, output_padding=1),
        nn.ReLU(inplace=True)
        ,nn.ConvTranspose2d(64, 64, 5, stride=(2, 2), padding=2, output_padding=1),
        nn.ReLU(inplace=True)
        ,nn.ConvTranspose2d(64, 64, 5, stride=(1, 1), padding=2),
        nn.ReLU(inplace=True)
        ,nn.ConvTranspose2d(64, 4, 3, stride=(1, 1), padding=1)])
        self.decoder_initial_size = (8, 8)
        self.decoder_pos = SoftPositionEmbed(64, self.decoder_initial_size)
        self.resolution = resolution

    def forward(self, x):
        x = self.decoder_pos(x)
        x = x.permute(0,3,1,2)
        for layer in self.deconv:
            x = layer(x)
#         print(f'shape after deconv6 {x.shape}')
        x = x[:,:,:self.resolution[0], :self.resolution[1]]
        x = x.permute(0,2,3,1)
        return x

class SlotAttentionAutoEncoder(nn.Module):
    def __init__(self, resolution, num_slots, num_iterations, hid_dim):

        super().__init__()
        self.hid_dim = hid_dim
        self.resolution = resolution
        self.num_slots = num_slots
        self.num_iterations = num_iterations

        self.encoder_cnn = Encoder(self.resolution, self.hid_dim)
        self.decoder_cnn = Decoder(self.hid_dim, self.resolution)

        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 64)

        self.slot_attention = SlotAttention(
            num_slots=self.num_slots,
            input_dim=64,
            num_iterations = self.num_iterations,
            eps = 1e-8, 
            intermediate_dim = 128)

    def forward(self, image=None, slots=None):
        # `image` has shape: [batch_size, num_channels, width, height].
        if image == None:
            N = slots.shape[0]
        else:
            N = image.shape[0]
        if slots == None:
            # Convolutional encoder with position embedding.
            x = self.encoder_cnn(image) 
    #         print(f'time after encoder {time.time() - start}')
            x = nn.LayerNorm(x.shape[1:]).to(device)(x)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)  # Feedforward network on set.
            # `x` has shape: [batch_size, width*height, input_size].

            # Slot Attention module.
            slots = self.slot_attention(x)
        slt = slots.clone()

        slots = slots.reshape((-1, slots.shape[-1])).unsqueeze(1).unsqueeze(2)
        slots = slots.repeat((1, 8, 8, 1))
        
        # `slots` has shape: [batch_size*num_slots, width_init, height_init, slot_size].
        x = self.decoder_cnn(slots)

        recons, masks = x.reshape(N, -1, x.shape[1], x.shape[2], x.shape[3]).split([3,1], dim=-1)

        masks = nn.Softmax(dim=1)(masks)
        recon_combined = torch.sum(recons * masks, dim=1)  # Recombine image.
        recon_combined = recon_combined.permute(0,3,1,2)

        return recon_combined, recons, masks, slots, slt

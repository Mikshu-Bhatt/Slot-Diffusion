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
from args import args
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import math
from slot_autoencoder import SlotAttentionAutoEncoder
from vae import VAE

vae = VAE().to(device)
# load the c h e c k p o i n t
# ( replace with your c h e c k p o i n t path )
ckpt = torch.load (args.vae_ckpt)
vae.load_state_dict(ckpt)
for param in vae.parameters():
    param.requires_grad = False


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class TimeEmbedding(nn.Module):
    def __init__(self, num_features: int):
        super(TimeEmbedding, self).__init__()
        self.num_features = num_features
        
        # Initialize MLP components
        self.project_initial = nn.Linear(num_features // 4, num_features)
        self.project_final = nn.Linear(num_features, num_features)
        self.activation = Swish()
        
    def forward(self, time_tensor: torch.Tensor):

        embedding_size = self.num_features // 8
        scale_factor = math.log(10000) / (embedding_size - 1)
        position = torch.exp(torch.arange(embedding_size, device=time_tensor.device) * -scale_factor)
        
        # Generate temporal embeddings
        temp_embedding = time_tensor[:, None] * position[None, :]
        sinusoid_embedding = torch.cat((torch.sin(temp_embedding), torch.cos(temp_embedding)), dim=1)
        
        # Apply the multi-layer perceptron
        x = self.activation(self.project_initial(sinusoid_embedding))
        encoded_output = self.project_final(x)
        
        return encoded_output
    
class ResidualBlock(nn.Module):

    def __init__(self, channels_in: int, channels_out: int, channels_time: int,
                 groups: int = 32, dropout_rate: float = 0.1):
        super(ResidualBlock, self).__init__()
        
#         print('residual block')
        # Normalization and activation setup
        self.normalize1 = nn.GroupNorm(groups, channels_in)
        self.activate1 = Swish()
        # Initial convolution to transform input feature maps
        self.transform1 = nn.Conv2d(channels_in, channels_out, kernel_size=(3, 3), padding=(1, 1))

        # Second normalization and activation
        self.normalize2 = nn.GroupNorm(groups, channels_out)
        self.activate2 = Swish()
        # Second convolution for further transformation
        self.transform2 = nn.Conv2d(channels_out, channels_out, kernel_size=(3, 3), padding=(1, 1))

        # Shortcut connection setup
        self.shortcut = nn.Conv2d(channels_in, channels_out, kernel_size=(1, 1)) if channels_in != channels_out else nn.Identity()

        # Time embedding components
        self.embed_time = nn.Linear(channels_time, channels_out)
        self.activate_time = Swish()

        # Dropout to prevent overfitting
        self.apply_dropout = nn.Dropout(dropout_rate)

    def forward(self, input_tensor: torch.Tensor, time_tensor: torch.Tensor):
        # Process input through the first transformation
        processed = self.transform1(self.activate1(self.normalize1(input_tensor)))
        # Time-dependent embedding added to the transformed feature maps
        time_embedding = self.embed_time(self.activate_time(time_tensor))[:, :, None, None]
        enhanced = processed + time_embedding
        
        # Further processing with dropout and the second set of transformations
        output = self.transform2(self.apply_dropout(self.activate2(self.normalize2(enhanced))))

        # Adding the shortcut connection and returning the output
        return output + self.shortcut(input_tensor)
    
class DownResBlock(nn.Module):

    def __init__(self, channels_in: int, channels_out: int, channels_time: int,
                 groups: int = 32, dropout_rate: float = 0.1):
        super(DownResBlock, self).__init__()
        
        # First normalization and activation, followed by average pooling for downsampling
        self.normalize1 = nn.GroupNorm(groups, channels_in)
        self.activate1 = Swish()
        self.downsample = nn.AvgPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        # Second normalization and activation, followed by convolution
        self.normalize2 = nn.GroupNorm(groups, channels_in)  # Notice using channels_in here, same as input to conv2
        self.activate2 = Swish()
        self.transform = nn.Conv2d(channels_in, channels_out, kernel_size=(3, 3), padding=(1, 1))

        # Shortcut connection
        self.shortcut = nn.Conv2d(channels_in, channels_out, kernel_size=(1, 1)) if channels_in != channels_out else nn.Identity()

        # Time embedding components
        self.embed_time = nn.Linear(channels_time, channels_in)
        self.activate_time = Swish()

        # Dropout to prevent overfitting
        self.apply_dropout = nn.Dropout(dropout_rate)

    def forward(self, input_tensor: torch.Tensor, time_tensor: torch.Tensor):
        # Apply initial normalization, activation, and downsampling
        processed = self.downsample(self.activate1(self.normalize1(input_tensor)))
        # Time-dependent embedding added to the downsampled feature maps
        time_embedding = self.embed_time(self.activate_time(time_tensor))[:, :, None, None]
        enhanced = processed + time_embedding
        
        # Further processing with dropout and convolution
        output = self.transform(self.apply_dropout(self.activate2(self.normalize2(enhanced))))

        # Adding the shortcut connection and returning the output
        return output + self.shortcut(enhanced)
    
class UpResBlock(nn.Module):

    def __init__(self, channels_in: int, channels_out: int, channels_time: int,
                 groups: int = 32, dropout_rate: float = 0.1):
        super(UpResBlock, self).__init__()
        
        # First normalization and activation
        self.normalize1 = nn.GroupNorm(groups, channels_in)
        self.activate1 = Swish()

        # Group normalization and activation for second convolution layer
        self.normalize2 = nn.GroupNorm(groups, channels_in)
        self.activate2 = Swish()
        self.transform = nn.Conv2d(channels_in, channels_out, kernel_size=(3, 3), padding=(1, 1))

        # Shortcut connection
        self.shortcut = nn.Conv2d(channels_in, channels_out, kernel_size=(1, 1)) if channels_in != channels_out else nn.Identity()

        # Time embedding components
        self.embed_time = nn.Linear(channels_time, channels_in)
        self.activate_time = Swish()

        # Dropout to prevent overfitting
        self.apply_dropout = nn.Dropout(dropout_rate)

    def forward(self, input_tensor: torch.Tensor, time_tensor: torch.Tensor):
        # Upsample and process input through normalization and activation
        upsampled = F.interpolate(input_tensor, scale_factor=2, mode='nearest')
        processed = self.activate1(self.normalize1(upsampled))
        
        # Time-dependent embedding added to the processed feature maps
        time_embedding = self.embed_time(self.activate_time(time_tensor))[:, :, None, None]
        enhanced = processed + time_embedding
        
        # Further processing with dropout and convolution
        output = self.transform(self.apply_dropout(self.activate2(self.normalize2(enhanced))))

        # Adding the shortcut connection and returning the output
        return output + self.shortcut(enhanced)

    
class MultiHeadAttention(nn.Module):
    def __init__(self, in_features, head_num, d_k = 64, dropout=0.1):
        super().__init__()
        self.head_num = head_num
        self.in_features = in_features
        self.head_dim = in_features // head_num  # Dimension per head

        self.wq = nn.Linear(in_features, head_num * self.head_dim, bias=False)
        self.wk = nn.Linear(d_k, head_num * self.head_dim, bias=False)
        self.wv = nn.Linear(d_k, head_num * self.head_dim, bias=False)
        self.wo = nn.Linear(head_num * self.head_dim, in_features, bias=False)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, key, value, query, mask=None):
        # Project queries, keys and values
#         print(query.shape)
        N = query.shape[0]
        q = self.wq(query).reshape(query.shape[0], query.shape[1], self.head_num, self.head_dim)
        k = self.wk(key).reshape(key.shape[0], key.shape[1], self.head_num, self.head_dim)
        v = self.wv(value).reshape(value.shape[0], value.shape[1], self.head_num, self.head_dim)
        
#         print('q shape:', q.shape)
#         print('k shape:', k.shape)
#         print('k trans:', k.transpose(-2, -1).shape)

        # Attention scores
#         scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # Scaled dot-product attention
        scores = torch.einsum("nqhd,nkhd->nhqk", [q, k]) / (self.head_dim ** (0.5))
    
        attention = F.softmax(scores, dim=3)

        # Weighted sum of values
        out = torch.einsum("nhql,nlhd->nqhd", [attention, v]).reshape(N, -1, self.head_num * self.head_dim)

        # Final linear transformation
        out = self.wo(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, in_channels, out_channels, embed_size, heads, n_groups = 32):
        super().__init__()
        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1))
        self.self_attn = MultiHeadAttention(embed_size, heads, d_k=embed_size)
        # Change batchnorms to layernorms
        self.norm2 = nn.BatchNorm2d(in_channels)
        self.cross_attn = MultiHeadAttention(embed_size, heads)
        self.norm3 = nn.BatchNorm2d(in_channels)
        self.linear = nn.Linear(in_channels, out_channels)
        self.norm4 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1))
        
    def forward(self, x, slots):
        x_shape = x.shape
        x = self.conv1(self.norm1(x))
#         print('x shape:', x.shape)
        x1 = x.view(x_shape[0], x_shape[1], -1).permute(0, 2, 1)

        x2 = self.self_attn(x1, x1, x1)
#         print (f"self attn Time: {datetime.timedelta(seconds=time.time() - start)}")
#         print('x2 shape:', x2.shape)
        x2 = self.norm2(x2.permute(0,2,1).reshape(x_shape)).view(x_shape[0], x_shape[1], -1).permute(0,2,1)
        x2 += x1

        x3 = self.cross_attn(slots, slots, x2)
#         print (f"cross attn Time: {datetime.timedelta(seconds=time.time() - start)}")
#         print('x3 shape:', x3.shape)
        x3 = self.norm3(x3.permute(0,2,1).reshape(x_shape)).view(x_shape[0], x_shape[1], -1).permute(0,2,1)
        x3 += x2

        x3 = self.linear(x3)
#         x3_shape = x3.shape
#         print('x3 shape:', x3.shape)
        x3 = x3.permute(0, 2, 1).view(x_shape)
        x4 = x3 + self.norm4(x3)
        # reshape x
        x4 = self.conv2(x4)
        
        return x4
    
class Unet(nn.Module):
    def __init__(self, n_groups = 32):
        super(Unet, self).__init__()
        self.conv_in = nn.Conv2d(3, 64, kernel_size=(3, 3), padding = 1)
        
        self.down_dict = {
                         'r1' : ResidualBlock(64, 64, 64),
                         'r2' : ResidualBlock(64, 64, 64),
                         'd1' : DownResBlock(64, 64, 64),
                         'r3' : ResidualBlock(64, 128, 64),
                         't1' : TransformerBlock(128, 128, 128, 128//32),
                         'r4' : ResidualBlock(128, 128, 64),
                         't2' : TransformerBlock(128, 128, 128, 128//32),
                         'd2' : DownResBlock(128, 128, 64),
                        'r5' : ResidualBlock(128, 192, 64),
                        't3' : TransformerBlock(192, 192, 192, 192//32),
                        'r6' : ResidualBlock(192, 192, 64),
                        't4' : TransformerBlock(192, 192, 192, 192//32),
                        'd3' : DownResBlock(192, 192, 64),
                        'r7' : ResidualBlock(192, 256, 64),
                        't5' : TransformerBlock(256, 256, 256, 256//32),
                        'r8' : ResidualBlock(256, 256, 64),
                        't6' : TransformerBlock(256, 256, 256, 256//32),
                        'r9' : ResidualBlock(256, 256, 64),
                        't7' : TransformerBlock(256, 256, 256, 256//32),
                        'r10' : ResidualBlock(256, 256, 64)}
        for key, val in self.down_dict.items():
#             print(val)
            self.down_dict[key] = val.to(device)
        self.time_embedding = TimeEmbedding(64)
        self.out_keys = ['conv_in', 'r1', 'r2', 'd1', 't1', 't2', 'd2', 't3', 't4', 'd3', 't5', 't6']
        self.map_dict = {'r22':'conv_in',
                        'r21':'r1',
                        'r20':'r2',
                        'r19':'d1',
                        'r18':'t1',
                        'r17':'t2',
                        'r16':'d2',
                        'r15':'t3',
                        'r14':'t4',
                        'r13':'d3',
                        'r12':'t5',
                        'r11':'t6'}
        
        self.up_dict = {'r11' : ResidualBlock(512, 256, 64),
                        't8' : TransformerBlock(256, 256, 256, 256//32),
                       'r12' : ResidualBlock(512, 256, 64),
                        't9' : TransformerBlock(256, 256, 256, 256//32),
                        'r13' : ResidualBlock(448, 256, 64),
                        't10' : TransformerBlock(256, 256, 256, 256//32),
                        'u1' : UpResBlock(256, 256, 64),
                        'r14' : ResidualBlock(448, 192, 64),
                        't11' : TransformerBlock(192, 192, 192, 192//32),
                        'r15' : ResidualBlock(384, 192, 64),
                        't12' : TransformerBlock(192, 192, 192, 192//32),
                        'r16' : ResidualBlock(320, 192, 64),
                        't13' : TransformerBlock(192, 192, 192, 192//32),
                        'u2' : UpResBlock(192, 192, 64),
                        'r17' : ResidualBlock(320, 128, 64),
                        't14' : TransformerBlock(128, 128, 128, 128//32),
                        'r18' : ResidualBlock(256, 128, 64),
                        't15' : TransformerBlock(128, 128, 128, 128//32),
                        'r19' : ResidualBlock(192, 128, 64),
                        't16' : TransformerBlock(128, 128, 128, 128//32),
                        'u3' : UpResBlock(128, 128, 64),
                        'r20' : ResidualBlock(192, 64, 64),
                        'r21' : ResidualBlock(128, 64, 64),
                        'r22' : ResidualBlock(128, 64, 64)
                       }
        for key, val in self.up_dict.items():
            self.up_dict[key] = val.to(device)
        self.norm = nn.GroupNorm(n_groups, 64)
        self.conv_op = nn.Conv2d(64, 3, kernel_size=(3, 3), padding = 1)
        
    def forward(self, x, time_emb, slots):
        out_dict = {}
        x = self.conv_in(x)
        out_dict['conv_in'] = x
#         start = time.time()
        for key, value in self.down_dict.items():
#             print(key)
#             start_layer = time.time()
            if key[0] != 't':
                x = value(x, time_emb)
            else:
                x = value(x, slots)
            if key in self.out_keys:
                out_dict[key] = x

        for key, layer in self.up_dict.items():
#             print(key)
#             print(x.shape)
#             start_layer = time.time()
            if key in self.map_dict.keys():
                
                mapped_key = self.map_dict[key]
#                 print('x:', x.shape)
#                 print(f'{mapped_key} out:', out_dict[mapped_key].shape)
                x = torch.cat([x, out_dict[mapped_key]], dim=1)
            if key[0] != 't':
                x = layer(x, time_emb)
            else:
                x = layer(x, slots)
#             print (f"{key} Time: {datetime.timedelta(seconds=time.time() - start_layer)}")

        x = self.norm(x)
        x = self.conv_op(x)
#         print('x shape:', x.shape)
#         print()
        
        return x
    
class DiffusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.slot_attn = SlotAttentionAutoEncoder((128, 128), 11, 4, 128)
        self.vae = vae.to(device)
        self.unet = Unet().to(device)
        self.time_embedding = TimeEmbedding(64)
        self.beta_start = 0.0015
        self.beta_end = 0.0195
        self.betas= torch.linspace(self.beta_start, self.beta_end, 1000)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
    def addNoise(self, x, t):
#         print ("add noise call Time: {}".format(datetime.timedelta(seconds=time.time() - start_noise)))
#         start = time.time()
        noise=torch.randn_like(x).to(device)
#         print ("sampling Time: {}".format(datetime.timedelta(seconds=time.time() - start)))
#         start = time.time()
        out = x*self.sqrt_alphas_cumprod[t] + noise*self.sqrt_one_minus_alphas_cumprod[t]
#         print ("calc Time: {}".format(datetime.timedelta(seconds=time.time() - start)))
        return out, noise
    
    def forward(self, images, t, t_tensor):
#         print('images shape:', images.shape)
#         start_slot = time.time()
        recon_combined, recons, masks, _, slots = self.slot_attn(images)
#         print ("Slot Time: {}".format(datetime.timedelta(seconds=time.time() - start_slot)))
#         start_vae = time.time()
        z = self.vae.encode(images)
#         print ("VAE Time: {}".format(datetime.timedelta(seconds=time.time() - start_vae)))
#         t_item = t[0]
#         start_noise = time.time()
        z, noise = self.addNoise(z, t)
#         print ("noise Time: {}".format(datetime.timedelta(seconds=time.time() - start_noise)))
#         z = z.to(device)
#         print('z shape:', z.shape)
        noise = noise.to(device)
#         start_noise = time.time()
        time_emb = self.time_embedding(t_tensor)
#         print ("time_emb Time: {}".format(datetime.timedelta(seconds=time.time() - start_noise)))
#         unet_start = time.time()
        z = self.unet(z, time_emb, slots)
#         print ("Unet Time: {}".format(datetime.timedelta(seconds=time.time() - unet_start)))
        
        return z, noise

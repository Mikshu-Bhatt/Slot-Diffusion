# Slot-Diffusion

Vae Checkpoint - https://drive.google.com/file/d/1d57Hm2mx_9GBGBMWvl38I3Hv2nPVYuCn/view?usp=sharing

We’ll implement a conditional diffusion model to decode the image conditioned on the slots.

Model specification

• Slot Encoder: Resnet Architecture from Assignment 1 is used as encoder
and the output of the resnet is feature map of size 64x64x64.

• Positional embeddings: Learnable grid of size equals to feature map
(64x64x64) is added to the feature map pointwise.

• Slot attention: Features + embeddings goes as an input to slot attention.
Slots are learnable parameters of dimension 64. Each slot itteratively learn
the represantion of itself via attending upont eh other slots.

• vae: The Image will go into the vae encoder and will be converted to
3x32x32 size latent representation. This representation will be added with
noise sampled from normal distribution and along with the slots from the
slot attention will go as an input to the Unet Diffusion model.

• Unet: Unet consist of down sampling,up sampling, residual block and
transformer block. Transformer block will take in as an input and other
blocks will take time embeddings as inputs. The output of this will be
3x32x32 sized image. The loss will be calculated between the noise we
sampled and the output imag from unet.

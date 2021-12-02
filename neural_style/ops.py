import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math


__all__ = ['PixelNorm', 'ConstantInput', 'EqualConv2d','Instancenorm']

#stau
class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)

    #stau
class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()
        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, batch):
        out = self.input.repeat(batch, 1, 1, 1)

        if hasattr(self, 'first_k_oup') and self.first_k_oup is not None:  # support dynamic channel
            assert self.first_k_oup <= out.shape[1]
            return out[:, :self.first_k_oup]
        else:
            return out

#stau
class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()  # random noise

        return image + self.weight * noise

#stau
def  make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()
    k = torch.flip(k, [0, 1])  # move from runtime to here
    return k


class EqualConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_channel, in_channel, kernel_size, kernel_size))
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))
        else:
            self.bias = None

    def forward(self, x):
        in_channel = x.shape[1]
        #print('conv',x.shape[1])
        weight = self.weight
        bbias = self.bias

        if hasattr(self, 'first_k_oup') and self.first_k_oup is not None:
            weight = weight[:self.first_k_oup]
            if self.bias!= None:
                bbias = bbias[:self.first_k_oup]


        weight = weight[:, :in_channel].contiguous()  # index sub channels for inference

        out = F.conv2d(
            x,
            weight * self.scale,
            bias=bbias,
            stride=self.stride,
            padding=self.padding,
        )
        
        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )


class Instancenorm(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.channel = channel
        self.gamma = nn.Parameter(torch.ones(1,channel,1,1))
        self.beta = nn.Parameter(torch.zeros(1,channel,1,1))



    def forward(self, x):
        results = 0.
        eps = 1e-5
        gamma = self.gamma
        beta = self.beta
        if hasattr(self, 'first_k_oup') and self.first_k_oup is not None:
            gamma = self.gamma[:,:self.first_k_oup].contiguous()
            beta = self.beta[:,:self.first_k_oup].contiguous()
        else:
            gamma = gamma[:,:self.channel].contiguous()
            beta = beta[:,:self.channel].contiguous()
        

        x_mean = torch.mean(x, axis=(2, 3), keepdims=True)
        x_var = torch.var(x, axis=(2, 3), unbiased=False, keepdims=True)# not bayles var
        x_normalized = (x - x_mean) / torch.sqrt(x_var + eps)
        results = gamma * x_normalized + beta
        return results


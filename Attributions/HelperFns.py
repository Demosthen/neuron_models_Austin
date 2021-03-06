
from __future__ import print_function, division
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torchvision import datasets, models
import matplotlib.pyplot as plt
import time
import os
import copy
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image, ImageFilter  

"""Imshow for normalized Image tensor with color channels as first dimension"""
def imshow(inp, title=None, 
           mean = np.array([0.485, 0.456, 0.406]),
           std = np.array([0.229, 0.224, 0.225])):
   
    inp = inp.numpy().transpose((1, 2, 0))
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

"""denormalize image tensor with set mean and std
    means and stds are reduced to color channels, 
    so means and stds should be 1D tensors of length 3
"""
def denormalize(tensor, mean, std):
    inp = torch.transpose(tensor, 0, 2)
    inp = torch.transpose(inp, 0, 1)
    denorm = std * inp + mean
    denorm = torch.transpose(denorm, 0, 2)
    denorm = torch.transpose(denorm, 1, 2)
    return denorm

"""inverse operation of denormalize
    means and stds are reduced to color channels, 
    so means and stds should be 1D tensors of length 3
"""
def renormalize(tensor, mean, std):
    inp = torch.transpose(tensor, 0, 2)
    inp = torch.transpose(inp, 0, 1)
    renorm =  (inp - mean) / std
    renorm = torch.transpose(renorm, 0, 2)
    renorm = torch.transpose(renorm, 1, 2)
    return renorm
    

"""normalize a tensor in place (means and stds)"""
def normalize(tensor, mean = None, std = None):
    if not std:
        std = torch.std(tensor)
    if not mean:    
        mean = torch.mean(tensor)
    tensor -= mean
    tensor /= std
    
"""Returns a tensor with upper (higher valued) half scaled from 0 to 1 
with the lower half <= 0"""
def scale(tensor):
    std = torch.std(tensor)
    mean = torch.mean(tensor)
    torch.clamp(tensor, mean - 2 * std, mean + 2 * std, out = tensor)
    maxVal = torch.max(tensor)
    minVal = torch.min(tensor)
    medVal = torch.median(tensor)
    numRange = maxVal - medVal
    tensor -= medVal
    tensor /= numRange
    
"""Returns a tensor with all values scaled linearly
   e.g. if top_percent = 20 and low_percent = 25, the value at the 25th percentile 
   in the tensor is subtracted from all values and all values are divided by the value at the 80th percentile.
"""
def simple_scale(tensor, top_percent = 1, low_percent = 1):
    
    minVal = get_top_percent(tensor, 100 - low_percent)
    tensor -= minVal
    maxVal = get_top_percent(tensor, top_percent)
    tensor /= maxVal

def get_top_percent(tensor, percent):
    idx = int(tensor.nelement() * percent / 100)
    flat = tensor.flatten()
    return torch.topk(flat, idx)[0][-1]

"""Overlay attributions over input image"""
def att_imshow(inp, att, title=None, 
               mean = np.array([0.485, 0.456, 0.406]),
               std = np.array([0.229, 0.224, 0.225])):
   
    inp = inp.numpy().transpose((1, 2, 0))
    inp = std * inp + mean
    inp = inp.transpose((2, 0, 1))
    inp *= att.numpy()
    inp = np.clip(inp, 0, 1)
    inp = inp.transpose((1, 2, 0))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    
"""Display attributions in grayscale"""
def heatmap_imshow(att, title = None):
    inp = np.clip(att, 0, 1)
    inp = att
    plt.imshow(inp.numpy(), cmap = "gray_r")
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

"""Returns a blurred version of the image with values from 0 to 1"""
def gaussBlur(image):
    toPil = transforms.ToPILImage()
    toTensor = transforms.ToTensor()
    inPil = toPil(image)
    plt.imshow(np.asarray(inPil))
    plt.pause(0.001)
    filtered = inPil.filter(ImageFilter.GaussianBlur(radius = 10)) 
    plt.imshow(np.asarray(filtered))
    plt.pause(0.001)
    return toTensor(filtered)

"""Returns a new tensor with a gaussian filter applied to it with same dimensions as input"""
def blurred_like(tensor, means, stds):
    denorm = denormalize(tensor, means, stds)
    base = gaussBlur(denorm)
    base = renormalize(base, means, stds)
    return base

"""Returns a new tensor filled with Gaussian noise with same dimensions as input"""
def noise_like(tensor, means, stds):
    ret = torch.randn_like(tensor) 
    return renormalize(ret, means, stds)

"""Returns a new all-black tensor with same dimensions as input"""
def black_like(tensor, means, stds):
    ret = torch.zeros_like(tensor)
    return renormalize(ret, means, stds)

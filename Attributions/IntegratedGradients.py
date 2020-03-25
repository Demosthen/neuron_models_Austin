import torch
from .HelperFns import *
import torch.nn.functional as F
"""
Computes an attribution vector by the integrated gradient method described in 
Axiomatic Attribution for Deep Networks for a single image.
Assumes input is a single color image with 3 dimensions (channel, width, height)
Can set debug_display parameter to see how much of the difference in score between
the input image and baseline is accounted for by the attributions to determine
if the number of steps is adequate.
Returns an attribution tensor of same dimensions as input image
"""
def attribute(model, image, base, steps = 100, target = None, 
              output_activation = lambda x: x, debug_display = False): 
    for param in model.parameters():
        param.requires_grad = True
    model.eval()
    baseShape = base.shape
    imageShape = image.shape
    baseBatch = base.view(1, baseShape[0], baseShape[1], baseShape[2])
    imageBatch = image.view(1, imageShape[0], imageShape[1], imageShape[2])
    baseScore = model(baseBatch)
    inScore = model(imageBatch)
    if target == None:
        target = torch.argmax(inScore[0])
    gradSum = torch.zeros([1, imageShape[0], imageShape[1], imageShape[2]], device=image.device)
    diff = imageBatch - baseBatch
    
    for i in range(steps):
        new = baseBatch + diff * (float(i) / steps)
        new.requires_grad = True
        newShape = new.shape
        model.zero_grad()
        out = model(new)
        out = output_activation(out)
        out[0][target].backward()
        gradTensor = new.grad
        gradSum += gradTensor
            
    gradSum /= steps
    result = gradSum * diff
    if debug_display:
        baseSum = output_activation(baseScore)[0][target]
        diffSum = output_activation(inScore)[0][target] - baseSum
        resultSum = torch.sum(result)
        print("gradients vs score diff: ", abs((resultSum - diffSum)/diffSum).item() * 100,  "%")
        print("gradients: ", resultSum.item(), " score diff: ", diffSum.item())
    return result, torch.argmax(out[0])

"""Computes attributions using random noise as baseline.
    Averages over num_iterations different noise baselines"""
def random_baseline_attribution(model, img, num_iterations, means, stds, output_activation = lambda x: x):
    generate_noise = lambda : noise_like(img.cuda(), torch.tensor(means).cuda(), torch.tensor(stds).cuda())
    atts = 0
    for i in range(num_iterations):
        att, _ = attribute(model, img.cuda(), generate_noise(), 50,
                           output_activation = output_activation)
        atts = att + atts
    atts /= num_iterations
    return atts
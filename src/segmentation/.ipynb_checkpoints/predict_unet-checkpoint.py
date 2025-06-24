import torch
from PIL import Image
import numpy as np
import os 
from torchvision import transforms
import cv2
import random
import matplotlib.pyplot as plt


def post_process(predType, num_classes, convert_numpy):
    if num_classes > 1:
        predType = torch.nn.functional.softmax(predType, dim=1)
        predType = torch.argmax(predType, dim=1).cpu().numpy().astype(np.uint8)
    else:
        predType = torch.sigmoid(predType.squeeze(1)).cpu()
        predType = (predType>0.5)   
        if convert_numpy:
            predType = predType.detach().numpy().astype('uint')
    return predType
    
def predict(img_path, model_path, convert_numpy = False):
    image = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
    img_size = image.shape
    image = transforms.Resize(size=(128, 128))(image)
    model = torch.load(model_path).to('cuda')
    model.eval()
    image = image.unsqueeze(0).to('cuda')
    out = model(image)
    out = transforms.Resize(size = (img_size[1], img_size[2]), interpolation=Image.NEAREST)(out)
    out = post_process(out, 3, convert_numpy)
    return out
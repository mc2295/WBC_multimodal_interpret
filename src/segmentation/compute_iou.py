from config import images_test_path
import torch
from PIL import Image
import numpy as np
import os 
from torchvision import transforms
import cv2
from segmentation.predict_unet import predict as predict_unet
from segmentation.predict_sam import predict as predict_sam
import matplotlib.pyplot as plt

def compute_iou(images_test_path, model_path, save_overlay = False):
    with torch.no_grad():
        count = 0
        iou = 0
    
        list_file = os.listdir(images_test_path + 'images/')
        random.shuffle(list_file)
        for img_path in list_file:
            if img_path == '.ipynb_checkpoints':
                continue 
            
            true_mask = cv2.imread(images_test_path + 'masks/' + img_path)[:,:,0]
            img_path_tot = images_test_path + 'images/' + img_path
            
            if source in {'barcelona', 'rabin'}:
                out = predict_unet(image_path_tot, model_path)
            if source in {'tianjin'}: 
                bbox_name = images_path_test + 'bounding_boxes/' + img_path[:-4] + '.txt'
                out, label = predict_sam(img_path_tot,bbox_name)
    
            if save_overlay:
                fig, ax = plt.subplots()
                image_cv2 = cv2.imread(images_test_path + 'images/' + img_path)
                image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
                ax.imshow(image_cv2)
                create_overlay(out, ax, title = 'out')
                ax.axis('off')
            
                fig, ax = plt.subplots()
                ax.imshow(image_cv2)
                create_overlay(true_mask, ax, title = 'true')
                ax.axis('off')
            
            intersection = np.logical_and(true_mask, out)
            union = np.logical_or(true_mask, out)
    
            # Calculate IoU for each pair
            iou_per_sample = np.sum(intersection, axis=(1, 2)) / np.sum(union, axis=(1, 2)) # ii is axis = (0,1) for tianjing
            # Average over the batch
            iou += np.mean(iou_per_sample)
        iou_avg = iou / len(os.listdir(images_test_path + 'images/'))
        print(iou_avg)
        return iou_avg
from data.features.color_features import compute_color_features
from data.features.geometrical_features import compute_geometrical_features
from data.features.texture_features import compute_texture_features
import numpy as np
import cv2
import functools as ft
import os
import matplotlib.pyplot as plt
# from extra_features import path_save_cropped
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


def create_dic_mask(mask):
    mask_nucleus = mask.copy()
    mask_cyto = mask.copy()
    mask_cytoonly = mask.copy()
    mask_nucleus[mask_nucleus!=2] = 0
    mask_nucleus[mask_nucleus==2] = 1
    mask_cyto[mask_cyto !=0] = 1
    mask_cytoonly[mask_cytoonly!=1] = 0
    dic_mask = {'cyto':mask_cyto, 'nucleus': mask_nucleus, 'cytoonly': mask_cytoonly}
    return dic_mask


def extract_features(cropped, mask_type, filename, label, source): 
    col_names = ["cell", "label", "dataset"]
    
    row = [filename, label, source]
    geometrical, geo_col_names = compute_geometrical_features(
        cropped, mask_type, filename
    )

    row += geometrical
    col_names += geo_col_names

    color, color_col_names = compute_color_features( cropped, mask_type)
    row += color
    col_names += color_col_names
    texture, text_col_names = compute_texture_features(cropped, mask_type)
    row += texture
    col_names += text_col_names
    return row, col_names

def compute_features(image, mask, filename, label, source, show_images = False, save_cropped = False):
    if source in ['barcelona', 'rabin']:
        image = (image.squeeze(0).permute(1,2,0).numpy()*255).astype(np.uint8)
    dic_mask = create_dic_mask(mask)
    dic_cropped = {k:cv2.bitwise_and(image, image, mask=cv2.convertScaleAbs(v)) for k,v in dic_mask.items()}
    if show_images: 
        fig, ax = plt.subplots(1,5, figsize = (15,3))
        ax[0].imshow(image)
        ax[0].set_title('image')
        ax[1].imshow(mask)
        ax[1].set_title('mask')
        for i, k in enumerate(dic_cropped.keys()):
            ax[i+2].imshow(dic_cropped[k])
            ax[i+2].set_title(k)
        
    merge = {}
        
    for mask_type, cropped in dic_cropped.items():
        if mask_type == 'nucleus':
            # if len(np.where(cropped != 0)[0]) > 1000:
            if np.unique(cropped!=0, return_counts = True)[1][1] > 150:

                row, col = extract_features(cropped, mask_type, filename, label, source)
                merge[mask_type] = pd.DataFrame([row], columns=col)                
        else:
            if len(np.where(cropped != 0)[0]) > 300:
                
                row, col = extract_features(cropped, mask_type, filename, label, source)
                merge[mask_type] = pd.DataFrame([row], columns=col)
                if save_cropped and mask_type == 'cyto':
                    plt.imsave( path_save_cropped + filename.split('/')[-1], cropped)
    return merge

def save_features(image, masks, filename, label, source, save_file, show_images = False, save_cropped = False):
    for i, mask in enumerate(masks):
        # pixels_x, pixels_y = np.where(mask != 0)
        # if pixels_x[0] >3 and pixels_y[0] > 3 and pixels_x[-1] < 125 and pixels_y[-1] <125:
        merge = compute_features(image, mask, filename.split('.png')[0] + '_' + str(i) + '.png', label[i], source, show_images = show_images, save_cropped = save_cropped)
        if merge == {}:
            continue
        dfs = [merge[k] for k in merge.keys()]
        if len(dfs) > 0:
            features_df = ft.reduce(lambda left, right: pd.merge(left, right, on='cell'), dfs)
        if os.path.exists(save_file):
            df = pd.read_csv(save_file)
            pd.concat([df, features_df]).to_csv(os.path.join(save_file), index = False)
        else: 
            features_df.to_csv(save_file, index=False)
                
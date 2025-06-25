import pickle
import shap
import pandas as pd
import torch
import matplotlib.pyplot as plt
import copy
import numpy as np
import os
import matplotlib as mpl
import matplotlib.font_manager as fm




def clean_feature_names(names):
    new_names = []
    for k in names:
        new_name = k
        new_name = new_name.replace('cyto_', 'Cell_')
        new_name = new_name.replace('nucleus_', 'Nucleus_')
        new_name = new_name.replace('cytoonly_', 'Cyto_')
        new_name = new_name.replace('lbp_', 'LBP_')
        new_name = new_name.replace('glcm_', 'GLCM_')
        new_name = new_name.replace('channel_', '')
        new_name = new_name.replace('_C', '_Cyan')
        new_name = new_name.replace('_M', '_Magenta')
        new_name = new_name.replace('_Y', '_Yellow')
        new_name = new_name.replace('_R', '_Red')
        new_name = new_name.replace('_G', '_Green')
        new_name = new_name.replace('_B', '_Blue')
        new_name = new_name.replace('_std', '_Std')
        new_name = new_name.replace('_H', '_Hue')
        new_name = new_name.replace('_S', '_Saturation')
        new_name = new_name.replace('_V', '_Value')
        new_name = new_name.replace('_Saturationtd', '_std')
        new_name = new_name.replace('_HueSV', '_HSV')
        new_name = new_name.replace('_RedGB', '_RGB')
        new_name = new_name.replace('_CyanMYK', '_CMYK')
        new_name = new_name.replace('_GreenLCM', '_GLCM')
        new_names.append(new_name)
    return new_names

def load_dataset_features(source):
    df = pd.read_csv(f'features_{source}.csv')
    columns_tabular = [i for i in df.columns if i not in ['cell', 'label', 'dataset', 'Unnamed: 0', 'is_valid', 'index', 'Unnamed: 0.1', 'Unnamed: 0.1.1']]
    columns_tabular_reduced = [
        (idx, name) for idx, name in enumerate(columns_tabular)
        if ((name.split('_')[1] in {'area,', 'perimeter', 'eccentricity', 'roundness', 'convex', 'eccentricity', 'aspect', 'circularity'}) or
            (name.split('_')[1] == 'glcm') or
            (name.split('_')[1] == 'lbp') or
            (name.split('_')[1] == 'channel' and name.split('_')[2] in {'mean', 'std', 'skewness', 'entropy', 'kurtosis'} and name.split('_')[3] in {'R', 'G', 'B', 'C', 'M', 'Y'}))]
    return columns_tabular_reduced, torch.load(f'X_features_{source}').detach().numpy()
    #return [(idx, name) for idx, name in enumerate(columns_tabular)], torch.load(f'X_features_{source}').detach().numpy()

def get_region_features(columns_tabular_reduced):
    return {
        'cell': [(idx, name) for idx, name in columns_tabular_reduced if 'cyto_' in name],
        'nucleus': [(idx, name) for idx, name in columns_tabular_reduced if 'nucleus_' in name],
        'cyto': [(idx, name) for idx, name in columns_tabular_reduced if 'cytoonly_' in name]
    }

def load_shap_values(source):
    with open(f'shap_{source}', 'rb') as file:
        return pickle.load(file)

def plot_combined_shap(region, region_features_all, X_all, shap_all, dataset_names):
    font_path = 'times_new_roman.ttf'  
    fm.fontManager.addfont(font_path)  
    plt.rcParams['font.family'] = 'Times New Roman Cyr'  # Replace 'YourCustomFont' with the actual font name
    mpl.rcParams.update({'font.size': 20})
    mpl.rcParams.update({
        'font.size': 20,
        'axes.labelsize': 20,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 20})

    for class_idx, class_name in enumerate(['Basophil', 'Eosinophil', 'Lymphocyte', 'Monocyte', 'Neutrophil']):
        print(f"Generating plot for class: {class_name}, region: {region}")
        
        # Use features from the first dataset for consistency
        features = region_features_all[dataset_names[0]][region]
        idx = [256 + i for i, _ in features]
        names = [name for _, name in features]
        new_names = clean_feature_names(names)
        # Combine SHAP values and feature inputs across datasets
        shap_vals_combined = np.vstack([shap_all[dataset][class_idx][:, idx] for dataset in dataset_names])
        X_combined = np.vstack([X_all[dataset][:, idx] for dataset in dataset_names])
        print(shap_vals_combined.shape, X_combined.shape)
        # Plot combined SHAP summary
        fig = plt.figure(figsize=(7, 5))
        shap.summary_plot(
            shap_vals_combined,
            X_combined,
            feature_names=new_names,
            show=False
        )
        plt.xticks(fontsize=20)                      # X-axis ticks font size
        plt.yticks(fontsize=22)  
        plt.xlabel(plt.gca().get_xlabel(), fontsize=20)
        plt.ylabel(plt.gca().get_ylabel(), fontsize=20)
        cbar = plt.gcf().axes[-1]  # The colorbar is usually the last axis
        cbar.tick_params(labelsize=20)
        cbar.set_title(cbar.get_title(), fontsize=20)
        dic_region = {'nucleus': 'Nucleus', 'cyto': 'Cytoplasm', 'cell': 'Full Cell'}
        plt.title(f'{class_name} - {dic_region[region]}')

        # Remove the color bar if it exists
        ax = plt.gca()
        if ax and ax.collections:
            # Remove the last colorbar (SHAP adds it at the end)
            try:
                fig = plt.gcf()
                for obj in fig.axes:
                    if obj != ax and 'Colorbar' in str(type(obj)):
                        fig.delaxes(obj)
            except Exception as e:
                print(f"Warning: couldn't remove colorbar - {e}")

        # Save and close
        os.makedirs('figures/shap/shap_combined', exist_ok=True)
        pdf_filename = f'figures/shap/shap_combined/{class_name}_{region}.pdf'
        plt.savefig(pdf_filename, bbox_inches='tight', format='pdf')
        plt.close()




# ---------------- MAIN CODE ---------------- #


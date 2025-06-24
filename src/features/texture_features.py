from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
import numpy as np
import cv2

def compute_gclm_features(img):
    glcm = graycomatrix(img, distances = [5], angles =[0] )
    properties = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']

    glcm_props = [propery for name in properties for propery in graycoprops(glcm, name)[0]]
    return glcm_props

def compute_lbp_features(img):
    radius = 3
    n_points = 8 * radius

    lbp = local_binary_pattern(img, n_points, radius, 'uniform')

    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)  # Normalize the histogram

    texture_energy = (hist[1:-1] ** 2).sum()
    texture_contrast = ((np.arange(0, n_points + 2) - hist.mean()) ** 2).sum()
    texture_entropy = -np.sum(hist[1:-1] * np.log2(hist[1:-1] + 1e-6))
    local_mean = lbp.mean()
    local_variance = lbp.var()
    return texture_energy,texture_contrast, texture_entropy, local_mean, local_variance

def compute_texture_features(cropped, mask_type):
    grey = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
    glcm_props = compute_gclm_features(grey)
    glcm_dissimilarity, glcm_correlation, glcm_homogeneity, glcm_contrast, glcm_ASM, glcm_energy = glcm_props[0], glcm_props[1], glcm_props[2], glcm_props[3], glcm_props[4], glcm_props[5]
    lbp_energy,lbp_contrast, lbp_entropy, lbp_mean, lbp_variance = compute_lbp_features(grey)
    column_names = [
        f"{mask_type}_glcm_dissimilarity",
        f"{mask_type}_glcm_correlation",
        f"{mask_type}_glcm_homogeneity",
        f"{mask_type}_glcm_contrast",
        f"{mask_type}_glcm_ASM",
        f"{mask_type}_glcm_energy",
        f"{mask_type}_lbp_energy",
        f"{mask_type}_lbp_contrast",
        f"{mask_type}_lbp_entropy",
        f"{mask_type}_lbp_mean",
        f"{mask_type}_lbp_variance"
    ]
    return (
    glcm_dissimilarity,
    glcm_correlation,
    glcm_homogeneity,
    glcm_contrast,
    glcm_ASM,
    glcm_energy,
    lbp_energy,
    lbp_contrast,
    lbp_entropy,
    lbp_mean,
    lbp_variance), column_names

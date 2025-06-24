import numpy as np
import cv2
from scipy import stats
import skimage.measure
import pyhdust.images as phim


def compute_mean_and_std(img):
    # gives the same result when filtering zeros
    channel_mean, channel_std = cv2.meanStdDev(img)
    channel_mean = np.hstack(channel_mean)
    channel_std = np.hstack(channel_std)
    img_mean = np.mean(channel_mean)
    img_std = np.mean(channel_std)
    channel_mean = ",".join(str(x) for x in channel_mean)
    channel_std = ",".join(str(x) for x  in channel_std)
    return channel_mean, channel_std, img_mean, img_std

def compute_skewness_and_kurtosis_and_entropy(img):
    non_zero_img = np.where(img !=0, img, 1)
    skewness = [stats.skew(channel.ravel()) for channel in cv2.split(img)]
    kurtosis = [stats.kurtosis(channel.ravel()) for channel in cv2.split(img)]
    entropy =  [stats.entropy(channel.ravel()) for channel in cv2.split(non_zero_img)]
    skewness = ",".join(str(x) for x in skewness)
    kurtosis = ",".join(str(x) for x in kurtosis)
    entropy = ",".join(str(x) for x in entropy)
    # print(stats.skew(img[0],axis=None))
    # print(stats.skew(img[1],axis=None))
    # print(stats.skew(img[2],axis=None))
    return skewness, kurtosis, entropy

def compute_uniformity_and_shannon_entropy(img):
    # same result when filtering zeros
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    nbins = 128
    hist, _ = np.histogram(gray.ravel(), bins=nbins, range=(0, nbins))
    prob_dist = hist / hist.sum()
    uniformity = np.sum(np.square(prob_dist))
    non_zero_prob = prob_dist[prob_dist != 0]
    entropy = stats.entropy(non_zero_prob, base=2)

    return uniformity, entropy

def compute_color_features(cropped, mask_type):

    img_RGB = cropped.copy()
    channel_mean_RGB, channel_std_RGB, img_mean_RGB, img_std_RGB = compute_mean_and_std(img_RGB)
    channel_skewness_RGB, channel_kurtosis_RGB, channel_entropy_RGB = compute_skewness_and_kurtosis_and_entropy(img_RGB) 
    uniformity_RGB, shannon_entropy_RGB = compute_uniformity_and_shannon_entropy(img_RGB) 

    img_CMYK = phim.rgb2cmyk(np.asanyarray(img_RGB))
    channel_mean_CMYK, channel_std_CMYK, img_mean_CMYK, img_std_CMYK = compute_mean_and_std(img_CMYK)
    channel_skewness_CMYK, channel_kurtosis_CMYK, channel_entropy_CMYK = compute_skewness_and_kurtosis_and_entropy(img_CMYK) 
    uniformity_CMYK, shannon_entropy_CMYK = compute_uniformity_and_shannon_entropy(img_CMYK) 

    img_HSV = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2HSV)
    channel_mean_HSV, channel_std_HSV, img_mean_HSV, img_std_HSV = compute_mean_and_std(img_HSV)
    channel_skewness_HSV, channel_kurtosis_HSV, channel_entropy_HSV = compute_skewness_and_kurtosis_and_entropy(img_HSV) 
    uniformity_HSV, shannon_entropy_HSV = compute_uniformity_and_shannon_entropy(img_HSV) 

    column_names = [
        f"{mask_type}_channel_mean_RGB",
        f"{mask_type}_channel_std_RGB",
        f"{mask_type}_img_mean_RGB",
        f"{mask_type}_img_std_RGB",
        f"{mask_type}_channel_skewness_RGB",
        f"{mask_type}_channel_kurtosis_RGB",
        f"{mask_type}_channel_entropy_RGB",
        f"{mask_type}_uniformity_RGB",
        f"{mask_type}_shannon_entropy_RGB",
        f"{mask_type}_channel_mean_CMYK",
        f"{mask_type}_channel_std_CMYK",
        f"{mask_type}_img_mean_CMYK",
        f"{mask_type}_img_std_CMYK",
        f"{mask_type}_channel_skewness_CMYK",
        f"{mask_type}_channel_kurtosis_CMYK",
        f"{mask_type}_channel_entropy_CMYK",
        f"{mask_type}_uniformity_CMYK",
        f"{mask_type}_shannon_entropy_CMYK",
        f"{mask_type}_channel_mean_HSV",
        f"{mask_type}_channel_std_HSV",
        f"{mask_type}_img_mean_HSV",
        f"{mask_type}_img_std_HSV",
        f"{mask_type}_channel_skewness_HSV",
        f"{mask_type}_channel_kurtosis_HSV",
        f"{mask_type}_channel_entropy_HSV",
        f"{mask_type}_uniformity_HSV",
        f"{mask_type}_shannon_entropy_HSV"
    ]
    return (
    channel_mean_RGB,
    channel_std_RGB,
    img_mean_RGB,
    img_std_RGB,
    channel_skewness_RGB,
    channel_kurtosis_RGB,
    channel_entropy_RGB,
    uniformity_RGB,
    shannon_entropy_RGB,
    channel_mean_CMYK,
    channel_std_CMYK,
    img_mean_CMYK,
    img_std_CMYK,
    channel_skewness_CMYK,
    channel_kurtosis_CMYK,
    channel_entropy_CMYK,
    uniformity_CMYK,
    shannon_entropy_CMYK,
    channel_mean_HSV,
    channel_std_HSV,
    img_mean_HSV,
    img_std_HSV,
    channel_skewness_HSV,
    channel_kurtosis_HSV,
    channel_entropy_HSV,
    uniformity_HSV,
    shannon_entropy_HSV,               
    ), column_names


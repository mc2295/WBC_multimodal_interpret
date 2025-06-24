import cv2
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import os
# from extract_features import path_save_cropped

def compute_main_geo_features(cropped, filename, save_geometry = False):
    gray = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
    # give the pixels a value according to the selected threshold (e.g. 1 in this case)
    _, thresh = cv2.threshold(gray, 1, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    temp_points = np.argwhere(thresh == 255)
    Ncl_points = np.zeros_like(temp_points)
    Ncl_points[:, 0] = temp_points[:, 1]
    Ncl_points[:, 1] = temp_points[:, 0]
    area = np.sum(thresh)
    perimeter = sum([cv2.arcLength(contour, True) for contour in contours])
    hull = ConvexHull(Ncl_points)
    hull_area = hull.volume

    Verc = hull.vertices
    Corners = []
    for idx in range(len(Verc)):
        tempcol = Ncl_points[Verc[idx], 0]
        temprow = Ncl_points[Verc[idx], 1]
        Corners.append([tempcol, temprow])
    Corners = np.array(Corners)
    Corners = np.reshape(Corners, newshape=(Corners.shape[0], 1, 2))

    ellipse = cv2.fitEllipse(np.array(Corners))

    eccentricity = np.sqrt(
        np.square(ellipse[1][1] / 2) - np.square(ellipse[1][0] / 2)
    ) / (ellipse[1][1] / 2)
    # draw the contours, the convex hull and the ellipse on the image
    if save_geometry:
        cv2.drawContours(cropped, contours, -1, (0, 255, 255), 1)
        cv2.drawContours(cropped, [Corners], 0, (0, 255, 0), 1)
        cv2.ellipse(cropped, ellipse, (255, 0, 0), 1)
        # estimate the convex hull area
        plt.axis("off")
        plt.imshow(cropped)
        plt.savefig(
            os.path.join(path_save_cropped, filename.stem + "_geometrical.png"), 
            bbox_inches="tight",
            pad_inches=0,
        )
        # plt.show()
    return area, perimeter, hull_area, eccentricity, (ellipse[1][1], ellipse[1][0])

def compute_circularity(area, perimeter):
    return perimeter**2 / (4 * np.pi * area)


def compute_solidity(area, hull_area):
    return area / hull_area


def compute_roundness(area, major_axis_ellipse):
    return 4 * area / (np.pi * np.square(major_axis_ellipse))


def compute_aspect_ratio(ellipse_axes):
    return ellipse_axes[0] / ellipse_axes[1]

def compute_geometrical_features(cropped, mask_type, filename):
    column_names = [
        f"{mask_type}_area",
        f"{mask_type}_perimeter",
        f"{mask_type}_convex_hull_area",
        f"{mask_type}_eccentricity",
        f"{mask_type}_aspect_ratio",
        f"{mask_type}_roundness",
        f"{mask_type}_circularity",
        f"{mask_type}_solidity",
    ]
    area,perimeter,hull_area,eccentricity, ellipse_axes = compute_main_geo_features(
        cropped, filename)
    circularity = compute_circularity(area, perimeter)
    solidity = compute_solidity(area, hull_area)
    roundness = compute_roundness(area, ellipse_axes[0])
    aspect_ratio = compute_aspect_ratio(ellipse_axes)

    return (
        area,
        perimeter,
        hull_area,
        eccentricity,
        aspect_ratio,
        roundness,
        circularity,
        solidity,
    ), column_names



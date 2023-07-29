import glob
import os

import cv2
import numpy as np
import torch
from scipy.ndimage import label
from skimage.morphology import skeletonize
from tqdm import tqdm


def remove_dots(img, width):
    """
    Removes small dots (contours) from a binary image.

    Args:
        img (np.ndarray): The input binary image.
        width (float): The expected width of the streets or roads.

    Returns:
        np.ndarray: The binary image with small dots removed.
    """
    # Find contours of the connected components
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours (dots) based on their compactness
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        # Calculate compactness, which indicates how many pixels long the street is    
        compactness = area/(np.pi*width**2)
        
        # Check if the contour is a small dot
        if compactness < 1:
            filtered_contours.append(contour)

    # Draw the filtered contours on the filtered_image to remove small dots
    filtered_image = np.copy(img)
    cv2.drawContours(filtered_image, filtered_contours, -1, 0, -1)

    # Return the filtered_image with small dots removed
    return filtered_image


def compute_width_label(img, max_width=70):
    """
    Computes the "rich label" representing the width of different regions in the binary image.

    Args:
        img (np.ndarray): Input binary image.
        max_width (int): Maximum expected width of regions to be computed.

    Returns:
        np.ndarray: The rich label image with computed width values for different regions.
    """
    rich_label = np.zeros_like(img)

    # Loop over width values to process regions of different sizes
    for i in range(2, max_width, 1):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (i, i))
        # remove thin lines
        bin = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
        bin = cv2.morphologyEx(bin, cv2.MORPH_DILATE, kernel)
        # remove dots
        bin = remove_dots(bin, i)

        # comute removed parts
        diff = img - bin
        # get rid of border issues
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        clean = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel)

        # Assign the width value to regions not labeled yet
        this_size = np.bitwise_and(clean != 0, rich_label == 0)
        rich_label[this_size] = i

    # Process borders to group connected components and assign maximum width
    r = 5
    top = np.s_[:r]
    bot = np.s_[-r:]
    left = np.s_[:, :r]
    right = np.s_[:, -r:]
    for border in [top, bot, left, right]:
        labels, num_labels = label(rich_label[border])
        for i in range(num_labels):
            max_val = np.max(rich_label[border][i + 1 == labels])
            rich_label[border][i + 1 == labels] = max_val

    return rich_label


def compute_sdf_label(img):
    """
    Computes the rich label signed distance function (SDF) for the input binary image.

    Args:
        img (np.ndarray): Input binary image.

    Returns:
        np.ndarray: The SDF label representing the distance to the nearest boundary for each pixel.
    """
    # Obtain thin lines preserving the topology
    skeleton = skeletonize(img)
    skeleton = (1-skeleton).astype(np.uint8)

    # Compute the distance transform using the complemented skeleton
    dist = cv2.distanceTransform(skeleton, cv2.DIST_LABEL_PIXEL, 0)

    # Clip the computed distances to the range [0, 255]
    score = np.clip(dist, 0, 255)

    return score


def compute_dir_label(img):
    """
    Computes the rich label direction for the input binary image.

    Args:
        img (np.ndarray): Input binary image.

    Returns:
        np.ndarray: The directional label representing the gradient direction for each pixel.
    """
    # Compute the sdf label
    sdf = compute_sdf_label(img).astype(np.float64)

    # Compute the gradients in the x and y directions
    dx = cv2.Sobel(sdf, cv2.CV_64F, 1, 0, ksize=15)
    dy = cv2.Sobel(sdf, cv2.CV_64F, 0, 1, ksize=15)

    # Calculate the angle of the gradient vector (in radians) and map it to the range [0, 255]
    angle = np.arctan2(dy, dx) % np.pi
    angle = (angle/np.pi*255).astype(np.uint8)

    # Apply median blur with a kernel size of 3 to smooth the directional label
    angle = cv2.medianBlur(angle, ksize=3)

    return angle


def dir_from_sdf(sdf):
    """
    Computes the rich label direction from the signed distance function.
    The angle is between 0 and 1 which corresponds to a half circle

    Args:
        img (np.ndarray): Input sdf image.

    Returns:
        np.ndarray: The directional label representing the gradient direction for each pixel.
    """
    # Compute the sdf label
    sdf = sdf.squeeze().numpy().astype(np.float64)/256

    # Compute the gradients in the x and y directions
    dx = cv2.Sobel(sdf, cv2.CV_64F, 1, 0, ksize=15)
    dy = cv2.Sobel(sdf, cv2.CV_64F, 0, 1, ksize=15)

    # Calculate the angle of the gradient vector (in radians) and map it to the range [0, 255]
    angle = np.arctan2(dy, dx) % np.pi
    angle = (angle / np.pi * 256).astype(np.uint8)

    # Apply median blur with a kernel size of 3 to smooth the directional label
    angle = cv2.medianBlur(angle, ksize=3)

    return torch.tensor(angle/256.).unsqueeze(0)


if __name__ == '__main__':
    # Obtain paths of input groundtruth images
    paths = glob.glob('data/training/groundtruth/*.png')

    # Define storage folders for width, SDF, and directional labels
    store_folder_width = r"./data/training/groundtruth_width"
    store_folder_sdf = r"./data/training/groundtruth_sdf"
    store_folder_dir = r"./data/training/groundtruth_dir"

    # Create the output folders if they don't exist
    os.makedirs(store_folder_width, exist_ok=True)
    os.makedirs(store_folder_sdf, exist_ok=True)
    os.makedirs(store_folder_dir, exist_ok=True)
    
    # Process each input groundtruth image
    for p in tqdm(paths):
        # Read the image and convert it to binary using a threshold of 128
        img = cv2.imread(p, 0)
        img = cv2.inRange(img, 128, 255)

        # Compute and save the width label
        width_label = compute_width_label(img)
        cv2.imwrite(f"{store_folder_width}/{os.path.basename(p)}", width_label)

        # Compute and save the SDF label
        sdf_label = compute_sdf_label(img)
        cv2.imwrite(f"{store_folder_sdf}/{os.path.basename(p)}", sdf_label)

        # Compute and save the directional label
        dir_label = compute_dir_label(img)
        cv2.imwrite(f"{store_folder_dir}/{os.path.basename(p)}", dir_label)
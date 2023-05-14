import cv2
import glob
from scipy.ndimage import label
import numpy as np
import matplotlib.pyplot as plt



def remove_dots(img, width):
    # Find contours of the connected components
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Filter out small contours (dots)
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        # compactness says how many pixel long is the street
        compactness = area/(np.pi*width**2)
        # print(compactness)
        if compactness < 1:
            filtered_contours.append(contour)
    filtered_image = np.copy(img)
    cv2.drawContours(filtered_image, filtered_contours, -1, 0, -1)
    return filtered_image

def compute_rich_label(img, max_width=70):
    rich_label = np.zeros_like(img)
    for i in range(2, max_width, 1):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (i, i))
        # remove thin lines
        bin = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
        bin = cv2.morphologyEx(bin, cv2.MORPH_DILATE, kernel)
        #
        bin = remove_dots(bin, i)

        # comute removed parts
        diff = img - bin
        # get rid of border issues
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        clean = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel)

        this_size = np.bitwise_and(clean != 0, rich_label == 0)
        rich_label[this_size] = i

    # crop or flood fill on the boundary
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
    # rich_label = rich_label[max_width:-max_width, max_width:-max_width]
    # img = img[max_width:-max_width, max_width:-max_width]
    return rich_label

if __name__ == '__main__':
    import os
    from tqdm import tqdm
    paths = glob.glob('data/training/groundtruth/*.png')
    store_folder = r"./data/training/groundtruth_rich"


    os.makedirs(store_folder, exist_ok=True)
    max_width = 70
    for p in tqdm(paths):
        img = cv2.imread(p, 0)
        rich_label = compute_rich_label(img)
        cv2.imwrite(f"{store_folder}/{os.path.basename(p)}", rich_label)

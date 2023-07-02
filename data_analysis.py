import cv2
import glob
from scipy.ndimage import label
import numpy as np
import matplotlib.pyplot as plt




if __name__ == '__main__':
    import os
    from tqdm import tqdm
    paths = glob.glob('data/training/groundtruth/*.png')
    width = glob.glob('data/training/groundtruth_width/*')

    all = np.zeros((len(width),400,400))
    for i, p in enumerate(tqdm(width)):
        all[i]=cv2.imread(p, 0)
    plt.hist(all.ravel(),bins=30)
    plt.show()

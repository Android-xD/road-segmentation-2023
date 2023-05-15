import cv2
import matplotlib.pyplot as plt
import numpy as np


def show_img_mask(img,mask):
    figure = plt.figure(figsize=(8, 8))
    figure.add_subplot(1, 2, 1)
    plt.imshow(np.transpose(img.squeeze(),(1,2,0)))
    figure.add_subplot(1, 2, 2)
    plt.imshow(mask.squeeze())
    plt.show()

def show_img_mask_canny(img,mask):
    img = np.transpose(img.squeeze(),(1,2,0)).numpy()
    mask = mask.squeeze().numpy()
    img[cv2.Canny(mask, 20, 40) != 0] = (0, 0, 255)
    plt.imshow(img)
    plt.show()

def show_img_mask_alpha(img, mask, alpha):
    img = np.transpose(img.squeeze(),(1,2,0)).numpy()
    img = (img + 1) / 2
    mask = mask.squeeze().numpy()
    img[mask != 0] = (1-alpha)*img[mask != 0] + (0, 0, alpha)
    plt.imshow(img)
    plt.show()

def output_target_alpha(input, output,alpha):
    img = np.transpose(input.squeeze(), (1, 2, 0)).numpy()
    img = (img + 1) / 2
    mask = output.squeeze().numpy()
    img[mask > 0.5] = (1-alpha)*img[mask > 0.5] + (0, 0, alpha)
    plt.imshow(img)
    plt.show()

def output_target_heat(input, output, alpha,target = None):
    img = np.transpose(input.squeeze(), (1, 2, 0)).numpy()
    img = (img + 1) / 2

    if not target is None:
        target = target.squeeze().numpy()
        img[cv2.Canny(target.astype(np.uint8)*255, 20, 40) != 0] = (0, 0, 1)

    mask = output.squeeze().numpy()
    color_map = plt.get_cmap('jet')  # choose a colormap
    color_image = color_map(mask)
    img = (1 - alpha) * img + alpha * color_image[:, :, :3]
    plt.imshow(img)
    plt.show()




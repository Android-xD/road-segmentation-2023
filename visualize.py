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

def plot_images(imgs, titles=None, cmaps='gray', dpi=100, pad=.5,
                adaptive=True, hpad=0.):
    """Plot a set of images horizontally.
    Args:
        imgs: a list of NumPy or PyTorch images, RGB (H, W, 3) or mono (H, W).
        titles: a list of strings, as titles for each image.
        cmaps: colormaps for monochrome images.
        adaptive: whether the figure size should fit the image aspect ratios.
    """
    n = len(imgs)
    if not isinstance(cmaps, (list, tuple)):
        cmaps = [cmaps] * n

    if adaptive:
        ratios = [i.shape[1] / i.shape[0] for i in imgs]  # W / H
    else:
        ratios = [4/3] * n
    figsize = [sum(ratios)*4.5, 4.5+ hpad]
    fig, ax = plt.subplots(
        1, n, figsize=figsize, dpi=dpi, gridspec_kw={'width_ratios': ratios})
    if n == 1:
        ax = [ax]
    for i in range(n):
        ax[i].imshow(imgs[i], cmap=plt.get_cmap(cmaps[i]))
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        ax[i].set_axis_off()
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
        if titles:
            ax[i].set_title(titles[i])
    fig.tight_layout(pad=pad)
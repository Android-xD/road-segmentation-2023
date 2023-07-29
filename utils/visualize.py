import matplotlib.pyplot as plt
import numpy as np


def show_img_mask(img, mask):
    """
    Displays an image and its corresponding mask side by side in a single figure.

    Args:
        img (torch.Tensor): The input image tensor of shape (c, h, w), where
                            c = channels, h = height, and w = width.
        mask (torch.Tensor): The input mask tensor of shape (h, w).
    """
    figure = plt.figure(figsize=(8, 8))
    figure.add_subplot(1, 2, 1)
    plt.imshow(np.transpose(img.squeeze(),(1,2,0)))
    figure.add_subplot(1, 2, 2)
    plt.imshow(mask.squeeze())
    plt.show()

def overlay(input,output, alpha=0.3):
    img = np.transpose(input.squeeze(), (1, 2, 0)).numpy()
    img = img / 255.

    mask = output.squeeze().numpy()
    color_map = plt.get_cmap('jet')  # choose a colormap
    color_image = color_map(mask)
    alpha = np.clip(mask[:, :, None],0,1)
    return (1 - alpha) * img + alpha * color_image[:, :, :3]


def plot_images(imgs, titles=None, cmaps='gray', dpi=100, pad=.5,
                adaptive=True, hpad=1.):
    """
    source: https://github.com/cvg/Hierarchical-Localization/blob/master/hloc/utils/viz.py
    Display a set of images horizontally in a single figure.

    Args:
        imgs (list): A list of images(RGB or monochrome).
        titles (list, optional): A list of strings used as titles for each image. If provided,
                                 it should have the same length as the number of images in `imgs`.
        cmaps (str or list, optional): Colormaps for displaying monochrome images.
                                       If a single string is provided, it will be used for all monochrome images.
                                       If a list is provided, each colormap will correspond to a monochrome image.
        dpi (int, optional): The resolution in dots per inch for the displayed figure. Default is 100.
        pad (float, optional): The padding between the images in the horizontal direction. Default is 0.5.
        adaptive (bool, optional): Whether the figure size should adjust to fit the image aspect ratios.
                                   If True, the aspect ratios of the images will be used to determine the figure size.
                                   If False, a fixed aspect ratio of 4:3 will be used for all images.
                                   Default is True.
        hpad (float, optional): The horizontal padding of the figure. Default is 1.
    """
    # Determine the number of images to be displayed
    n = len(imgs)

    # If a single colormap is provided, replicate it to match the number of images
    if not isinstance(cmaps, (list, tuple)):
        cmaps = [cmaps] * n

    # Calculate the aspect ratios of the images for adaptive figure size
    if adaptive:
        ratios = [i.shape[1] / i.shape[0] for i in imgs]  # W / H
    else:
        ratios = [4/3] * n
    figsize = [sum(ratios)*4.5, 4.5 + hpad]

    # Create a new figure with the specified size and resolution
    fig, ax = plt.subplots(1, n, figsize=figsize, dpi=dpi, gridspec_kw={'width_ratios': ratios}, squeeze=False)

    # Iterate over each image and display it with the corresponding settings
    for i in range(n):
        ax[0][i].imshow(imgs[i], cmap=cmaps[i])

        # Remove ticks and frame
        ax[0][i].get_yaxis().set_ticks([])
        ax[0][i].get_xaxis().set_ticks([])
        ax[0][i].set_axis_off()
        for spine in ax[0][i].spines.values():
            spine.set_visible(False)
        
        # Set the title if provided
        if titles:
            ax[0][i].set_title(titles[i],fontsize=25)
        
    fig.tight_layout(pad=pad)
import cv2
import os
import matplotlib.pyplot as plt
from dataset import CustomImageDataset,test_train_split
import numpy as np
import torch
import visualize as vis
import torchvision.transforms as T
from deeplabv3 import createDeepLabv3,load_model
from sklearn.metrics import f1_score, accuracy_score
import torch.nn.functional as F
from mask_to_submission import mask_to_submission_strings
from utils import aggregate_tile, f1_score
from visualize import plot_images

torch.manual_seed(0)


def eval(model, dataset:CustomImageDataset, num_samples=10):
    for i in range(int(len(dataset)*0.9), len(dataset)):
        combined = torch.zeros(dataset[0][1].size()[-3:])
        for j in range(num_samples):
            img, target = dataset[i]

            output = model(model.preprocess(img.unsqueeze(0)))
            output = dataset.affineTransform.backward(output)
            #combined[output>0.01] = combined[output>0.01]*0.5 + output*0.5
            img = dataset.affineTransform.backward(img)
            vis.show_img_mask(img.detach(), output.detach())

def nms(img, dir=None ):
    if dir is None:
        img = cv2.GaussianBlur(img,ksize=(15,15),sigmaX=5)
        dx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        dy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        dir = (-np.arctan2(dy, dx)) % np.pi / np.pi
        gradient_magnitude = (dx ** 2 + dy ** 2) ** 0.5
    #else:
        #dir =cv2.GaussianBlur(dir,ksize=(15,15),sigmaX=5)

    angle_quantization = 4  # Number of discrete directions

    quantized_angle = np.round(dir*angle_quantization)

    suppressed = np.zeros_like(img)
    height, width = img.shape
    r = 4
    ri = np.array([i_ for i_ in range(-r, r + 1) if i_ != 0])
    for i in range(0, height):
        for j in range(0, width):
            if img[i, j] <= 0.1:
                suppressed[i, j] = 0
            else:
                direction = quantized_angle[i, j]
                if direction == 0:
                    idx_i, idx_j = i, j + ri
                elif direction == 1:
                    idx_i, idx_j = i + ri, j + ri
                elif direction == 2:
                    idx_i, idx_j = i + ri, j
                else:
                    idx_i, idx_j = i + ri, j - ri
                idx_i, idx_j = np.clip(idx_i, 0, height-1), np.clip(idx_j, 0, width-1)
                neighbors = img[idx_i, idx_j]

                if img[i, j] <= np.max(neighbors):
                    suppressed[i, j] = 1
                else:
                    suppressed[i, j] = 0

    return suppressed

def prop(img,dir):
    if dir is None:
        img = cv2.GaussianBlur(img,ksize=(15,15),sigmaX=5)
        dx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        dy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        dir = (-np.arctan2(dy, dx)) % np.pi / np.pi
        gradient_magnitude = (dx ** 2 + dy ** 2) ** 0.5
    #else:
        #dir =cv2.GaussianBlur(dir,ksize=(15,15),sigmaX=5)

    angle_quantization = 4  # Number of discrete directions

    quantized_angle = np.round(dir*angle_quantization)

    suppressed = np.zeros_like(img)
    height, width = img.shape
    r = 4
    p = np.zeros((5,*img.shape))
    for it in range(5):
        ri = np.array([i_ for i_ in range(-r, r + 1) if i_ != 0])
        for i in range(0, height):
            for j in range(0, width):
                if img[i, j] == 0:
                    suppressed[i, j] = 0
                else:
                    direction = (quantized_angle[i, j] + 2 )% 4
                    if direction == 0:
                        idx_i, idx_j = i, j + ri
                    elif direction == 1:
                        idx_i, idx_j = i + ri, j + ri
                    elif direction == 2:
                        idx_i, idx_j = i + ri, j
                    else:
                        idx_i, idx_j = i + ri, j - ri
                    idx_i, idx_j = np.clip(idx_i, 0, height-1), np.clip(idx_j, 0, width-1)
                    neighbors = img[idx_i, idx_j]

                    suppressed[i, j] = np.mean(neighbors)
        img = suppressed
        p[it]=suppressed
    # while(True):
    #     for it in range(5):
    #         cv2.imshow("",p[it])
    #         cv2.waitKey(100)

    return suppressed


if __name__ == '__main__':
    test_set = r"./data/test/images"
    training_set = r"./data/training"

    # Check if GPU is available
    use_cuda = torch.cuda.is_available()

    # Define the device to be used for computation
    device = torch.device("cuda" if use_cuda else "cpu")

    dataset = CustomImageDataset(training_set, test=True)

    train_dataset, val_dataset = test_train_split(dataset, 0.8)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=1,
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    model, preprocess = createDeepLabv3(5, 400)
    state_dict = torch.load("out/model_best.pth.tar", map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)

    for i, (input, target) in enumerate(val_loader):

        # Move input and target tensors to the device (CPU or GPU)
        input = input.to(device)
        #input = input.squeeze()
        target = target.to(device)
        output = model(preprocess(input))['out']
        # normalize the output
        pred = output[:, :1]
        sdf = output[:, 1:2]
        width = output[:, 2:3]
        pred = F.sigmoid(pred)
        sdf = F.sigmoid(sdf)
        width = F.relu6(width)
        dir = output[:,3:4] % 1
        tile = F.sigmoid(output[:,4:5])
        #dir[tile < 0.5] = 0
        #width[tile < 0.5] = 0

        tiled = tile > 0.5
        gt = target[:,2:3]
        print(f1_score(gt, tiled))
        for j in range(4):
            w = (width[j,0].cpu().detach().numpy() * 70).astype(np.uint8)
            w = cv2.medianBlur(w, ksize=15)/70.
            plt.scatter(gt[j].cpu().detach().numpy().ravel(), w.ravel(), alpha=0.1)
        plt.show()

        for j in range(target.shape[0]):
            img = np.transpose(input.cpu().detach()[j] / 255., (1, 2, 0))

            out = [pred, sdf, width, dir, tile, tiled, gt]
            out = [o[j, 0].cpu().detach().numpy() for o in out]
            #for x in [0, 1, 2, 3, 4, 5]:
            #    w = (out[x] * 255).astype(np.uint8)
            #    out[x] = cv2.medianBlur(w, ksize=15)/255.
            out[5] = out[1] < out[2]*0.5

            #vis.direction_field(out[3], out[0],8)
            images = [img]+out
            names = ["img", "probability", "signed distance","width", "direction", "patch prediction", "final prediction", "gt prediction"]
            #plot_images(images, names)
            plot_images([img[:, :, 0] + prop(out[0], out[3]), out[3]])
            #plt.show()
            plt.show()
            #plt.savefig(f"./figures/out_{i}_{j}.jpg")

    target = aggregate_tile(target.to(float))

    pred = (output > 0.2)
    pred_string = mask_to_submission_strings(pred)

    pred = aggregate_tile(pred*1.0)
    #cv2.imwrite(f"prediction/{}")

    plt.imshow(pred[0,0] * 1.0)
    plt.show()
    plt.imshow(target[0,0] * 1.0)
    plt.show()
    """accuracy = torch.count_nonzero(target == pred)/target.numel()
    TP = torch.count_nonzero(target[1 == pred])
    recall = TP / torch.count_nonzero(target)
    precision = TP / torch.count_nonzero(pred)
    f1 = 2./(1/recall + 1/precision)
    print(accuracy, recall, precision, f1)"""


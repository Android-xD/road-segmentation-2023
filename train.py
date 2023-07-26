import cv2
import matplotlib.pyplot as plt
from dataset import CustomImageDataset, split
import numpy as np
import torch
import time
import visualize as vis
import torchvision.transforms as T
from sklearn.metrics import f1_score, accuracy_score
from deeplabv3 import createDeepLabv3, load_model
from unet_backbone import get_Unet
from other_models import get_resnext, get_fpn
import os
from tensorboardX import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler
from utils import aggregate_tile, bce_loss, rich_loss, f1_score, accuracy_precision_and_recall
from torchgeometry.losses import dice, focal, tversky
import torch.nn.functional as F
from torch.utils.data import Subset

torch.manual_seed(0)
# Check if GPU is available
use_cuda = torch.cuda.is_available()
# Define the device to be used for computation
device = torch.device("cuda:0" if use_cuda else "cpu")

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Train image segmentation network')
    # training
    parser.add_argument('--out_dir',
                        help='directory to save outputs',
                        default='out',
                        type=str)
    parser.add_argument('--frequent',
                        help='frequency of logging',
                        default=1,
                        type=int)
    parser.add_argument('--eval_interval',
                        help='evaluation interval',
                        default=1,
                        type=int)
    parser.add_argument('--epochs',
                        help='num of epochs',
                        default=25,
                        type=int)

    args = parser.parse_args()

    return args


def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth.tar'):
    """Save model checkpoint

    Args:
        states: model states.
        is_best (bool): whether to save this model as best model so far.
        output_dir (str): output directory to save the checkpoint
        filename (str): checkpoint name
    """
    os.makedirs(output_dir, exist_ok=True)
    # torch.save(states, os.path.join(output_dir, filename))
    if is_best and 'state_dict' in states:
        torch.save(states['state_dict'],
                   os.path.join(output_dir, 'model_best.pth.tar'))


if __name__ == '__main__':
    args = parse_args()

    training_set = r"./data/training"

    run_id = time.strftime("_%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=f"out/logs/{run_id}")

    train_epochs = 30  # 20 epochs should be enough
    rich_labels = False

    get_model = get_Unet #createDeepLabv3 # get_fpn#get_Unet #

    output_channels = 5 if rich_labels else 1
    model, preprocess, postprocess = get_model(output_channels, 400)
    #state_dict = torch.load("out/model_best_google30.pth.tar", map_location=torch.device("cpu"))
    #model.load_state_dict(state_dict)

    dataset_aug = CustomImageDataset(training_set, train=True, rich=rich_labels, geo_aug=True, color_aug=True)
    dataset_clean = CustomImageDataset(training_set, train=True, rich=rich_labels, geo_aug=False, color_aug=False)

    train_indices, cal_indices, val_indices = split(len(dataset_aug), [0.90, 0.05, 0.05])
    train_dataset = Subset(dataset_aug, train_indices)
    cal_dataset = Subset(dataset_clean, cal_indices)
    val_dataset = Subset(dataset_clean, val_indices)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        drop_last=True
    )
    cal_loader = torch.utils.data.DataLoader(
        cal_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        drop_last=True
    )

    if rich_labels:
        loss_fn = rich_loss
    else:
        loss_fn = bce_loss


    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1., end_factor=1.0, total_iters=60)


    best_score = 100
    for epoch in range(train_epochs):
        # train for one epoch
        model.train()
        train_loss = 0.0
        train_accuracy = 0.0
        train_precision = 0.0
        train_recall = 0.0
        for i, (input, target) in enumerate(train_loader):
            # Move input and target tensors to the device (CPU or GPU)
            input = input.to(device)
            target = target.to(device)

            # Clear the gradients
            optimizer.zero_grad()

            # Forward pass
            output = postprocess(model(preprocess(input)))

            loss = loss_fn(output, target)
            loss.backward()
            # Accumulate loss
            train_loss += loss.item()

            y_gt = target[:, :1]
            y_pred = F.sigmoid(output[:, :1])

            a, b, c = accuracy_precision_and_recall(aggregate_tile(y_gt), aggregate_tile((y_pred>0.5)*1.))
            train_accuracy += a
            train_precision += b
            train_recall += c

            optimizer.step()

            # Print progress
            if (i + 1) % args.frequent == 0:
                print(f'Train Epoch: {epoch + 1} [{i + 1}/{len(train_loader)}]\t'
                      f'Loss: {train_loss / (i + 1):.4f}\t'
                      f'Accuracy: {train_accuracy/(i+1):.4f}')

        scheduler.step()
        # print(f"lr{optimizer.param_groups[0]['lr']}")
        train_accuracy /= i + 1
        train_loss /= i + 1
        train_recall /= i+1
        train_precision /= i+1
        train_f1 = 1/(1/train_recall + 1/train_precision)
        # Calibration
        cal_loss = 0.0
        cal_accuracy = 0.0
        cal_precision = 0.0
        cal_recall = 0.0

        model.eval()
        with torch.no_grad():
            # calibrate thresholds on calibration split
            m = len(cal_loader)
            n_ticks = 101
            recall_space_16 = torch.zeros((m, n_ticks, n_ticks))
            precision_space_16 = torch.zeros((m, n_ticks, n_ticks))
            ticks = torch.linspace(0, 1, n_ticks)

            for i, (input, target) in enumerate(cal_loader):
                # Move input and target tensors to the device (CPU or GPU)
                input = input.to(device)
                target = target.to(device)
                output = postprocess(model(preprocess(input)))
                y_pred = F.sigmoid(output[:, :1])
                y_gt = target[:, :1]
                agg_target = aggregate_tile(target[:, :1])
                for r, th1 in enumerate(ticks):
                    for c, th2 in enumerate(ticks):
                        _, precision_space_16[i, r, c], recall_space_16[i, r, c] = \
                            accuracy_precision_and_recall(agg_target, aggregate_tile((y_pred > th1) * 1.0, thresh=th2))

                a, b, c = accuracy_precision_and_recall(aggregate_tile(y_gt), aggregate_tile((y_pred > 0.5) * 1.))
                cal_accuracy += a
                cal_precision += b
                cal_recall += c

                # Print progress
                if (i + 1) % args.frequent == 0:
                    print(f'Calibration Epoch: {epoch + 1} [{i + 1}/{len(cal_loader)}]\t'
                          f'Loss: {cal_loss / (i + 1):.4f}\t'
                          f'Accuracy: {cal_accuracy / (i + 1):.4f}')

            recall_space_16 = torch.mean(recall_space_16, dim=0)
            precision_space_16 = torch.mean(precision_space_16, dim=0)
            f1_space = 2. / (1 / recall_space_16 + 1 / precision_space_16)
            cal_f1_cal = torch.max(f1_space)
            loc = (f1_space == torch.max(f1_space)).nonzero()[0]
            cal_recall_cal = recall_space_16[loc[0],loc[1]]
            cal_precision_cal = precision_space_16[loc[0],loc[1]]
            th1, th2 = ticks[loc[0]], ticks[loc[1]]
            cal_accuracy /= i + 1
            cal_precision /= i + 1
            cal_recall /= i+1
            cal_f1 = 2/(1/train_recall + 1/train_precision)
            print(f'Calibration: \t'
                  f'F1 Uncalibrated: {cal_f1:.4f}\t'
                  f'F1 Calibrated: {cal_f1_cal:.4f}\t'
                  f'Thresholds: ({th1:.2f},{th2:.2f})')

            print(
                f'Uncalibrated \tLoss: {cal_loss:.4f}\t F1: {cal_f1:.4f} \t Accuracy: {cal_accuracy:.4f}\t'
                f'Recall: {cal_recall:.4f}\t Precision: {cal_precision:.4f}')
            print(
                f'Calibrated \tLoss: {cal_loss:.4f}\t F1: {cal_f1_cal:.4f} \t Accuracy: ---- \t'
                f'Recall: {cal_recall_cal:.4f}\t Precision: {cal_precision_cal:.4f}')

            # validation
            val_loss = 0.0
            val_recall = 0.0
            val_precision = 0.0
            val_accuracy = 0.0
            val_recall_cal = 0.0
            val_precision_cal = 0.0
            val_accuracy_cal = 0.0
            for i, (input, target) in enumerate(val_loader):
                # Move input and target tensors to the device (CPU or GPU)
                input = input.to(device)
                target = target.to(device)

                y_gt = target[:, :1]
                # Forward pass
                output = postprocess(model(preprocess(input)))

                loss = loss_fn(output, target)

                # Accumulate loss
                val_loss += loss.item()

                y_pred = F.sigmoid(output[:, :1])
                pred = (y_pred > 0.5)*1.

                a, b, c = accuracy_precision_and_recall(aggregate_tile(y_gt), aggregate_tile((y_pred > 0.5) * 1.))
                val_accuracy += a
                val_precision += b
                val_recall += c

                a, b, c = accuracy_precision_and_recall(aggregate_tile(y_gt), aggregate_tile((y_pred > th1) * 1., th2))
                val_accuracy_cal += a
                val_precision_cal += b
                val_recall_cal += c


            val_loss /= i + 1
            val_accuracy /= i + 1
            val_recall /= i + 1
            val_precision /= i + 1
            val_accuracy_cal /= i + 1
            val_recall_cal /= i + 1
            val_precision_cal /= i + 1
            val_f1 = 2/(1/val_recall + 1/val_precision)
            val_f1_cal = 2/(1/val_recall_cal + 1/val_precision_cal)

        is_best = val_loss < best_score
        if is_best:
            best_score = val_loss

        # Print progress
        print(f'Validation Epoch: {epoch + 1}\tLoss: {val_loss:.4f}\t F1: {val_f1:.4f} \t Accuracy: {val_accuracy:.4f}\t'
              f'Recall: {val_recall:.4f}\t Precision: {val_precision:.4f}')
        print(f'Val Calibr Epoch: {epoch + 1}\tLoss: {val_loss:.4f}\t F1: {val_f1_cal:.4f} \t Accuracy: {val_accuracy_cal:.4f}\t'
              f'Recall: {val_recall_cal:.4f}\t Precision: {val_precision_cal:.4f}')
        writer.add_scalars("Loss", {"val": val_loss, "train": train_loss}, epoch)
        writer.add_scalar("F1/val", val_f1 / (i + 1), epoch)
        writer.add_scalars("Accuracy", {"val": val_accuracy, "train": train_accuracy}, epoch)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'perf': val_loss,
            'last_epoch': epoch,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.out_dir, filename=f'checkpoint{epoch + 1}.pth.tar')

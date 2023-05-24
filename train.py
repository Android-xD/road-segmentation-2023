import cv2
import matplotlib.pyplot as plt
from dataset import CustomImageDataset,test_train_split
import numpy as np
import torch
import time
import visualize as vis
import torchvision.transforms as T
from sklearn.metrics import f1_score, accuracy_score
from deeplabv3 import createDeepLabv3, load_model
import os
from tensorboardX import SummaryWriter

torch.manual_seed(0)


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
    parser.add_argument('--gpu',
                        action='store_true',
                        help='whether to use GPU or not')
    parser.add_argument('--num_workers',
                        help='num of dataloader workers',
                        default=4,
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
    #torch.save(states, os.path.join(output_dir, filename))
    if is_best and 'state_dict' in states:
        torch.save(states['state_dict'],
                   os.path.join(output_dir, 'model_best.pth.tar'))

if __name__ == '__main__':
    dataset_path = r"./data/test_set_images"
    training_set = r"./data/training"

    # Check if GPU is available
    use_cuda = torch.cuda.is_available()

    # Define the device to be used for computation
    device = torch.device("cuda:0" if use_cuda else "cpu")
    run_id = time.strftime("_%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=f"out/logs/{run_id}")

    dataset = CustomImageDataset(training_set)

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
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    model, preprocess = createDeepLabv3(2, 400)

    args = parse_args()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    def loss_fn(output,target):
        """ balanced binary cross entropy loss"""
        # need to balance the segmentation
        fg_ratio = torch.clamp(torch.count_nonzero(target) / target.numel(), 0.05, 0.95)
        weight = torch.tensor([1/(1-fg_ratio), 1/fg_ratio])
        weight /= torch.sum(weight)
        # Compute loss
        weight = weight.to(device)
        loss_fn = torch.nn.CrossEntropyLoss()#weight=weight)
        target = target.squeeze(1)
        return loss_fn(output, target)

    train_epochs = 20  # 20 epochs should be enough, if your implementation is right
    best_score = 0
    for epoch in range(train_epochs):
        # train for one epoch
        model.train()
        train_loss = 0.0
        train_accuracy = 0.0
        for i, (input, target) in enumerate(train_loader):
            # Move input and target tensors to the device (CPU or GPU)
            input = input.to(device)
            target = target.to(device)

            # Clear the gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(preprocess(input))['out']

            loss = loss_fn(output, target)

            # Backward pass and update weights
            loss.backward()
            optimizer.step()

            # Accumulate loss
            train_loss += loss.item()
            train_accuracy += torch.count_nonzero(target == (output[:, 1:2] > 0.5))/target.numel()


            # Print progress
            if (i+1) % args.frequent == 0:
                print(f'Train Epoch: {epoch+1} [{i+1}/{len(train_loader)}]\t'
                      f'Loss: {train_loss / (i + 1):.4f}')
        train_accuracy/=i+1
        train_loss/=i+1
        # Validation
        model.eval()
        val_loss = 0.0
        val_f1 = 0.0
        val_accuracy = 0.0
        with torch.no_grad():
            for i, (input, target) in enumerate(val_loader):
                # Move input and target tensors to the device (CPU or GPU)
                input = input.to(device)
                target = target.to(device)

                # Forward pass
                output = model(preprocess(input))['out']
                # Compute loss
                loss = loss_fn(output, target)

                # Accumulate loss
                val_loss += loss.item()

                pred = output.round().detach().cpu().numpy()
                true = target.detach().cpu().numpy()
                #val_f1 += f1_score(true.ravel(), pred.ravel())
                val_accuracy += torch.count_nonzero(target == (output[:, 1:2] > 0.5))/target.numel()
            val_accuracy/=i+1
            val_loss/=i+1	

        is_best = val_loss < best_score
        if is_best:
            best_score = val_loss

        # Print progress
        print(f'Validation Epoch: {epoch+1}\tLoss: {val_loss:.4f}\t F1: {val_f1} \t Accuracy: {val_accuracy}')
        writer.add_scalars("Loss",{"val":val_loss,"train": train_loss}, epoch)
        writer.add_scalar("F1/val", val_f1 / (i + 1), epoch)
        writer.add_scalars("Accuracy", {"val": val_accuracy,"train": train_accuracy}, epoch)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'perf': val_loss,
            'last_epoch': epoch,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.out_dir, filename=f'checkpoint{epoch+1}.pth.tar')

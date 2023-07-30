import os

import torch
import torch.nn.functional as F
import torch.nn.functional as func
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from torch.autograd import Variable

from dataset import CustomImageDataset, split
from models.deeplabv3 import createDeepLabv3
from resample import resample_output
from utils.utils import (aggregate_tile, f1_loss, f1_score, nanstd,
                         quantile_aggregate_tile)

torch.manual_seed(0)

# Check if GPU is available
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class decoder(torch.nn.Module):
    """
    A simple decoder network that takes in a vector of features and outputs a single value.
    """
    def __init__(self, num_inputs):
        super(decoder, self).__init__()
        self.linear1 = torch.nn.Linear(num_inputs, 512)
        self.linear2 = torch.nn.Linear(512, 512)
        self.linear3 = torch.nn.Linear(512, 1)

    def forward(self, x):
        x = (func.relu(self.linear1(x)))
        x = (func.relu(self.linear2(x)))
        return (func.sigmoid(self.linear3(x)))


def get_patch_dataset():
    """
    Extract features and labels from the custom image dataset.

    Returns:
        X (torch.Tensor): A tensor containing the extracted features from the dataset.
        y (torch.Tensor): A tensor containing the corresponding labels for the features.
    """
    training_set = r"./data/training"

    # Load DeepLabv3 model
    model, preprocess, _ = createDeepLabv3(1, 400)
    state_dict = torch.load("out/model_best.pth.tar", map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()

    # Prepare dataset
    dataset = CustomImageDataset(training_set, train=True, color_aug=False, geo_aug=False)
    m = len(dataset)
    num_patches_per_image = (400//16)**2
    num_patches = m * num_patches_per_image
    num_ticks = 11
    num_features = num_ticks*4
    X = torch.zeros((num_patches, num_features))
    y = torch.zeros((num_patches, 1))
    print(F"num images {m}\t num patches: {num_patches}")

    # Function to query the DeepLabv3 model for output
    query = lambda input : model(preprocess(input))['out']

    # Extract features and labels
    for i in range(m):
        target = dataset[i][1].unsqueeze(0)
        output_samples, output_masks = resample_output(query, training_set, i, 20)
        output_i = F.sigmoid(output_samples[:1])
        output_samples[output_masks == 0] = torch.nan
        output_mode, _ = F.sigmoid(output_samples).nanmedian(keepdim=True, dim=0)
        output_mean = F.sigmoid(output_samples).nanmean(keepdim=True, dim=0)
        output_std = nanstd(F.sigmoid(output_samples), keepdim=True, dim=0)
        for j, output in enumerate([output_i, output_mean, output_mode, output_std]):
            quantiles = quantile_aggregate_tile(output, num_ticks)
            quantiles = torch.flatten(quantiles, start_dim=1, end_dim=4).T
            X[num_patches_per_image*i:num_patches_per_image*(i+1), num_ticks*j:num_ticks*(j+1)] = quantiles

        agg = aggregate_tile(target[:, :1], 0.25)
        agg = torch.flatten(agg, start_dim=0, end_dim=3).reshape(-1,1)
        y[num_patches_per_image*i:num_patches_per_image*(i+1)] = agg
        X = torch.nan_to_num_(X,0)
    return X, y


if __name__ == '__main__':
    # Get the feature dataset and labels
    X, y = get_patch_dataset()

    # Save the feature dataset and labels for future use
    torch.save(X, "X.pt")
    torch.save(y, "y.pt")

    # Load the dataset and labels
    X, y = torch.load("X.pt"), torch.load("y.pt")

    # Get the number of samples and features
    n_samples, n_features = X.shape

    output_dir = "out"

    # Set the random seed for reproducibility
    torch.manual_seed(0)

    # Split the data into training and testing sets
    train_indices, test_indices = split(n_samples, [0.90, 0.1])
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    print(X_train.shape, y_train.shape)

    # Train a logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train.ravel())

    # Evaluate the model using the F1 score
    predicted_probs = model.predict_proba(X_test)[:, 1]  # Probability of the positive class (class 1)
    predicted_labels = model.predict(X_test)
    print(f1_score(y_test.ravel(), torch.tensor(predicted_labels)))

    store_folder = "out/prediction"
    os.makedirs(store_folder, exist_ok=True)

    # move the model to the device (GPU or CPU)
    classifier = decoder(n_features)
    classifier = classifier.to(device)

    # define the loss function and the optimizer
    loss = torch.nn.BCELoss()  # note that CrossEntropyLoss is for targets with more than 2 classes.
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.0005)

    # train the classifier
    n_epochs = 100
    for e in range(n_epochs):
        x_data = Variable(X_train, requires_grad=False)
        y_data = Variable(y_train, requires_grad=False)
        x_b, y_b = x_data.to(device), y_data.to(device)

        # Get the baseline prediction
        baseline_pred = (torch.mean((x_b > 0.25) * 1., dim=1) > 0.25) * 1.
        y = classifier(x_b)

        # Compute the loss
        l = f1_loss(y_b, y)
        baseline_pred = (torch.mean((x_b > 0.25) * 1., dim=1) > 0.25) * 1.
        learned_pred = (y > 0.5) * 1.

        # Compute the F1 score for the learned and baseline predictions
        f1_learned = f1_score(y_b, learned_pred)
        f1_thresh = f1_score(y_b, baseline_pred)
        print(f"Train epoch:\t\t{e}\tloss:\t{l.item():.4f},\t pred:\t{f1_learned:.4f},\t baseline:\t{f1_thresh:.4f}\t impr:{f1_learned - f1_thresh:.4f}")
        optimizer.zero_grad()

        # backprop
        l.backward()
        optimizer.step()

        # validation
        classifier.eval()
        with torch.no_grad():
            x_b, y_b = X_test.to(device), y_test.to(device)
            baseline_pred = (torch.mean((x_b > 0.25) * 1., dim=1) > 0.25) * 1.
            y = classifier(x_b)
            l = f1_loss(y_b, y)
            baseline_pred = (torch.mean((x_b > 0.25) * 1., dim=1) > 0.25) * 1.
            learned_pred = (y > 0.5) * 1.
            f1_learned = f1_score(y_b, learned_pred)
            f1_thresh = f1_score(y_b, baseline_pred)
            print(
                f"Validation epoch:\t{e}\tloss:\t{l.item():.4f},\t pred:\t{f1_learned:.4f},\t baseline:\t{f1_thresh:.4f}\t impr:{f1_learned - f1_thresh:.4f}")

    # save the model
    torch.save(classifier.state_dict(), os.path.join(output_dir, f'best_classifier.pth.tar'))
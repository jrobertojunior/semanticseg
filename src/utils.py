import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2 as cv
### data processing ###


def toOneHot(mask, nb_class=10):
    """
    Convert label image to onehot encoding

    Args:
        mask (Image): mask containing pixels labels
        nb_class (int): number of class
    """
    classes = [
        [0, 0, 0],
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [128, 128, 128],
        [255, 255, 0],
        [255, 0, 255],
        [0, 255, 255],
        [255, 255, 255],
        [255, 192, 192],
        [0, 128, 128],
    ]

    _mask = np.array(mask)
    _categorical = np.zeros((_mask.shape[0], _mask.shape[1], nb_class))

    # enumerate classes
    for i, c in enumerate(classes):
        _categorical[np.where(_mask == c), i] = 1

    # # enumerate classes
    # for i, c in enumerate(classes):
    #     # loop over mask
    #     for x in range(_mask.shape[0]):
    #         for y in range(_mask.shape[1]):
    #             if np.all(_mask[x, y] == c):
    #                 _categorical[x, y, i] = 1

    

    # print(_categorical)

    # reshape mask to [65536, 3]
    mask = np.array(mask).reshape(-1, 3)

    # categorical = torch.from_numpy(np.array(mask)).long()
    categorical = torch.from_numpy(_categorical).long()
    # return categorical

    # print(categorical.shape, categorical)
    # categorical = F.one_hot(categorical, nb_class).transpose(1, 4).squeeze(-1)

    return categorical.permute(2, 0, 1).float()


### plots ###


def show_learning(model):
    """
    Plot loss and accuracy from model

    Args:
        model: Unet model
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # plot losses
    axes[0].plot(model.train_loss, label="train")
    axes[0].plot(model.valid_loss, label="validation")
    axes[0].set_xlabel("Epochs")

    try:
        axes[0].set_ylabel(model.criterion._get_name())
    except:
        axes[0].set_ylabel("Loss")

    axes[0].set_title("Loss evolution")
    axes[0].legend()

    # plot accuracy
    axes[1].plot(model.train_accuracy, label="train")
    axes[1].plot(model.valid_accuracy, label="validation")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("IOU score")
    axes[1].set_title("Accuracy evolution")

    axes[1].legend()

    plt.show()


### losses & accuracy ###


def dice_loss(yhat, ytrue, epsilon=1e-6):
    """
    Computes a soft Dice Loss

    Args:
        yhat (Tensor): predicted masks
        ytrue (Tensor): targets masks
        epsilon (Float): smoothing value to avoid division by 0
    output:
        DL value with `mean` reduction
    """
    # compute Dice components
    intersection = torch.sum(yhat * ytrue, (1, 2, 3))
    cardinal = torch.sum(yhat + ytrue, (1, 2, 3))

    return torch.mean(1.0 - (2 * intersection / (cardinal + epsilon)))


def tversky_index(yhat, ytrue, alpha=0.3, beta=0.7, epsilon=1e-6):
    """
    Computes Tversky index

    Args:
        yhat (Tensor): predicted masks
        ytrue (Tensor): targets masks
        alpha (Float): weight for False positive
        beta (Float): weight for False negative
                    `` alpha and beta control the magnitude of penalties and should sum to 1``
        epsilon (Float): smoothing value to avoid division by 0
    output:
        tversky index value
    """
    TP = torch.sum(yhat * ytrue, (1, 2, 3))
    FP = torch.sum((1.0 - ytrue) * yhat, (1, 2, 3))
    FN = torch.sum((1.0 - yhat) * ytrue, (1, 2, 3))

    return TP / (TP + alpha * FP + beta * FN + epsilon)


def tversky_loss(yhat, ytrue):
    """
    Computes tversky loss given tversky index

    Args:
        yhat (Tensor): predicted masks
        ytrue (Tensor): targets masks
    output:
        tversky loss value with `mean` reduction
    """
    return torch.mean(1 - tversky_index(yhat, ytrue))


def tversky_focal_loss(yhat, ytrue, alpha=0.7, beta=0.3, gamma=0.75):
    """
    Computes tversky focal loss for highly umbalanced data
    https://arxiv.org/pdf/1810.07842.pdf

    Args:
        yhat (Tensor): predicted masks
        ytrue (Tensor): targets masks
        alpha (Float): weight for False positive
        beta (Float): weight for False negative
                    `` alpha and beta control the magnitude of penalties and should sum to 1``
        gamma (Float): focal parameter
                    ``control the balance between easy background and hard ROI training examples``
    output:
        tversky focal loss value with `mean` reduction
    """

    return torch.mean(torch.pow(1 - tversky_index(yhat, ytrue, alpha, beta), gamma))


def focal_loss(yhat, ytrue, alpha=0.75, gamma=2):
    """
    Computes α-balanced focal loss from FAIR
    https://arxiv.org/pdf/1708.02002v2.pdf

    Args:
        yhat (Tensor): predicted masks
        ytrue (Tensor): targets masks
        alpha (Float): weight to balance Cross entropy value
        gamma (Float): focal parameter
    output:
        loss value with `mean` reduction
    """

    # compute the actual focal loss
    focal = -alpha * torch.pow(1.0 - yhat, gamma) * torch.log(yhat)
    f_loss = torch.sum(ytrue * focal, dim=1)

    return torch.mean(f_loss)


def iou_accuracy(yhat, ytrue, threshold=0.5, epsilon=1e-6):
    """
    Computes Intersection over Union metric

    Args:
        yhat (Tensor): predicted masks
        ytrue (Tensor): targets masks
        threshold (Float): threshold for pixel classification
        epsilon (Float): smoothing parameter for numerical stability
    output:
        iou value with `mean` reduction
    """
    intersection = ((yhat > threshold).long() & ytrue.long()).float().sum((1, 2, 3))
    union = ((yhat > threshold).long() | ytrue.long()).float().sum((1, 2, 3))

    return torch.mean(intersection / (union + epsilon)).item()

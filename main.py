#!/usr/bin/env python3

from xml.dom.minidom import parse
import torch
import numpy as np

from src.framework import Context
from src.model import Unet
from src.utils import show_learning, focal_loss
from src.parsexml import parse_training_data

if __name__ == "__main__":
    # 1) Train the model
            
    # use gpu if available
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nAvailable device: {DEVICE}")

    print(f'\nParsing training data...\n')
    # path_dir = parse("./data/headsegmentation_dataset_ccncsa/training.xml")
    path_dir = parse_training_data("./data/headsegmentation_dataset_ccncsa/training.xml")

    img_path = path_dir["img"]
    masks_path = path_dir["mask"]
    # data folders
    # img_path = "./data/images"
    # masks_path = "./data/masks"

    # create model
    model = Unet(in_channels=3, out_channels=10, nb_features=16).to(DEVICE)

    print("nb_parameters: ", sum([param.nelement() for param in model.parameters()]))

    # Set training context
    ctx = Context().fit(
        img_names=img_path,
        mask_names=masks_path,
        img_size=256,
        mask_size=256,
        shuffle_data=False,
        device=DEVICE,
    )
    ctx.set(nb_epochs=100, batch_size=16)

    # Train
    ctx.train(model)

    # save model
    torch.save(model, "unet_16.pt")
    # training stats
    show_learning(model)

    # 2) Inference
    """
    saved_model = torch.load("unet_16.pt", map_location=torch.device('cpu'))

    img_name = "./data/images/001_scaled.png"
    mask_name = "./data/masks/001_scaled.png"

    ctx2 = Context()
    yhat = ctx2.predict(saved_model, img_name, mask_name, plot=True)
    """

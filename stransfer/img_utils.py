import torch
from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from stransfer import constants


def image_loader(image_name):
    image = Image.open(image_name)

    load_transforms = transforms.Compose([
        transforms.Resize(constants.IMSIZE),  # scale imported image
        transforms.ToTensor(),  # transform it into a torch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # normalize image using ImageNet mean and std
                             std=[0.229, 0.224, 0.225]),
    ])

    # fake batch dimension required to fit network's input dimensions
    image = load_transforms(image).unsqueeze(0)
    image = image.to(constants.DEVICE, torch.float)

    return image


def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension

    tpil = transforms.ToPILImage()
    image = tpil(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

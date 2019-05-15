import torch
from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from stransfer import constants


def image_loader(image_name):
    image = Image.open(image_name)

    min_dimension = min(transforms.ToTensor()(image).shape[1:])
    
    load_transforms = transforms.Compose([  
        transforms.CenterCrop(min_dimension), # crop imported image to be sqaured (min between height and width) 
        transforms.Resize(constants.IMSIZE),  # scale imported image
        transforms.ToTensor(),  # transform it into a torch tensor
    ])

    # fake batch dimension required to fit network's input dimensions
    image = load_transforms(image).unsqueeze(0)
    image = image.to(constants.DEVICE, torch.float)

    return image


def imshow(tensor, path="out.bmp"):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension

    tpil = transforms.ToPILImage()
    image = tpil(image)
    
    image.save(path)

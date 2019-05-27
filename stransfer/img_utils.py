import torch
from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from stransfer import constants


def image_loader_transform(image: Image) -> torch.Tensor:
    min_dimension = min(transforms.ToTensor()(image).shape[1:])

    load_transforms = transforms.Compose([
        # crop imported image to be sqaured (min between height and width)
        transforms.CenterCrop(min_dimension),
        transforms.Resize(constants.IMSIZE),  # scale imported image
        transforms.ToTensor(),  # transform it into a torch tensor
        # transforms.Normalize(  # Normalize using imagenet mean and std
        #     mean=constants.IMAGENET_MEAN,
        #     std=constants.IMAGENET_STD
        # )
    ])

    # fake batch dimension required to fit network's input dimensions
    image = load_transforms(image).unsqueeze(0)

    img_mean = (torch.tensor(constants.IMAGENET_MEAN)
                .view(-1, 1, 1)
                .to(constants.DEVICE))
    img_std = (torch.tensor(constants.IMAGENET_STD)
               .view(-1, 1, 1)
               .to(constants.DEVICE))

    image = image.to(constants.DEVICE, torch.float)

    image = (image - img_mean) / img_std

    return image


def concat_images(im1, im2, dim=2) -> torch.Tensor:
    return torch.cat([
        im1,
        im2],
        dim=dim)


def image_loader(image_name) -> torch.Tensor:
    image = Image.open(image_name)

    return image_loader_transform(image)


def imshow(image_tensor, ground_truth_image=None, path="out.bmp"):
    # concat with ground truth if specified
    if ground_truth_image is not None:
        image_tensor = concat_images(image_tensor, ground_truth_image)

    # denormalize image
    img_mean = (torch.tensor(constants.IMAGENET_MEAN)
                .view(-1, 1, 1))
    img_std = (torch.tensor(constants.IMAGENET_STD)
               .view(-1, 1, 1))

    image_tensor = (image_tensor * img_std) + img_mean

    # clamp image to legal RGB values before showing
    image = torch.clamp(
        image_tensor.cpu().clone(),  # we clone the tensor to not do changes on it
        min=0,
        max=255
    )

    image = image.squeeze(0)      # remove the fake batch dimension, if any

    # concat with ground truth if any
    tpil = transforms.ToPILImage()
    image = tpil(image)

    image.save(path)

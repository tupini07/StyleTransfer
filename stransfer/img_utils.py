"""
This module holds functionality related to loading and saving images.
"""

import torch
from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from stransfer import constants


def image_loader_transform(image: Image) -> torch.Tensor:
    """
    Transform a PIL.Image instance into a torch.Tensor one.

    :param image: image to transform
    :return: torch.Tensor representing the image
    """
    min_dimension = min(transforms.ToTensor()(image).shape[1:])

    load_transforms = transforms.Compose([
        # crop imported image to be sqaured (min between height and width)
        transforms.CenterCrop(min_dimension),
        transforms.Resize(constants.IMSIZE),  # scale imported image
        transforms.ToTensor(),  # transform it into a torch tensor
    ])

    # fake batch dimension required to fit network's input dimensions
    image = load_transforms(image).unsqueeze(0)

    # normalize image with IMAGENET mean and std.
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
    """
    Simple wrapper function that concatenates two images
    along a given dimension.

    :param im1:
    :param im2:
    :param dim: the dimension we want to concatenate across. By default it is
        the third dimension (width if images only have 3 dims)
    :return: tensor representing concatenated image
    """
    return torch.cat([
        im1,
        im2],
        dim=dim)


def image_loader(image_path: str) -> torch.Tensor:
    """
    Loads an image from `image_path`, and transforms it into a
    torch.Tensor

    :param image_path: path to the image that will be loaded
    :return: tensor representing the image
    """
    image = Image.open(image_path)

    return image_loader_transform(image)


def imshow(image_tensor: torch.Tensor, ground_truth_image: torch.Tensor = None,
           denormalize=True, path="out.bmp") -> None:
    """
    Utility function to save an input image tensor to disk.

    :param image_tensor: the tensor representing the image to be saved
    :param ground_truth_image: another tensor, representing another image. If provided, it will
        be concatenated to the `image_tensor` across the width dimension and this result will be
        saved to disk
    :param denormalize: whether or not to denormalize (using IMAGENET mean and std)
        the tensor before saving it to disk.
    :param path: the path where the image will be saved
    """

    # concat with ground truth if specified
    if ground_truth_image is not None:
        image_tensor = concat_images(image_tensor, ground_truth_image)

    if denormalize:
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

    image = image.squeeze(0)  # remove the fake batch dimension, if any

    # concat with ground truth if any
    tpil = transforms.ToPILImage()
    image = tpil(image)

    image.save(path)

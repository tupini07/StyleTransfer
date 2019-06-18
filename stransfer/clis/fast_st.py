import os

import click
import torch

from stransfer import c_logging, constants, img_utils, network

LOGGER = c_logging.get_logger()


@click.group()
def fast_st():
    """Fast Style Transfer"""
    pass


@fast_st.command()
@click.argument('style-image-path')
@click.option('-e', '--epochs', default=50,
              help="How many epochs the training will take")
@click.option('-b', '--batch-size', default=4, help="Batch size for training")
@click.option('-cw', '--content-weight', default=1,
              help="The weight we will assign to the content loss during the optimization")
@click.option('-sw', '--style-weight', default=100_000,
              help="The weight we will assign to the style loss during the optimization")
def train(style_image_path, epochs, batch_size, content_weight, style_weight):
    """
    Perform the training for the fast style transfer network. A checkpoint will be created
    at the end of each epoch in the `data/models/` directory.
    """
    style_name = style_image_path.split("/")[-1]
    LOGGER.info('Training fast style transfer network with style name: %s', style_name)

    style_image_path = os.path.join(constants.PROJECT_ROOT_PATH, style_image_path)

    style_image = img_utils.image_loader(style_image_path)

    net = network.ImageTransformNet(style_image, batch_size)
    net.static_train(style_name=style_name,
                     epochs=epochs,
                     style_weight=style_weight,
                     content_weight=content_weight)


@fast_st.command()
@click.argument('image-path')
@click.argument('style-name')
@click.option('-o', '--out-dir', default='results/',
              help='The results directory where the converted image will be saved')
def convert_image(image_path, style_name, out_dir):
    """
    Converts the image at `image-path` using the network pretrained with `style-name`
    and saves the resulting transformed image in `out-dir`.

    A pretrained model should exist in `data/models/` for the specified `style-name`.

    The files in `data/models/` contain the style names in their file names. For example,
    `fast_st_the_great_wave_epoch1.pth` was trained with the style `the_great_wave`
    """
    sty = network.ImageTransformNet(torch.rand([3, 255, 255]))
    sty.process_image(image_path=image_path,
                      style_name=style_name,
                      out_dir=out_dir)

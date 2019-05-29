import datetime
import logging

import click
import torch
from tqdm import tqdm

import colored_traceback
from stransfer import c_logging, constants, img_utils, network

LOGGER = logging.getLogger(__name__)


def run_gatys_style_transfer(style_path, content_path, steps=220, outdir=""):

    # TODO remove these 2
    content_path = "data/dancing.jpg"
    style_path = "data/styles/picasso.jpg"

    style_image = img_utils.image_loader(style_path)
    content_image = img_utils.image_loader(content_path)

    st_net = network.StyleNetwork(style_image, content_image)

    return st_net.train_gatys(style_image, content_image)


def run_fast_style_transfer(style_path, content_path, steps=220, outdir=""):

    content_path = "data/dancing.jpg"
    style_path = "data/styles/picasso.jpg"

    style_image = img_utils.image_loader(style_path)

    fast_net = network.ImageTransformNet(style_image).to(constants.DEVICE)

    fast_net.static_train('picasso')
    # fast_net.img_net_video_train()




@click.command()
@click.argument('style-image')
@click.argument('content')
@click.option('--video', is_flag=True, help="")
@click.option('--no-fast', is_flag=True, help="")  # don't use autoencoder
@click.option('--start-from-random-noise', is_flag=True, help="")
# TODO: check if this default is sensible
@click.option('-s', '--steps', default=300, help="")
# TODO: Do we want this option?
@click.option('-o', '--optimizer', type=click.Choice(['Adama', 'SGD']))
@click.option('-l', '--log-level', type=click.Choice([]))  # TODO
def cli(style_image, content, video, no_fast, start_from_random_noise, steps, optimizer):
    """
    Some doc
    """
    LOGGER.info('logging some stuff')
    print(123123)


def train_video():
    # TODO remove these 2
    content_path = "data/dancing.jpg"
    style_path = "data/styles/picasso.jpg"

    # TODO move to parameters
    style_weight = 1000000
    content_weight = 1

    style_image = img_utils.image_loader(style_path)
    content_image = img_utils.image_loader(content_path)

    net = network.VideoTransformNet(style_image)
    net.video_train()

if __name__ == "__main__":
    colored_traceback.add_hook()
    run_gatys_style_transfer(0,0)
    # train_video()
    # run_fast_style_transfer(0, 0)
    # cli(**{})  # suppress warning

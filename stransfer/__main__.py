import datetime
import logging
import os

import click
import torch
from pygments.formatters import img
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
    net.video_train("picasso")


def convert_video(video_path: str, style_name='picasso'):
    sty = network.VideoTransformNet(torch.rand([3, 255, 255]))
    sty.process_video(video_path, style_name=style_name)


def convert_image_fast(image_path: str, style_name='picasso'):
    sty = network.ImageTransformNet(torch.rand([3, 255, 255]))
    sty.process_image(image_path, style_name=style_name)


def convert_image_gatys(content_image_path: str,
                        style_image_path: str,
                        style_weight=1000000,
                        content_weight=1,
                        steps=500,
                        out_name='gatys_converted.png'):

    content_image_path = os.path.join(constants.PROJECT_ROOT_PATH, content_image_path)
    style_image_path = os.path.join(constants.PROJECT_ROOT_PATH, style_image_path)

    content_image = img_utils.image_loader(content_image_path)
    style_image = img_utils.image_loader(style_image_path)

    net = network.StyleNetwork(style_image, content_image)
    converted_image = net.train_gatys(style_image=style_image,
                                      content_image=content_image,
                                      style_weight=style_weight,
                                      content_weight=content_weight,
                                      steps=steps)

    out_dir = os.path.join(constants.PROJECT_ROOT_PATH, 'results')

    # ensure that the result directory exist
    os.makedirs(out_dir, exist_ok=True)

    out_file = os.path.join(out_dir, out_name)

    img_utils.imshow(converted_image, path=out_file)


if __name__ == "__main__":
    colored_traceback.add_hook()
    # convert_video()
    # convert_image_fast('data/dancing.jpg')
    # run_gatys_style_transfer(0,0)
    convert_image_gatys(content_image_path='data/dancing.jpg', style_image_path='data/styles/the_scream.jpg')

    # train_video()
    # run_fast_style_transfer(0, 0)
    # cli(**{})  # suppress warning

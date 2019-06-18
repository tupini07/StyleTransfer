import os

import click
import torch

from stransfer import c_logging, constants, img_utils, network

LOGGER = c_logging.get_logger()


@click.group()
def video_st():
    """Video Style Transfer"""
    pass


@video_st.command()
@click.argument('style-image-path')
@click.option('-e', '--epochs', default=50,
              help="How many epochs the training will take")
@click.option('-b', '--batch-size', default=4, help="Batch size for training")
@click.option('-cw', '--content-weight', default=1,
              help="The weight we will assign to the content loss during the optimization")
@click.option('-sw', '--style-weight', default=100_000,
              help="The weight we will assign to the style loss during the optimization")
@click.option('-tw', '--temporal-weight', default=0.8,
              help="The weight we will assign to the temporal loss during the optimization")
@click.option('--use-pretrained-fast-st', is_flag=True,
              help='States whether we want to start training the video model from pretrained fast '
                   'style transfer weights (which was trained on the same style name)')
def train(style_image_path, epochs, batch_size, content_weight, style_weight, temporal_weight, use_pretrained_fast_st):
    """
    Perform the training for the video style transfer network. A checkpoint will be created
    at the end of each epoch in the `data/models/` directory.

    We have the possibility of starting the training process for video style transfer from
    pretrained weights for a fast style transfer trained with the same `style name`. The
    weights that will be used are those of the highest epoch which correspond to that `style_name`.
    """

    style_name = style_image_path.split("/")[-1]
    LOGGER.info('Training video style transfer network with style name: %s', style_name)

    ft_pretrained_w = None
    if use_pretrained_fast_st:
        LOGGER.info('Trying to load pretrained fast ST weights')
        try:
            ft_pretrained_w = network._load_latest_model_weigths('fast_st', style_name)
        except AssertionError:
            LOGGER.warning("Couldn't load pretrained weights")

    style_image_path = os.path.join(constants.PROJECT_ROOT_PATH, style_image_path)
    style_image = img_utils.image_loader(style_image_path)

    net = network.VideoTransformNet(style_image,
                                    batch_size,
                                    fast_transfer_dict=ft_pretrained_w)

    net.video_train(style_name=style_name,
                    epochs=epochs,
                    style_weight=style_weight,
                    content_weight=content_weight,
                    temporal_weight=temporal_weight)


@video_st.command()
@click.argument('video-path')
@click.argument('style-name')
@click.option('-o', '--out-dir', default='results/',
              help='The results directory where the converted style will be saved')
@click.option('--fps', default=24.0, help='The FPS that will be used when saving the transformed video')
def convert_video(video_path, style_name, out_dir, fps):
    """
    Converts the video at `video-path` using the network pretrained with `style-name`
    and saves the resulting transformed video in `out-dir`.

    A pretrained model should exist in `data/models/` for the specified `style-name`.

    The files in `data/models/` contain the style names in their file names. For example,
    `video_st_starry_night_epoch3.pth` was trained with the style `starry_night`
    """

    sty = network.VideoTransformNet(torch.rand([3, 255, 255]))
    sty.process_video(video_path=video_path,
                      style_name=style_name,
                      out_dir=out_dir,
                      fps=fps)

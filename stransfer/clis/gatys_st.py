import os

import click

from stransfer import c_logging, constants, img_utils, network

LOGGER = c_logging.get_logger()


@click.command()
@click.argument('style-image-path')
@click.argument('content-image-path')
@click.option('-n', '--out-name', default='gatys_converted.png', help='The name of the result file (transformed image)')
@click.option('-s', '--steps', default=300, help="How many iterations should the optimization go through.")
@click.option('-cw', '--content-weight', default=1,
              help="The weight we will assign to the content loss during the optimization")
@click.option('-sw', '--style-weight', default=1_000_000,
              help="The weight we will assign to the style loss during the optimization")
def cli(style_image_path, content_image_path, out_name, content_image, steps, content_weight, style_weight):
    """
    Run the original Gatys style transfer (slow). Both `style-image` and
    `content-image` should be the paths to the image we want to take the content from
    and the one we want to take the style from (respectively).
    """

    style_image_path = os.path.join(constants.PROJECT_ROOT_PATH, style_image_path)
    content_image_path = os.path.join(constants.PROJECT_ROOT_PATH, content_image_path)

    style_image = img_utils.image_loader(style_image_path)
    content_image = img_utils.image_loader(content_image_path)

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

    LOGGER.info('Done! Transformed image has been saved to: %s', out_file)

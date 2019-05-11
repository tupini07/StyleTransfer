import click
from stransfer import img_utils, network, constants
from tqdm import tqdm


import datetime

import torch


def run_static_style_transfer(style_path, content_path, steps=220, dir=""):

    # TODO remove these 2
    content_path = "data/dancing.jpg"
    style_path = "data/picasso.jpg"

    # TODO move to parameters
    style_weight = 1000000
    content_weight = 1

    style_image = img_utils.image_loader(style_path)
    content_image = img_utils.image_loader(content_path)

    # clamp content image before creating network
    content_image.data.clamp_(0, 1)

    st_net = network.StyleNetwork(style_image, content_image)

    # start from content image
    input_image = content_image.clone()

    # or start from random image
    # input_image = torch.randn(content_image.data.size(), device=constants.DEVICE)

    optimizer = network.get_content_optimizer(input_image)

    for step in tqdm(range(steps)):

        def closure():
            # clamp content image in place each step
            input_image.data.clamp_(0, 1)

            optimizer.zero_grad()

            # pass content image through net
            st_net(input_image)

            # get losses
            style_loss = st_net.get_total_current_style_loss()
            content_loss = st_net.get_total_current_content_loss()

            style_loss *= style_weight
            content_loss *= content_weight

            total_loss = style_loss + content_loss
            total_loss.backward()

            return total_loss

        optimizer.step(closure)


    # TODO check if this is necessary
    input_image.data.clamp_(0, 1)

    return input_image


def analyze_static_style_transfer(style_path, content_path, steps=220, dir="", optimizer=torch.optim.LBFGS):

    # TODO remove these 2
    content_path = "data/dancing.jpg"
    style_path = "data/picasso.jpg"

    # TODO move to parameters
    style_weight = 1000000
    content_weight = 1

    style_image = img_utils.image_loader(style_path)
    content_image = img_utils.image_loader(content_path)

    # clamp content image before creating network
    content_image.data.clamp_(0, 1)

    st_net = network.StyleNetwork(style_image, content_image)

    # start from content image
    input_image = content_image.clone()

    # or start from random image
    # input_image = torch.randn(content_image.data.size(), device=constants.DEVICE)

    if optimizer == torch.optim.SGD:
        optimizer = optimizer([input_image.requires_grad_()], lr=0.01, momentum=0.9)
    else:
        optimizer = optimizer([input_image.requires_grad_()])


    ###
    starting_time = datetime.datetime.now()
    import sys
    import os
    import shutil

    dpath = "results/" + dir + "/"
    if os.path.exists(dpath):
        shutil.rmtree(dpath)

    if not os.path.exists(dpath):
        os.makedirs(dpath)
        with open(dpath + "results.txt", "w+") as ff:
            ff.write("step_num,elapsed_seconds,loss\n")

    ###

    for step in tqdm(range(steps)):

        ###
        metric_losses = []
        ###

        def closure():
            # clamp content image in place each step
            input_image.data.clamp_(0, 1)

            optimizer.zero_grad()

            # pass content image through net
            st_net(input_image)

            # get losses
            style_loss = st_net.get_total_current_style_loss()
            content_loss = st_net.get_total_current_content_loss()

            style_loss *= style_weight
            content_loss *= content_weight

            total_loss = style_loss + content_loss
            total_loss.backward()

            ###
            metric_losses.append(total_loss)
            ###

            return total_loss

        optimizer.step(closure)

        ########
        best_loss = min(metric_losses)
        print(best_loss)
        
        with open(dpath + "results.txt", "a") as ff:
            ff.write(
                f"{step},{(datetime.datetime.now() - starting_time).total_seconds()},{best_loss}\n")

        img_utils.imshow(input_image.data.clamp(0, 1), dpath + f"{step}.bmp")
        ###########

    # TODO check if this is necessary
    input_image.data.clamp_(0, 1)

    return input_image


@click.command()
@click.argument('style-image')
@click.argument('content-image')
def cli(style_image, content_image):
    print(123123)


if __name__ == "__main__":
    # cli(**{}) # suppress warning

    # run_static_style_transfer(1, 1, 500, "Adam")
    analyze_static_style_transfer(1, 1, 500, "Adadelta", torch.optim.Adadelta)
    analyze_static_style_transfer(1, 1, 500, "Adamax", torch.optim.Adamax)
    analyze_static_style_transfer(1, 1, 500, "SGD", torch.optim.SGD)


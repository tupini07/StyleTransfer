import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from stransfer import constants

_VGG = torchvision.models.vgg19(pretrained=True)
_VGG = (_VGG

        # we only want the `features` part of VGG19 (see print(_vgg) for structure)
        .features

        .to(constants.DEVICE)
        .eval())  # by default we set the network to evaluation mode


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super().__init__()

        # TODO Check that detach is actually necesary here
        # Here the target is the conv layer which we're taking as reference
        # as the style source
        self.target = self.gram_matrix(target_feature).detach()

    def gram_matrix(self, input):
        # TODO: check that gram matrix implementation is actually correct

        # The size would be [batch_size, depth, height, width]
        # in our style transfer application `bs` should always be 1
        bs, depth, height, width = input.size()

        features = input.view(bs * depth, height * width)

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(bs * depth * height * width)

    def forward(self, input):
        G = self.gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


class ContentLoss(nn.Module):
    def __init__(self, target,):
        super().__init__()

        # Here the target is the conv layer which we're taking as reference
        # as the content source
        self.target = target.detach()

    def forward(self, input):
        # Content loss is just the per pixel distance between an input and
        # the target
        self.loss = F.mse_loss(input, self.target)
        return input


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(
            mean).view(-1, 1, 1).type(torch.FloatTensor).to(constants.DEVICE)
        self.std = torch.tensor(
            std).view(-1, 1, 1).type(torch.FloatTensor).to(constants.DEVICE)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


class StyleNetwork(nn.Sequential):

    content_layers = [  # from where image content will be taken
        #  'Conv2d_1',
        #  'Conv2d_2',
        #  'Conv2d_3',
        'Conv2d_4',
        #  'Conv2d_5',
    ]

    style_layers = [  # were the image style will be taken from
        'Conv2d_1',
        'Conv2d_2',
        'Conv2d_3',
        'Conv2d_4',
        'Conv2d_5',
    ]

    def __init__(self, style_image, content_image, *args):
        super().__init__(*args)

        self._content_loss_nodes = []
        self._style_loss_nodes = []

        vgg = copy.deepcopy(_VGG)

        self.add_module('normalization', Normalization(
            # normalize image using ImageNet mean and std
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]).to(constants.DEVICE))

        i = 0
        for layer in vgg:
            if isinstance(layer, nn.Conv2d):
                i += 1

            if isinstance(layer, nn.ReLU):
                layer.inplace = False

            # one of {'Conv2d', 'MaxPool2d', 'ReLU'}
            layer_name = type(layer).__name__ + f"_{i}"
            self.add_module(layer_name, layer)

            if layer_name in self.content_layers:
                # TODO: try not detaching
                # we detach since we don't want to calculate gradients based on this
                layer_output = self(content_image).detach()
                content_loss = ContentLoss(layer_output)
                self.add_module(f"{layer_name}_content_loss", content_loss)
                self._content_loss_nodes.append(content_loss)

            if layer_name in self.style_layers:
                # TODO: try not detaching
                layer_output = self(style_image).detach()
                style_loss = StyleLoss(layer_output)
                self.add_module(f"{layer_name}_style_loss", style_loss)
                self._style_loss_nodes.append(style_loss)

    def get_total_current_content_loss(self):
        """
        Returns the sum of all the `loss` present in all
        *content* nodes
        """

        return torch.stack([x.loss for x in self._content_loss_nodes]).sum()

    def get_total_current_style_loss(self):
        """
        Returns the sum of all the `loss` present in all
        *style* nodes
        """

        return torch.stack([x.loss for x in self._style_loss_nodes]).sum()


# based on the residual block implementation from:
# https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/deep_residual_network/main.py
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, downsample=None,
                 padding=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        # architecure of the rasidual block was taken from
        # Gross and Wilber (Training and investigating residual nets)
        # http://torch.ch/blog/2016/02/04/resnets.html
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.bn2(out)
        return out


class ScaledTanh(nn.Module):
    def __init__(self, min_=0, max_=255):
        super().__init__()
        self.tanh = nn.Tanh()
        self.min = min_
        self.max = max_

    def forward(self, x):
        x = self.tanh(x)

        x = ((x+1)/(2)) * (self.max - self.min)

        return x


class ImageTransformNet(nn.Sequential):
    def __init__(self):
        super().__init__(

            # TODO in paper the ouptut of this should be
            # 32x256x256, but as it currently is the output is
            # 32x250x250
            nn.Conv2d(in_channels=3,
                      out_channels=32,
                      kernel_size=9,
                      stride=1,
                      padding=4),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=3,
                      stride=2,
                      padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                      stride=2,
                      padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),

            ResidualBlock(in_channels=128,
                          out_channels=128,
                          kernel_size=3),

            ResidualBlock(in_channels=128,
                          out_channels=128,
                          kernel_size=3),

            ResidualBlock(in_channels=128,
                          out_channels=128,
                          kernel_size=3),

            ResidualBlock(in_channels=128,
                          out_channels=128,
                          kernel_size=3),

            ResidualBlock(in_channels=128,
                          out_channels=128,
                          kernel_size=3),

            nn.ConvTranspose2d(in_channels=128,
                               out_channels=64,
                               kernel_size=3,
                               stride=2,
                               padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),


            nn.ConvTranspose2d(in_channels=64,
                               out_channels=32,
                               kernel_size=3,
                               stride=2,
                               padding=0),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),

            nn.ZeroPad2d((1, 0, 1, 0)),

            # TODO currently a bit hackish since we use
            # padding to have correct shape. Check if this is needed
            nn.Conv2d(in_channels=32,
                      out_channels=3,
                      kernel_size=9,
                      stride=1,
                      padding=4),

            # TODO check if batch norm is needed
            # for the last layer
            nn.BatchNorm2d(num_features=3),
            ScaledTanh(min_=0, max_=255)
        )


def get_content_optimizer(input_img):
    # we want to apply the gradient to the content image, so we
    # need to mark it as such

    # TODO find out which is the best optimizer in this case
    # optimizer = optim.LBFGS([input_img.requires_grad_()])
    optimizer = optim.Adam([input_img.requires_grad_()])
    # optimizer = optim.Adadelta([input_img.requires_grad_()])
    # optimizer = optim.Adamax([input_img.requires_grad_()])
    # optimizer = optim.SGD([input_img.requires_grad_()])

    return optimizer

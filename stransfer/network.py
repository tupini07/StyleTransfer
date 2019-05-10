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

    def _get_total_current_content_loss(self):
        """
        Returns the sum of all the `loss` present in all
        *content* nodes
        """

        return torch.stack([x.loss for x in self._content_loss_nodes]).sum()

    def _get_total_current_style_loss(self):
        """
        Returns the sum of all the `loss` present in all
        *style* nodes
        """

        return torch.stack([x.loss for x in self._style_loss_nodes]).sum()

    def st_forward(self, style_image, content_image):
        """
        Custom forward method. Applies the standard
        Sequential model forward to calculate the losses
        then return the total losses for content loss and style loss
        """

        # first pass content image and get total loss
        self(content_image)
        total_content_loss = self._get_total_current_content_loss()
        total_style_loss = self._get_total_current_style_loss()

        return total_style_loss, total_content_loss


def get_content_optimizer(content_img):
    # we want to apply the gradient to the content image, so we
    # need to mark it as such

    # TODO find out which is the best optimizer in this case
    # optimizer = optim.LBFGS([content_img.requires_grad_()])
    optimizer = optim.Adam([content_img.requires_grad_()])

    return optimizer

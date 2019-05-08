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

    def __init__(self, *args):
        super().__init__(*args)

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

        # remove all layers after old the last content/style loss
        # first find out which is the first relevant index
        last_index = 0
        for i, (name, _) in enumerate(self.named_children()):
            if any(name in l for l in [self.style_layers,
                                       self.content_layers]):
                last_index = i

        # then just remove the extra layers by
        # resetting network "modules"
        self._modules = OrderedDict(
            list(self._modules.items())[:(last_index + 1)])

    def forward(self, style_image, content_image):
        temp_net = nn.Sequential()

        style_losses = []
        content_losses = []

        # if the current layer is either one of the style or
        # content layers we specified above then we want to calculate
        # the style/content loss appropriately.
        # see https://forums.fast.ai/t/pytorch-best-way-to-get-at-intermediate-layers-in-vgg-and-resnet/5707/2
        for name, layer in self.named_children():
            temp_net.add_module(name, layer)

            if name in self.content_layers:
                # TODO: try not detaching
                # we detach since we don't want to calculate gradients based on this
                layer_output = temp_net(content_image).detach()
                content_loss = ContentLoss(layer_output)
                temp_net.add_module(f"{name}_content_loss", content_loss)
                content_losses.append(content_loss)

            if name in self.style_layers:
                layer_output = temp_net(style_image).detach()
                style_loss = StyleLoss(layer_output)
                temp_net.add_module(f"{name}_style_loss", style_loss)
                style_losses.append(style_loss)

        return style_losses, content_losses

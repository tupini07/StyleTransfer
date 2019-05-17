import copy
import logging
from collections import OrderedDict


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from tqdm import tqdm

from stransfer import c_logging, constants, img_utils, dataset
from tensorboardX import SummaryWriter

LOGGER = c_logging.get_logger()
TB_WRITER = SummaryWriter('runs/optimize-after-step_feature-style-loss')


_VGG = torchvision.models.vgg19(pretrained=True)
_VGG = (_VGG

        # we only want the `features` part of VGG19 (see print(_vgg) for structure)
        .features

        .to(constants.DEVICE)
        .eval())  # by default we set the network to evaluation mode


class StyleLoss(nn.Module):
    def __init__(self, target):
        super().__init__()

        self.set_target(target)

    def gram_matrix(self, input):
        # TODO: check that gram matrix implementation is actually correct
        # when batch size > 1 the sizes are different to the ones of the target

        # The size would be [batch_size, depth, height, width]
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

    def set_target(self, target):
        # TODO Check that detach is actually necesary here
        # Here the target is the conv layer which we're taking as reference
        # as the style source
        self.target = self.gram_matrix(target).detach()


class ContentLoss(nn.Module):
    def __init__(self, target):
        super().__init__()

        # Here the target is the conv layer which we're taking as reference
        # as the content source
        self.set_target(target)

    def set_target(self, target):
        self.target = target.detach()

    def forward(self, input):
        # Content loss is just the per pixel distance between an input and
        # the target
        self.loss = F.mse_loss(input, self.target)
        return input


class FeatureReconstructionLoss (nn.Module):
    def __init__(self, target):
        super().__init__()

        # Here the target is the conv layer which we're taking as reference
        # as the content source
        self.set_target(target)

    def set_target(self, target):
        self.target = target.detach()

    def forward(self, input):
        # Content loss is just the per pixel distance between an input and
        # the target
        l2_norm = F.mse_loss(input, self.target)
        l2_squared = l2_norm.pow(2)

        # The size would be [batch_size, depth, height, width]
        bs, depth, height, width = input.size()

        self.loss = l2_squared.div(bs * depth * height * width)

        return input


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = (torch.tensor(mean)
                     .view(-1, 1, 1)
                     .type(torch.FloatTensor)
                     .to(constants.DEVICE))
        self.std = (torch.tensor(std)
                    .view(-1, 1, 1)
                    .type(torch.FloatTensor)
                    .to(constants.DEVICE))

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


class StyleNetwork(nn.Module):
    # TODO check if these layers are ok
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

    feature_loss_layers = [
        'ReLU_4',
    ]

    def __init__(self, style_image, content_image=torch.zeros([1, 3, 256, 256])):
        super().__init__()

        self.content_losses = []
        self.style_losses = []
        self.feature_losses = []

        vgg = copy.deepcopy(_VGG)

        self.net_pieces = [
            nn.Sequential(
                Normalization(
                    # normalize image using ImageNet mean and std
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]).to(constants.DEVICE)
            )
        ]

        loss_added = False
        current_piece = 0
        i = 0
        for layer in vgg:
            if isinstance(layer, nn.Conv2d):
                i += 1

            if isinstance(layer, nn.ReLU):
                layer.inplace = False

            # one of {'Conv2d', 'MaxPool2d', 'ReLU'}
            layer_name = type(layer).__name__ + f"_{i}"
            self.net_pieces[current_piece].add_module(layer_name, layer)

            if layer_name in self.content_layers:
                layer_output = self.run_through_pieces(content_image)
                content_loss = ContentLoss(layer_output)
                self.content_losses.append([content_loss, current_piece])
                loss_added = True

            if layer_name in self.style_layers:
                layer_output = self.run_through_pieces(style_image)
                style_loss = StyleLoss(layer_output)
                self.style_losses.append([style_loss, current_piece])
                loss_added = True

            if layer_name in self.feature_loss_layers:
                layer_output = self.run_through_pieces(content_image)
                feature_loss = FeatureReconstructionLoss(layer_output)
                self.feature_losses.append([feature_loss, current_piece])
                loss_added = True

            if loss_added:
                self.net_pieces.append(current_piece)
                current_piece += 1
                self.net_pieces[current_piece] = nn.Sequential()
                loss_added = False

    def run_through_pieces(self, input_g, until=-1):
        x = input_g

        # if no array of pieces is provided then we just run the input
        # through all pieces in the network
        if until == -1:
            pieces = self.net_pieces

        else:
            pieces = self.net_pieces[:until + 1]

        # finally run the input image through the pieces
        for piece in pieces:
            x = piece(x)

        return x

    def get_total_current_content_loss(self):
        """
        Returns the sum of all the `loss` present in all
        *content* nodes
        """

        return torch.stack([x[0].loss for x in self.content_losses]).sum()

    def get_total_current_feature_loss(self):
        """
        Returns the sum of all the `loss` present in all
        *content* nodes
        """

        return torch.stack([x[0].loss for x in self.feature_losses]).sum()

    def get_total_current_style_loss(self):
        """
        Returns the sum of all the `loss` present in all
        *style* nodes
        """

        return torch.stack([x[0].loss for x in self.style_losses]).sum()

    def forward(self, input_image, content_image=None, style_image=None):

        # first set content, feature, and style targets
        for (loss, piece_idx) in self.content_losses + self.feature_losses:
            if content_image is not None:
                loss.set_target(
                    self.run_through_pieces(content_image, piece_idx)
                )

            loss(
                self.run_through_pieces(input_image, piece_idx)
            )

        # if we provide a style image to override the one
        # provided in the __init__
        for (loss, piece_idx) in self.style_losses:
            if style_image is not None:
                loss.set_target(
                    self.run_through_pieces(content_image, piece_idx)
                )

            loss(
                self.run_through_pieces(input_image, piece_idx)
            )

        # we don't need the network output, so there is no need to run
        # the input through the whole network

    def get_content_optimizer(self, input_img, optt=optim.Adam):
        # we want to apply the gradient to the content image, so we
        # need to mark it as such

        optimizer = optt([input_img.requires_grad_()])

        return optimizer

    def train(self, style_image, content_image, steps=220):
        """
        To train on only one content and style images
        """

        assert isinstance(
            style_image, torch.Tensor), 'Images need to be already loaded'
        assert isinstance(
            content_image, torch.Tensor), 'Images need to be already loaded'

        # TODO move to parameters
        style_weight = 1000000
        content_weight = 1

        # clamp content image before creating network
        content_image.data.clamp_(0, 1)

        # start from content image
        input_image = content_image.clone()

        # start from content image
        input_image = content_image.clone()

        # or start from random image
        # input_image = torch.randn(content_image.data.size(), device=constants.DEVICE)

        optimizer = self.get_content_optimizer(input_image)

        for step in tqdm(range(steps)):

            def closure():
                # clamp content image in place each step
                input_image.data.clamp_(0, 1)

                optimizer.zero_grad()

                # pass content image through net
                self(input_image, content_image)

                # get losses
                style_loss = self.get_total_current_style_loss()
                content_loss = self.get_total_current_content_loss()

                style_loss *= style_weight
                content_loss *= content_weight

                total_loss = style_loss + content_loss
                total_loss.backward()

                print(total_loss)
                return total_loss

            optimizer.step(closure)

        # TODO check if this is necessary
        input_image.data.clamp_(0, 1)

        return input_image


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
    def __init__(self, style_image):
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

        # finally, set the style image which
        # transform network represents
        assert isinstance(
            style_image, torch.Tensor), 'Style image need to be already loaded'
        self.style_image = style_image

    def get_optimizer(self, optimizer=optim.Adam):
        params = self.parameters()

        return optimizer(params)

    def train(self):
        # TODO: parametrize
        epochs = 50
        steps = 30
        style_weight = 1000000
        feature_weight = 1

        loss_network = StyleNetwork(self.style_image,
                                    torch.rand([1, 3, 256, 256]).to(constants.DEVICE))

        optimizer = self.get_optimizer()
        iteration = 0

        test_loader, train_loader = dataset.get_coco_loader()
        for epoch in range(epochs):

            LOGGER.info('Starting epoch %d', epoch)

            for batch in train_loader:

                for image in batch:

                    assert isinstance(
                        image, torch.Tensor), 'Images need to be already loaded'

                    for step in tqdm(range(steps)):
                        optimizer.zero_grad()

                        tansformed_image = self(image)  # transfor the image
                        # evaluate how good the transformation is
                        loss_network(tansformed_image)

                        # Get losses
                        style_loss = loss_network.get_total_current_style_loss()
                        feature_loss = loss_network.get_total_current_feature_loss()

                        style_loss *= style_weight
                        feature_loss *= feature_weight

                        total_loss = style_loss + feature_loss

                        total_loss.backward()

                        # TODO currently the optimization step is done once for each step
                        # of the image. Need to see if this is good or if we should do
                        # the optimization step once every batch and remove the loop on the
                        # steps
                        TB_WRITER.add_scalar(
                            'data/fst_train_loss', total_loss, iteration)

                        if iteration + 1 % 9999:
                            average_test_loss = self.test(test_loader)
                            TB_WRITER.add_scalar(
                                'data/fst_test_loss', average_test_loss, iteration)
                            TB_WRITER.add_image('data/fst_images',
                                                torch.cat([tansformed_image.squeeze(),
                                                           image.squeeze()],
                                                          dim=2),
                                                iteration)
                        LOGGER.info('Loss: %.8f', total_loss)
                        iteration += 1

                        # after processing the batch, run the gradient update
                        optimizer.step()

    def test(self, test_loader):
        # TODO: parametrize
        epochs = 50
        steps = 30
        style_weight = 1000000
        feature_weight = 1

        loss_network = StyleNetwork(self.style_image,
                                    torch.rand([1, 3, 256, 256]).to(constants.DEVICE))

        total_test_loss = []
        for test_batch in test_loader:
            for test_img in test_batch:
                tansformed_image = self(test_img)
                loss_network(tansformed_image)

                style_loss = style_weight * loss_network.get_total_current_style_loss()
                feature_loss = feature_weight * loss_network.get_total_current_feature_loss()

                total_test_loss.append(style_loss + feature_loss)

        return torch.mean(torch.stack(total_test_loss))

    def evaluate(self, image):
        raise NotImplementedError()

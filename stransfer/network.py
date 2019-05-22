import copy
import logging
import shutil
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from tensorboardX import SummaryWriter
from tqdm import tqdm

from stransfer import c_logging, constants, dataset, img_utils

LOGGER = c_logging.get_logger()

TENSORBOARD_PATH = 'runs/optimize-after-real-batch_feature-style-loss'

shutil.rmtree(TENSORBOARD_PATH, ignore_errors=True)

TB_WRITER = SummaryWriter(TENSORBOARD_PATH)
TB_WRITER.add_text('note', ('For this run, the loss is calculated for a real batch'
                            'and then the optimization step '
                            'is made. Images in batch are only seen once '
                            '(only one "step" is made for each image. '
                            'The new feature+style loss is used.'), 0)

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
        # The size would be [batch_size, depth, height, width]
        bs, depth, height, width = input.size()

        features = input.view(bs,  depth, height * width)
        features_t = features.transpose(1, 2)

        # compute the gram product
        G = torch.bmm(features, features_t)

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(depth * height * width)

    def forward(self, input):

        G = self.gram_matrix(input)
        self.loss = F.mse_loss(G,
                               # correct the fact that we only have one
                               # style image for the whole batch
                               self.target.expand_as(G))

        return input

    def set_target(self, target):
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

    def get_total_current_content_loss(self, weight=1):
        """
        Returns the sum of all the `loss` present in all
        *content* nodes
        """

        return weight * torch.stack([x[0].loss for x in self.content_losses]).sum()

    def get_total_current_feature_loss(self, weight=1):
        """
        Returns the sum of all the `loss` present in all
        *content* nodes
        """

        return weight * torch.stack([x[0].loss for x in self.feature_losses]).sum()

    def get_total_current_style_loss(self, weight=1):
        """
        Returns the sum of all the `loss` present in all
        *style* nodes
        """

        return weight * torch.stack([x[0].loss for x in self.style_losses]).sum()

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
                style_loss = self.get_total_current_style_loss(
                    weight=style_weight)
                content_loss = self.get_total_current_content_loss(
                    weight=content_weight)

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
                 kernel_size=3, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride)
        self.insn1 = nn.InstanceNorm2d(out_channels, affine=True)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride)

        self.insn2 = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        # architecure of the residual block was taken from
        # Gross and Wilber (Training and investigating residual nets)
        # http://torch.ch/blog/2016/02/04/resnets.html
        residual = x

        out = self.conv1(x)
        out = self.insn1(out)

        out = self.relu(out)
        out = self.conv2(out)

        out += residual

        out = self.insn2(out)

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
    def __init__(self, style_image, batch_size=1):
        super().__init__(

            # * Initial convolutional layers
            # First Conv
            nn.Conv2d(in_channels=3,
                      out_channels=32,
                      kernel_size=9,
                      stride=1,
                      padding=4,
                      padding_mode='reflection'),
            nn.InstanceNorm2d(num_features=32, affine=True),
            nn.ReLU(),

            # Second Conv
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=3,
                      stride=2,
                      padding=1),
            nn.InstanceNorm2d(num_features=64, affine=True),
            nn.ReLU(),

            # Third Conv
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                      stride=2,
                      padding=1,
                      padding_mode='reflection'),
            nn.InstanceNorm2d(num_features=128, affine=True),
            nn.ReLU(),

            # * Residual blocks
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

            # * Deconvolution layers
            # ? According to https://distill.pub/2016/deconv-checkerboard/
            # ? an upsampling layer followed by a convolution layer
            # ? yields better results
            # First 'deconvolution' layer
            nn.Upsample(mode='nearest',
                        scale_factor=2),
            nn.Conv2d(in_channels=128,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      padding_mode='reflection'),
            nn.InstanceNorm2d(num_features=64, affine=True),
            nn.ReLU(),

            # Second 'deconvolution' layer
            nn.Upsample(mode='nearest',
                        scale_factor=2),
            nn.Conv2d(in_channels=64,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      padding_mode='reflection'),
            nn.InstanceNorm2d(num_features=32, affine=True),
            nn.ReLU(),

            # * Final convolutional layer
            nn.Conv2d(in_channels=32,
                      out_channels=3,
                      kernel_size=9,
                      stride=1),
        )

        # finally, set the style image which
        # transform network represents
        assert isinstance(
            style_image, torch.Tensor), 'Style image need to be already loaded'

        # we need to ensure that we have enough style images for the batch
        # NOTE: this could also be accomplished by letting pytorch broadcast
        self.style_image = style_image
        self.batch_size = batch_size

    def get_total_variation_regularization_loss(self, transformed_image: torch.Tensor,
                                                regularization_factor=1e-6) -> torch.Tensor:
        # ? see: https://en.wikipedia.org/wiki/Total_variation_denoising#2D_signal_images
        return regularization_factor * (
            torch.sum(torch.abs(
                transformed_image[:, :, :, :-1] - transformed_image[:, :, :, 1:])
            ) +
            torch.sum(
                torch.abs(
                    transformed_image[:, :, :-1, :] - transformed_image[:, :, 1:, :])
            ))

    def get_optimizer(self, optimizer=optim.Adam):
        params = self.parameters()

        return optimizer(params)

    def train(self):
        # TODO: parametrize
        epochs = 50
        steps = 30
        style_weight = 100_000
        feature_weight = content_weight = 1

        # TODO: try adding the following so that grads are not computed
        # with torch.no_grad():
        loss_network = StyleNetwork(self.style_image,
                                    torch.rand([1, 3, 256, 256]).to(
                                        constants.DEVICE)).eval()

        optimizer = self.get_optimizer(optimizer=optim.Adam)
        # optimizer = self.get_optimizer(optimizer=optim.LBFGS)

        LOGGER.info('Training network with "%s" optimizer', type(optimizer))

        iteration = 0

        test_loader, train_loader = dataset.get_coco_loader(test_split=0.10,
                                                            test_limit=20,
                                                            batch_size=self.batch_size)
        for epoch in range(epochs):

            LOGGER.info('Starting epoch %d', epoch)

            for batch in train_loader:

                for image in batch:

                    def closure():
                        optimizer.zero_grad()
                        # transformed_image = torch.clamp(
                        #     self(image),  # transfor the image
                        #     min=0,
                        #     max=255 
                        # )

                        transformed_image = self(image)
                        img_utils.imshow(
                            torch.cat([
                                transformed_image.squeeze(),
                                image.squeeze()],
                                dim=2)
                        )

                        # evaluate how good the transformation is
                        loss_network(transformed_image,
                                     content_image=image)

                        # Get losses
                        style_loss = loss_network.get_total_current_style_loss(
                            weight=style_weight
                        )
                        feature_loss = loss_network.get_total_current_feature_loss(
                            weight=feature_weight
                        )
                        content_loss = loss_network.get_total_current_content_loss(
                            weight=content_weight
                        )
                        regularization_loss = self.get_total_variation_regularization_loss(
                            transformed_image
                        )

                        # total_loss = feature_loss + style_loss
                        # total_loss = style_loss + content_loss
                        # total_loss = style_loss
                        # total_loss = feature_loss
                        total_loss = style_loss + content_loss + regularization_loss

                        total_loss.backward()

                        LOGGER.debug('Max of each channel: %s', [
                                     x.max().item() for x in transformed_image.squeeze()])
                        LOGGER.debug('Min of each channel: %s', [
                                     x.min().item() for x in transformed_image.squeeze()])
                        LOGGER.debug('Sum of each channel: %s', [
                                     x.sum().item() for x in transformed_image.squeeze()])
                        LOGGER.debug('Closure loss: %.8f', total_loss)

                        return style_loss + content_loss

                    total_loss = closure()

                    TB_WRITER.add_scalar(
                        'data/fst_train_loss',
                        total_loss,
                        iteration)

                    if iteration % 10 == 0:
                        LOGGER.info('Batch Loss: %.8f', total_loss)

                    if iteration % 150 == 0:
                        average_test_loss = self.test(
                            test_loader, loss_network)

                        TB_WRITER.add_scalar(
                            'data/fst_test_loss', average_test_loss, iteration)

                    if iteration % 50 == 0:

                        transformed_image = torch.clamp(
                            self(image),  # transfor the image
                            min=0,
                            max=255
                        )

                        TB_WRITER.add_image('data/fst_images',
                                            torch.cat([
                                                transformed_image.squeeze(),
                                                batch[0].squeeze()],
                                                dim=2),
                                            iteration)
                    iteration += 1

                    # after processing the batch, run the gradient update
                    optimizer.step(closure)

    def test(self, test_loader, loss_network):
        # TODO: parametrize
        epochs = 50
        steps = 30
        style_weight = 1000000
        feature_weight = 1

        total_test_loss = []
        for test_batch in test_loader:

            transformed_image = torch.clamp(
                self(test_batch.squeeze(1)),  # transfor the image
                min=0,
                max=255
            )

            loss_network(transformed_image, content_image=test_batch.squeeze(1))

            style_loss = style_weight * loss_network.get_total_current_style_loss()
            feature_loss = feature_weight * loss_network.get_total_current_feature_loss()

            total_test_loss.append((style_loss + feature_loss).item())

        average_test_loss = torch.mean(torch.Tensor(total_test_loss))
        LOGGER.info('Average test loss: %.8f', average_test_loss)

        return average_test_loss

    def evaluate(self, image):
        raise NotImplementedError()

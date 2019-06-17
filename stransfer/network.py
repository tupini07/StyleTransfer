import copy
import os
import shutil

import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from PIL import Image
from tensorboardX import SummaryWriter
from tqdm import tqdm

from stransfer import c_logging, constants, dataset, img_utils

LOGGER = c_logging.get_logger()


def get_tensorboard_writer(path):
    shutil.rmtree(path, ignore_errors=True)
    return SummaryWriter(path)


_VGG = torchvision.models.vgg19(pretrained=True)
_VGG = (_VGG

        # we only want the `features` part of VGG19 (see print(_vgg) for structure)
        .features

        .to(constants.DEVICE)
        .eval())  # by default we set the network to evaluation mode


def adaptive_torch_load(weights_path):
    if constants.DEVICE.type == "cuda":
        return torch.load(weights_path, map_location='gpu')
    else:
        return torch.load(weights_path, map_location='cpu')


def _load_latest_model_weigths(model_name: str,
                               style_name: str,
                               models_path='data/models/'):
    """
    :return: the weights file for the latest epoch corresponding
    to the model and style specified.
    :param models_path:
    :return:
    """

    models_path = os.path.join(constants.PROJECT_ROOT_PATH, models_path)

    try:
        latest_weight_name = sorted([x for x in os.listdir(models_path)
                                     if x.startswith(model_name) and
                                     style_name in x])[-1]
    except IndexError:
        LOGGER.critical('There are no weights for the specified model name (%s) '
                        'and style (%s). In the specified path: %s',
                        model_name, style_name, models_path)

        raise AssertionError('There are no weights for the specified '
                             'model name and style.')

    return adaptive_torch_load(models_path + latest_weight_name)


class StyleLoss(nn.Module):
    def __init__(self, target):
        super().__init__()

        self.set_target(target)

    def gram_matrix(self, input):
        # The size would be [batch_size, depth, height, width]
        bs, depth, height, width = input.size()

        features = input.view(bs, depth, height * width)
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


class FeatureReconstructionLoss(nn.Module):
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

        # TODO: It is safer to do a deepcopy. But in the meantime
        # we don't want to occupy extra memory
        vgg = copy.deepcopy(_VGG)
        # vgg = _VGG

        self.net_pieces = [
            nn.Sequential()
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

    def train_gatys(self, style_image, content_image, steps=550):
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

        # start from content image
        input_image = content_image.clone()

        # or start from random image
        # input_image = torch.randn(
        #     content_image.data.size(), device=constants.DEVICE)

        optimizer = self.get_content_optimizer(input_image, optt=optim.LBFGS)

        for step in tqdm(range(steps)):
            def closure():
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

                LOGGER.info('Loss: %s', total_loss)
                return total_loss

            optimizer.step(closure)

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
                               stride=stride,
                               padding=kernel_size // 2,
                               padding_mode='reflection')
        self.insn1 = nn.InstanceNorm2d(out_channels, affine=True)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=kernel_size // 2,
                               padding_mode='reflection')

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

        x = ((x + 1) / (2)) * (self.max - self.min)

        return x


class ImageTransformNet(nn.Sequential):
    def __init__(self, style_image, batch_size=4):
        super().__init__(

            # * Initial convolutional layers
            # First Conv
            nn.Conv2d(in_channels=3,
                      out_channels=32,
                      kernel_size=9,
                      stride=1,
                      padding=9 // 2,
                      padding_mode='reflection'),
            nn.InstanceNorm2d(num_features=32, affine=True),
            nn.ReLU(),

            # Second Conv
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=3,
                      stride=2,
                      padding=3 // 2,
                      padding_mode='reflection'),
            nn.InstanceNorm2d(num_features=64, affine=True),
            nn.ReLU(),

            # Third Conv
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                      stride=2,
                      padding=3 // 2,
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
                      padding=3 // 2,
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
                      padding=3 // 2,
                      padding_mode='reflection'),
            nn.InstanceNorm2d(num_features=32, affine=True),
            nn.ReLU(),

            # * Final convolutional layer
            nn.Conv2d(in_channels=32,
                      out_channels=3,
                      kernel_size=9,
                      stride=1,
                      padding=9 // 2,
                      padding_mode='reflection'),

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

    def static_train(self, style_name='nsp', epochs=50,
                     style_weight=100_000, content_weight=1):
        """
        Trains a fast style transfer network for style transfer on still images.
        """
        tb_writer = get_tensorboard_writer(
            f'runs/fast-image-style-transfer-still-image_{style_name}')

        # TODO: try adding the following so that grads are not computed
        # with torch.no_grad():
        loss_network = StyleNetwork(self.style_image,
                                    torch.rand([1, 3, 256, 256]).to(
                                        constants.DEVICE))

        optimizer = self.get_optimizer(optimizer=optim.Adam)

        LOGGER.info('Training network with "%s" optimizer', type(optimizer))

        iteration = 0

        test_loader, train_loader = dataset.get_coco_loader(test_split=0.10,
                                                            test_limit=20,
                                                            batch_size=self.batch_size)
        for epoch in range(epochs):

            LOGGER.info('Starting epoch %d', epoch)
            epoch_checkpoint_name = f'data/models/fast_st_{style_name}_epoch{epoch}.pth'

            # if the checkpoint file for this epoch exists then we
            # just load it and go over to the next epoch
            if os.path.isfile(epoch_checkpoint_name):
                self.load_state_dict(
                    adaptive_torch_load(epoch_checkpoint_name)
                )
                continue

            for batch in tqdm(train_loader):
                batch = batch.squeeze(1)

                def closure():
                    optimizer.zero_grad()

                    transformed_image = self(batch)

                    # evaluate how good the transformation is
                    loss_network(transformed_image,
                                 content_image=batch)

                    # Get losses
                    style_loss = loss_network.get_total_current_style_loss(
                        weight=style_weight
                    )

                    # Feature loss doesn't seem to be much better than the normal
                    # content loss
                    # feature_weight = 1
                    # feature_loss = loss_network.get_total_current_feature_loss(
                    #     weight=feature_weight
                    # )

                    content_loss = loss_network.get_total_current_content_loss(
                        weight=content_weight
                    )
                    regularization_loss = self.get_total_variation_regularization_loss(
                        transformed_image
                    )

                    # calculate loss
                    total_loss = style_loss + content_loss + regularization_loss

                    total_loss.backward()

                    LOGGER.debug('Max of each channel: %s', [
                        x.max().item() for x in transformed_image[0].squeeze()])
                    LOGGER.debug('Min of each channel: %s', [
                        x.min().item() for x in transformed_image[0].squeeze()])
                    LOGGER.debug('Sum of each channel: %s', [
                        x.sum().item() for x in transformed_image[0].squeeze()])
                    LOGGER.debug('Closure loss: %.8f', total_loss)

                    return total_loss

                if iteration % 20 == 0:
                    total_loss = closure()

                    tb_writer.add_scalar(
                        'data/fst_train_loss',
                        total_loss,
                        iteration)

                    LOGGER.info('Batch Loss: %.8f', total_loss)

                if iteration % 150 == 0:
                    average_test_loss = self.static_test(
                        test_loader, loss_network)

                    tb_writer.add_scalar(
                        'data/fst_test_loss', average_test_loss, iteration)

                if iteration % 50 == 0:
                    transformed_image = torch.clamp(
                        self(batch),  # transform the image
                        min=0,
                        max=255
                    )[0]

                    tb_writer.add_image('data/fst_images',
                                        img_utils.concat_images(
                                            transformed_image.squeeze(),
                                            batch[0].squeeze()),
                                        iteration)
                iteration += 1

                # after processing the batch, run the gradient update
                optimizer.step(closure)

            torch.save(
                self.state_dict(),
                epoch_checkpoint_name
            )

    def static_test(self, test_loader, loss_network, style_weight=100_000, feature_weight=1):
        """
        Tests the performance of a fast style transfer network on still images
        """

        total_test_loss = []
        for test_batch in test_loader:
            transformed_image = torch.clamp(
                self(test_batch.squeeze(1)),  # transfor the image
                min=0,
                max=255
            )

            loss_network(transformed_image,
                         content_image=test_batch.squeeze(1))

            style_loss = style_weight * loss_network.get_total_current_style_loss()
            feature_loss = feature_weight * loss_network.get_total_current_feature_loss()

            total_test_loss.append((style_loss + feature_loss).item())

        average_test_loss = torch.mean(torch.Tensor(total_test_loss))
        LOGGER.info('Average test loss: %.8f', average_test_loss)

        return average_test_loss


class VideoTransformNet(ImageTransformNet):

    def __init__(self, style_image, batch_size=4, fast_transfer_dict=None):
        super().__init__(style_image, batch_size)

        self[0] = nn.Conv2d(in_channels=6,
                            out_channels=32,
                            kernel_size=9,
                            stride=1,
                            padding=9 // 2,
                            padding_mode='reflection')

        # since this video net is exactly the same as the ImageTransformNet
        # 'fast transfer' then we can reuse it's backup weights, already trained
        # on imagenet, for this task.
        if fast_transfer_dict is not None:
            # if 'fast_transfer_dict' is a string then we take it to be the path
            # to a dump of the weights. So we load it.
            if isinstance(fast_transfer_dict, str):
                fast_transfer_dict = adaptive_torch_load(fast_transfer_dict)

            # but first we have to remove the 'weight' and 'bias' for the first layer,
            # since this is the one we will be replacing in this 'VideoTransformNet'
            del fast_transfer_dict['0.weight']
            del fast_transfer_dict['0.bias']

            # update video net state dict so that we ensure we have
            # the correct weight/biases for the first layer
            m_sd = self.state_dict().copy()
            m_sd.update(fast_transfer_dict)

            # finally load weights into network
            self.load_state_dict(m_sd)

            # flag to use when training, to know if we've loaded
            # external weights or not
            self.has_external_weights = True
        else:
            self.has_external_weights = False

    def get_temporal_loss(self, old_content, old_stylized,
                          current_content, current_stylized,
                          temporal_weight=1):

        # see https://github.com/tupini07/StyleTransfer/issues/5

        change_in_style = (current_stylized - old_stylized).norm()
        change_in_content = (current_content - old_content).norm()

        return (change_in_style / (change_in_content + 1)) * temporal_weight

    def video_train(self, style_name='nsp',
                    epochs=50, temporal_weight=0.8, style_weight=100_000, feature_weight=1, content_weight=1):

        tb_writer = get_tensorboard_writer(
            f'runs/video-style-transfer_{style_name}')

        VIDEO_FOLDER = f'video_samples_{style_name}/'
        shutil.rmtree(VIDEO_FOLDER, ignore_errors=True)
        os.makedirs(VIDEO_FOLDER, exist_ok=True)

        # TODO: try adding the following so that grads are not computed
        # with torch.no_grad():
        style_loss_network = StyleNetwork(self.style_image,
                                          torch.rand([1, 3, 256, 256]).to(
                                              constants.DEVICE))

        optimizer = self.get_optimizer(optimizer=optim.Adam)
        LOGGER.info('Training video network with "%s" optimizer',
                    type(optimizer))
        iteration = 0

        video_loader = dataset.VideoDataset(batch_size=self.batch_size)

        for epoch in range(epochs):
            # we freeze the 'external_weights' during the first epoch
            # if these are present
            if epoch == 0 and self.has_external_weights:
                LOGGER.info(
                    'Freezing weights imported from fast transfer network for the first epoch')
                for name, param in self.named_parameters():
                    # all layers which are not the first one
                    if not name.startswith('0.'):
                        param.requires_grad = False

            # next epoch we just 'unfreeze' all
            if epoch == 1 and self.has_external_weights:
                LOGGER.info('Unfreezing all weights')
                for param in self.parameters():
                    param.requires_grad = True

            epoch_checkpoint_name = f'data/models/video_st_{style_name}_epoch{epoch}.pth'

            # if the checkpoint file for this epoch exists then we
            # just load it and go over to the next epoch
            if os.path.isfile(epoch_checkpoint_name):
                self.load_state_dict(
                    adaptive_torch_load(epoch_checkpoint_name)
                )
                continue

            LOGGER.info('Starting epoch %d', epoch)

            for video_batch in video_loader:

                # of shape [content, stylized]
                old_images = None

                for batch in dataset.iterate_on_video_batches(video_batch):

                    # if we're in new epoch then previous frame is None
                    if old_images is None:
                        old_images = [batch, batch]

                    # ? make images available as simple vars
                    old_content_images = old_images[0]
                    old_styled_images = old_images[1]

                    batch_with_old_content = torch.cat(
                        [batch, old_styled_images],
                        dim=1)

                    def closure():
                        optimizer.zero_grad()

                        transformed_image = self(batch_with_old_content)

                        style_loss_network(transformed_image,
                                           content_image=batch)

                        style_loss = style_loss_network.get_total_current_style_loss(
                            weight=style_weight
                        )

                        feature_loss = style_loss_network.get_total_current_feature_loss(
                            weight=feature_weight
                        )
                        content_loss = style_loss_network.get_total_current_content_loss(
                            weight=content_weight
                        )
                        regularization_loss = self.get_total_variation_regularization_loss(
                            transformed_image
                        )

                        temporal_loss = self.get_temporal_loss(
                            old_content_images,
                            old_styled_images,
                            batch,
                            transformed_image,
                            temporal_weight=temporal_weight
                        )

                        # * agregate losses
                        total_loss = style_loss + content_loss + regularization_loss + temporal_loss

                        # * set old content and stylized versions
                        old_images[0] = batch.detach()
                        old_images[1] = transformed_image.detach()

                        total_loss.backward()

                        # * debug messages
                        LOGGER.debug('Max of each channel: %s', [
                            x.max().item() for x in transformed_image[0].squeeze()])
                        LOGGER.debug('Min of each channel: %s', [
                            x.min().item() for x in transformed_image[0].squeeze()])
                        LOGGER.debug('Sum of each channel: %s', [
                            x.sum().item() for x in transformed_image[0].squeeze()])
                        LOGGER.debug('Closure loss: %.8f', total_loss)

                        return total_loss

                    if iteration % 20 == 0:
                        total_loss = closure()

                        tb_writer.add_scalar(
                            'data/fst_train_loss',
                            total_loss,
                            iteration)
                        LOGGER.info('Epoch: %d\tBatch Loss: %.4f',
                                    epoch, total_loss)

                    if iteration % 50 == 0:
                        transformed_image = torch.clamp(
                            self(batch_with_old_content),  # transform the image
                            min=0,
                            max=255
                        )[2]

                        tb_writer.add_image('data/fst_images',
                                            img_utils.concat_images(
                                                transformed_image.squeeze(),
                                                batch[2].squeeze()),
                                            iteration)
                    iteration += 1

                    # after processing the batch, run the gradient update
                    optimizer.step(closure)

            torch.save(
                self.state_dict(),
                epoch_checkpoint_name
            )

    def video_process(self, video_path: str, style_name='nsp',
                      working_dir='workdir/',
                      out_dir='results/', fps=24.0):

        video_path = os.path.join(constants.PROJECT_ROOT_PATH, video_path)
        working_dir = os.path.join(constants.PROJECT_ROOT_PATH, working_dir)
        out_dir = os.path.join(constants.PROJECT_ROOT_PATH, out_dir)

        # set weights to latest checkpoint
        self.load_state_dict(
            _load_latest_model_weigths(
                model_name='video_st',
                style_name=style_name
            )
        )

        # we can treat this as a video batch of 1
        video_reader = [imageio.get_reader(video_path)]

        # first we process each video frame, then we join those frames into
        # a final video

        # ensure that working_dir is empty
        shutil.rmtree(working_dir, ignore_errors=True)

        # ensure that the working and result directories exist
        os.makedirs(working_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)

        # of shape [content, stylized]
        old_image = None

        LOGGER.info('Starting to process video into stylized frames')

        # Stylize all frames separately
        for i, video_frame in enumerate(dataset.iterate_on_video_batches(video_reader)):

            # if we're in new epoch then previous frame is None
            if old_image is None:
                old_image = video_frame

            batch_with_old_content = torch.cat(
                [video_frame, old_image],
                dim=1)

            # Get the transformed video image
            transformed_image = self(batch_with_old_content)

            # set old image variable
            old_image = transformed_image.detach()

            img_utils.imshow(transformed_image[0],
                             path=f'{working_dir}{i}.png')

            if i % 50 == 0:
                LOGGER.info('.. processing, currently frame %d', i)

        # convert stylized frames into video
        LOGGER.info('All frames have been stylized.')

        final_path = os.path.join(out_dir, f'video_st_{style_name}.mp4')

        LOGGER.info('Joining stylized frames into a video')

        video_writer = imageio.get_writer(final_path, fps=fps)

        frame_files = sorted(os.listdir(working_dir))
        for frame_name in tqdm(frame_files):
            video_writer.append_data(np.array(Image.open(working_dir + frame_name)))

        LOGGER.info('Done! Final stylized video can be found in: %s', final_path)

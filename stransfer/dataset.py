import json
import os
from typing import Tuple

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset

from stransfer import c_logging, img_utils

LOGGER = c_logging.get_logger()

BASE_COCO_PATH = 'data/coco_dataset/'
IMAGE_FOLDER_PATH = BASE_COCO_PATH + 'images'


def download_coco_images():
    json_file_path = os.path.join(BASE_COCO_PATH,
                                  'image_info_test2017.json')

    coco_dataset = torchvision.datasets.coco.CocoCaptions(
        root=BASE_COCO_PATH,
        annFile=json_file_path)

    # if we haven't downloaded all images then just continue downloading
    n_images = len(json.load(open(json_file_path, 'r'))['images'])

    # try to create images path
    os.makedirs(IMAGE_FOLDER_PATH, exist_ok=True)

    if n_images > len(os.listdir(IMAGE_FOLDER_PATH)):
        coco_dataset.coco.download(tarDir=IMAGE_FOLDER_PATH)


class CocoDataset(Dataset):
    def __init__(self, images, image_limit=None):

        if images is None:
            self.images = os.listdir(IMAGE_FOLDER_PATH)
        else:
            self.images = images

        if image_limit:
            try:
                self.images = self.images[:image_limit]
            except IndexError:
                LOGGER.warn('The provided image limit is larger than '
                            'the actual image set. So will use the whole set')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_path = os.path.join(IMAGE_FOLDER_PATH, self.images[idx])
        image = img_utils.image_loader(img_path)

        # if the image with the specified index doesn't have 3 channels
        # then we discard it
        if image.shape[1] != 3:
            LOGGER.warn('Discarding image with %d color channels',
                        image.shape[1])
            self.images.pop(idx)
            return self.__getitem__(idx)

        return image


def get_coco_loader(batch_size=4, test_split=0.10, test_limit=None, train_limit=None) -> Tuple[DataLoader, DataLoader]:
    # ensure that we have cocoimages
    download_coco_images()

    all_images = os.listdir(IMAGE_FOLDER_PATH)
    split_idx = int(len(all_images) * test_split)

    test_images = all_images[:split_idx]
    train_images = all_images[split_idx:]

    LOGGER.info('Loading train and test set')
    LOGGER.info('Train set has %d entries', len(train_images))
    LOGGER.info('Test set has %d entries', len(test_images))

    test_dataset = CocoDataset(images=test_images,
                               image_limit=test_limit)
    train_dataset = CocoDataset(images=train_images,
                                image_limit=train_limit)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True
    )

    return test_loader, train_loader

import json
import os
import random
import urllib.request
from typing import Tuple

import requests
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import imageio
from PIL import Image
from stransfer import c_logging, img_utils

LOGGER = c_logging.get_logger()

BASE_COCO_PATH = 'data/coco_dataset/'
IMAGE_FOLDER_PATH = BASE_COCO_PATH + 'images'

VIDEO_DATA_PATH = 'data/epic/'


def download_from_url(url, dst):
    """
    @param: url to download file
    @param: dst place to put the file
    """
    file_size = int(requests.head(url).headers["Content-Length"])
    if os.path.exists(dst):
        first_byte = os.path.getsize(dst)
    else:
        first_byte = 0
    if first_byte >= file_size:
        return file_size
    header = {"Range": "bytes=%s-%s" % (first_byte, file_size)}
    pbar = tqdm(
        total=file_size, initial=first_byte,
        unit='B', unit_scale=True, desc=url.split('/')[-1])
    req = requests.get(url, headers=header, stream=True)
    with(open(dst, 'ab')) as f:
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                pbar.update(1024)
    pbar.close()
    return file_size


def download_epic_video():

    videos_to_download = [
        "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/videos/test/P01/P01_11.MP4",
        # "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/videos/test/P01/P01_12.MP4",
        # "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/videos/test/P01/P01_13.MP4",
        # "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/videos/test/P01/P01_14.MP4",
        # "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/videos/test/P01/P01_15.MP4",
    ]

    os.makedirs(VIDEO_DATA_PATH, exist_ok=True)

    if len(videos_to_download) > len(os.listdir(VIDEO_DATA_PATH)):
        for url in videos_to_download:
            filename = url.split('/')[-1]
            filepath = VIDEO_DATA_PATH + filename

            download_from_url(url, VIDEO_DATA_PATH + filename)


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

        try:
            img_path = os.path.join(IMAGE_FOLDER_PATH, self.images[idx])
            image = img_utils.image_loader(img_path)

            # if the image with the specified index doesn't have 3 channels
            # then we discard it
            if image.shape[1] != 3:
                LOGGER.warn('Discarding image with %d color channels',
                            image.shape[1])

                self.images.pop(idx)
                return self.__getitem__(idx)

            else:
                return image

        # catch if file is not image or if idx is out of bounds
        # TODO: change this to proper exceptions (don't leave generic Exception)
        except Exception:
            # not very pretty, but if above we're at the end of the
            # list then idx will be out of bounds. In that case just
            # return a random image from those that do exist
            return self.__getitem__(random.randint(
                0,
                len(self.images)
            ))


class VideoDataset(Dataset):
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

        try:
            img_path = os.path.join(IMAGE_FOLDER_PATH, self.images[idx])
            image = img_utils.image_loader(img_path)

            # if the image with the specified index doesn't have 3 channels
            # then we discard it
            if image.shape[1] != 3:
                LOGGER.warn('Discarding image with %d color channels',
                            image.shape[1])

                self.images.pop(idx)
                return self.__getitem__(idx)

            else:
                return image

        # catch if file is not image or if idx is out of bounds
        # TODO: change this to proper exceptions (don't leave generic Exception)
        except Exception:
            # not very pretty, but if above we're at the end of the
            # list then idx will be out of bounds. In that case just
            # return a random image from those that do exist
            return self.__getitem__(random.randint(
                0,
                len(self.images)
            ))


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
        drop_last=True,
        shuffle=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=0,
        drop_last=True,
        shuffle=True
    )

    return test_loader, train_loader


def get_video_loader(batch_size=4, test_split=0.10, test_limit=None, train_limit=None) -> Tuple[DataLoader, DataLoader]:
    # reader.get_next_data()
    # Image.Image.fromarray()
    # video =
    pass

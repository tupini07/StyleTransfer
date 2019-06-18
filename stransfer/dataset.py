"""
This module holds functionality related to dataset management. Both for downloading
and iterating on the dataset.

In the project we use 2 datasets:

- COCO (for training the fast image transform net)
- Some public videos (for training the Video style transfer net)
"""

import json
import os
import random
from typing import Tuple, List, Any, Generator

import imageio
import requests
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from stransfer import c_logging, img_utils

LOGGER = c_logging.get_logger()

# Where we'll place our coco dataset
BASE_COCO_PATH = 'data/coco_dataset/'
IMAGE_FOLDER_PATH = os.path.join(BASE_COCO_PATH, 'images')

# Where we'll place our video dataset
VIDEO_DATA_PATH = 'data/video/'


def download_from_url(url: str, dst: str) -> int:
    """
    :param url: to download file
    :param dst: place to put the file
    :return: the size of the downloaded file
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


def download_list_of_urls(urls: List[str], destination_folder=VIDEO_DATA_PATH) -> None:
    """
    Download a list of `urls` into `destination_folder`

    :param urls: list of urls to download
    :param destination_folder: the destination folder to which they will be downloaded
    :return: None
    """
    name_counter = 0

    for url in urls:
        try:
            filename = url.split('/')[-1]
            if len(filename) > 20:
                raise Exception
        except Exception:
            filename = f'{name_counter}.mp4'
            name_counter += 1

        filepath = os.path.join(destination_folder, filename)

        download_from_url(url, filepath)


def download_videos_dataset() -> None:
    """
    Ensures that the videos in the video dataset have been downloaded

    :return: None
    """
    videos_to_download = [
        "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4",
        "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4",
        "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/Sintel.mp4",
        "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/TearsOfSteel.mp4"
    ]

    os.makedirs(VIDEO_DATA_PATH, exist_ok=True)

    # if we haven't downloaded all videos then just continue downloading
    if len(videos_to_download) != len(os.listdir(VIDEO_DATA_PATH)):
        download_list_of_urls(videos_to_download)


def download_coco_images() -> None:
    """
    Ensures that the coco dataset is downloaded

    :return: None
    """
    json_file_path = os.path.join(BASE_COCO_PATH,
                                  'image_info_test2017.json')

    images_urls = [x['coco_url']
                   for x in json.load(open(json_file_path, 'r'))['images']]

    os.makedirs(IMAGE_FOLDER_PATH, exist_ok=True)

    # if we haven't downloaded all images then just continue downloading
    if len(images_urls) != len(os.listdir(IMAGE_FOLDER_PATH)):
        download_list_of_urls(images_urls)


def make_batches(l: List[Any], n: int) -> List[List[Any]]:
    """
    Yield successive n-sized chunks from l.

    :param l: list of elements we want to convert into batches
    :param n: size of the batches
    :return: list of batches of elements
    """

    batches = []
    for i in range(0, len(l), n):
        batches.append(l[i:i + n])

    return batches


class CocoDataset(Dataset):
    """
    An implementation of the torch Dataset class, specific for the
    COCO dataset
    """

    def __init__(self, images=None, image_limit=None):
        """
        :param images: list of paths of images. If not specified then the images
            in `IMAGE_FOLDER_PATH` will be used.
        :param image_limit: the maximum amount of images we want to use
        """

        if images is None:
            self.images = os.listdir(IMAGE_FOLDER_PATH)
        else:
            self.images = images

        if image_limit:
            try:
                self.images = self.images[:image_limit]
            except IndexError:
                LOGGER.warning('The provided image limit is larger than '
                               'the actual image set. So will use the whole set')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        try:
            img_path = os.path.join(IMAGE_FOLDER_PATH, self.images[idx])
            image = img_utils.image_loader(img_path)

            # if the image with the specified index doesn't have 3 channels
            # then we discard it
            # In the coco dataset there are some greyscale images (only one channel)
            if image.shape[1] != 3:
                LOGGER.warning('Discarding image with %d color channels',
                               image.shape[1])

                self.images.pop(idx)
                return self.__getitem__(idx)

            else:
                return image

        # catch if file is not image or if idx is out of bounds
        # TODO: might want to change this to catch proper exceptions (instead of generic Exception)
        except Exception:
            # not very pretty, but if above we're at the end of the
            # list then idx will be out of bounds. In that case just
            # return a random image from those that do exist
            return self.__getitem__(random.randint(
                0,
                len(self.images)
            ))


class VideoDataset:
    """
    Dataset wrapper for the video dataset
    """

    def __init__(self, videos=None, data_limit=None, batch_size=3):
        """
        :param videos: list of paths of videos. If not specified then the videos
            in `VIDEO_DATA_PATH` will be used.
        :param data_limit: maximum amount of videos to use as part of the dataset
        :param batch_size: the batch size we will split our videos in
        """

        # Ensure video have been downloaded
        download_videos_dataset()

        if videos is None:
            self.videos = os.listdir(VIDEO_DATA_PATH)
        else:
            self.videos = videos

        if data_limit:
            try:
                self.videos = self.videos[:data_limit]
            except IndexError:
                LOGGER.warning('The provided video limit is larger than '
                               'the actual amount of videos in the video set. So will use the whole set')

        # set proper value for batch size
        if batch_size > len(self.videos):
            LOGGER.warning('The batch size is larger than the amount of videos in the '
                           f'video set. Will use complete set as a batch of size {len(self.videos)}')
            self.batch_size = len(self.videos)
        else:
            self.batch_size = batch_size

        # create video loaders for each video
        self.video_paths = []
        for vid_name in self.videos:
            vid_path = os.path.join(VIDEO_DATA_PATH, vid_name)
            self.video_paths.append(
                vid_path
            )

        # separate video loaders into batches
        self.video_paths = make_batches(self.video_paths,
                                        self.batch_size)

        # Throw away last batch if it is not the perfect batchsize length
        if len(self.video_paths[-1]) != self.batch_size:
            self.video_paths = self.video_paths[:-1]

        # vars for iterator management
        self.current_i = 0

    def __len__(self):
        # number of batches
        return len(self.video_paths)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            # we're iterating over batches of videos
            # so each iteration we get the batch corresponding to
            # `current_i`
            video_paths = self.video_paths[self.current_i]
        except IndexError:
            # once we reach the end of the list we stop
            # iteration
            self.current_i = 0
            raise StopIteration

        self.current_i += 1

        # for each video in the batch we return a video reader for said video
        return [imageio.get_reader(vp) for vp in video_paths]


def iterate_on_video_batches(batch: List[imageio.core.format.Format.Reader],
                             max_frames=90 * 24) -> Generator[torch.Tensor, None, None]:
    """
    Generator that, given a list of video readers, will yield
    at each iteration a list composed of one frame from each video.

    :param batch: batch of video readers we want to iterate on
    :param max_frames: the maximum number of frames we want to yield. By default we limit
        to 90 seconds which is the same as 90*24 if the videos are 24 FPS
    """

    try:
        for _ in range(max_frames):

            next_data = []
            for video_reader in batch:
                frame = video_reader.get_next_data()

                # convert image to tensor
                image = Image.fromarray(frame)
                tensor = img_utils.image_loader_transform(image)

                # add to data we'll yield for the current batch
                next_data.append(tensor)

            # concatenate frames across their batch dimension and yield
            yield torch.cat(next_data, dim=0)

    # when one of the videos finishes imageio will
    # throw an IndexError when getting `get_next_data`
    except IndexError:
        pass


def get_coco_loader(batch_size=4, test_split=0.10, test_limit=None, train_limit=None) -> Tuple[DataLoader, DataLoader]:
    """
    Sets up and returns the dataloaders for the coco dataset

    :param batch_size: the amount of elements we want per batch
    :param test_split: the percentage of items from the whole set that we want to be part
        of the test set
    :param test_limit: the maximum amount of items we want in our test set
    :param train_limit: the maximum amount of items we want in the training set
    :return: the test set dataloader, and the train set dataloader
    """

    # ensure that we have coco images
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

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

VIDEO_DATA_PATH = 'data/video/'


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


def download_videos_dataset():

    videos_to_download = [
        "https://r2---sn-b5gg-ca9e.googlevideo.com/videoplayback?id=o-AEklCRSCfHCzvOFgc32RXwdiYwHMN2BZiGzchQ0qrgoh&itag=22&source=youtube&requiressl=yes&pl=20&ei=w7XrXLivF4OQ1gKs07TYCg&mime=video%2Fmp4&ratebypass=yes&dur=176.285&lmt=1558107423152746&fvip=4&beids=9466587&c=WEB&txp=5535432&ip=195.201.8.116&ipbits=0&expire=1558972963&sparams=dur,ei,expire,id,ip,ipbits,itag,lmt,mime,mip,mm,mn,ms,mv,pl,ratebypass,requiressl,source&signature=438DA84712ED30AAF322B7D41FD7C2F2EABB962B.2818B9539C365D9D0FB2D272F7CB3184BE89F76D&key=cms1&title=How+People+Walk&title=How+People+Walk&cms_redirect=yes&mip=193.205.210.82&mm=31&mn=sn-b5gg-ca9e&ms=au&mt=1558960747&mv=m",
    ]

    os.makedirs(VIDEO_DATA_PATH, exist_ok=True)

    name_counter = 0
    if len(videos_to_download) > len(os.listdir(VIDEO_DATA_PATH)):
        for url in videos_to_download:
            try:
                filename = url.split('/')[-1]
                if len(filename) > 20:
                    raise Exception
            except Exception:
                filename = f'{name_counter}.mp4'
                name_counter += 1

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


def make_batches(l, n):
    """Yield successive n-sized chunks from l."""
    batches = []
    for i in range(0, len(l), n):
        batches.append(l[i:i + n])

    return batches


class CocoDataset(Dataset):
    def __init__(self, images=None, image_limit=None):

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


class VideoDataset:
    def __init__(self, videos=None, data_limit=None, batch_size=3):
        download_videos_dataset()

        if videos is None:
            self.videos = os.listdir(VIDEO_DATA_PATH)
        else:
            self.videos = videos

        if data_limit:
            try:
                self.videos = self.videos[:data_limit]
            except IndexError:
                LOGGER.warn('The provided video limit is larger than '
                            'the actual amount of videos in the video set. So will use the whole set')

        # set proper value for batch size
        if batch_size > len(self.videos):
            LOGGER.warn('The batch size is larger than the amount of videos in the '
                        f'video set. Will use complete set as a batch of size {len(self.videos)}')
            self.batch_size = len(self.videos)
        else:
            self.batch_size = self.videos

        # create video loaders for each video
        self.video_loaders = []
        for vid_name in self.videos:
            vid_path = os.path.join(VIDEO_DATA_PATH, vid_name)
            self.video_loaders.append(
                imageio.get_reader(vid_path)
            )

        # separate video loaders into batches
        self.video_loaders = make_batches(self.video_loaders,
                                          self.batch_size)

        # Throw away last batch if it is not the perfect batchsize length
        if len(self.video_loaders[-1]) != self.batch_size:
            self.video_loaders = self.video_loaders[:-1]

        # vars for iterator management
        self.current_i = 0

    def __len__(self):
        # number of batches
        return len(self.video_loaders)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            result = self.video_loaders[self.current_i]
        except IndexError:
            self.current_i = 0
            raise StopIteration

        self.current_i += 1
        return result


def iterate_on_video_batches(batch):
    try:
        while True:
            next_data = []
            for tt in batch:
                frame = tt.get_next_data()
                image = Image.fromarray(frame)
                tensor = img_utils.image_loader_transform(image)
                next_data.append(tensor)

            yield torch.cat(next_data, dim=0)

    except IndexError:
        raise StopIteration


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

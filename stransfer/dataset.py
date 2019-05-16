import os
import json
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset

from stransfer import img_utils

BASE_COCO_PATH = 'data/coco_dataset/'
IMAGE_FOLDER_PATH = BASE_COCO_PATH + 'images'


def download_coco_images():
    json_file_path = os.path.join(BASE_COCO_PATH,
                                  'image_info_test2017.json')

    coco_dataset = torchvision.datasets.coco.CocoCaptions(
        annFile=json_file_path)

    # if we haven't downloaded all images then just continue downloading
    n_images = len(json.load(open(json_file_path, 'r'))['images'])

    if n_images > len(os.listdir(IMAGE_FOLDER_PATH)):
        os.makedirs(IMAGE_FOLDER_PATH, exist_ok=True)
        coco_dataset.coco.download(tarDir=IMAGE_FOLDER_PATH)


class CocoDataset(Dataset):
    def __init__(self, image_limit=None):
        # ensure that we have cocoimages
        download_coco_images()

        self.images = os.listdir(IMAGE_FOLDER_PATH)

        if image_limit:
            self.images = self.images[:image_limit]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_path = os.path.join(IMAGE_FOLDER_PATH, self.images[idx])
        image = img_utils.image_loader(img_path)

        # if the image with the specified index doesn't have 3 channels
        # then we discard it
        if image.shape[1] != 3:
            self.images.pop(idx)
            return self.__getitem__(idx)

        return image


def get_coco_loader(image_limit=None) -> torch.Tensor:
    dataset = CocoDataset(image_limit)

    c_loader = DataLoader(
        dataset,
        batch_size=4,
        num_workers=0,
        shuffle=True
    )

    return c_loader

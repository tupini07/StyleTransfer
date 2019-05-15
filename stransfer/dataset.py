import torchvision
import os
BASE_COCO_PATH = 'data/coco_dataset/'
IMAGE_FOLDER_PATH = BASE_COCO_PATH + 'images'

_COCO_DATASET = torchvision.datasets.coco.CocoCaptions(
    BASE_COCO_PATH, 
    BASE_COCO_PATH + 'image_info_test2017.json')

# image folder path does not exist or if it is empty then we doenload the data
if not os.path.exists(IMAGE_FOLDER_PATH) or not os.listdir(IMAGE_FOLDER_PATH):
    os.makedirs(IMAGE_FOLDER_PATH, exist_ok=True)
    _COCO_DATASET.coco.download(tarDir=IMAGE_FOLDER_PATH)

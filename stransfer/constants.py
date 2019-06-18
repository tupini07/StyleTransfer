"""
This module holds constant values used throughout the application
"""
import os

import torch

# where information of the runs will be saved.
# both for the log of the run and for tensorboard results
RUNS_PATH = 'runs/'

LOG_PATH = os.path.join(RUNS_PATH, 'runtime.log')

# used for normalizing and denormalizing input and output
# images respectively
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Always use cuda if available, but fallback to CPU if not.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DEVICE.type == "cuda":
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    torch.set_default_tensor_type(torch.FloatTensor)

# size of input images and output images
# if input images are not IMSIZExIMSIZE then a square of size
# IMSIZE will be cropped from the center of the image.
IMSIZE = 256

PROJECT_ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

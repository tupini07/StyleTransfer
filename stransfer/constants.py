import torch

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# desired size of the output image
# IMSIZE = 512 if torch.cuda.is_available() else 128  # use small size if no gpu
IMSIZE = 256

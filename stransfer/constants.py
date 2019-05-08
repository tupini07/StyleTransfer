import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# desired size of the output image
IMSIZE = 512 if torch.cuda.is_available() else 128  # use small size if no gpu

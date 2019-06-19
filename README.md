# Style Transfer

[![Build Status](https://travis-ci.org/tupini07/StyleTransfer.svg?branch=master)](https://tupini07.github.io/StyleTransfer/index.html)

In this repository you'll find a PyTorch implementation of the following papers:

- [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)
- [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155)

And also of a network able to apply a style to a video, using our own `temporal loss`.

All of these are implemented in the `stransfer.network` module.

Documentation of each module and their members can be found [here](https://tupini07.github.io/StyleTransfer/index.html)

## Running in Google Colab

If you don't want to run this in your local PC then you can also run this project in Google Colab.

See [this notebook](https://github.com/tupini07/StyleTransfer/blob/master/Colab_Style_Transfer.ipynb)
for an example of how to clone this repository, download the COCO dataset, and install the dependencies
in a Colab environment. Once everything is downloaded and installed you can use the commands
provided by the command line interface as if you were in a normal terminal (see [How to use](https://github.com/tupini07/StyleTransfer#how-to-use)
section for those). 

The example notebook also has a button at the top which will automatically create for you a copy 
of the example notebook in Colab.

## Installing Dependencies

You can use `pipenv` to create an environment for this project, with all the dependencies installed.

If you don't know how to use _pipenv_ then just ensure you have `pipenv` installed, 
clone this repository, `cd` into it, and run
`pipenv install`. Then you can execute `pipenv shell` to activate the virtual environment.

## How to use

The `stransfer` package provides a command line interface that exposes all of the functionality to the user. 

Execution is separated in groups of commands. Each of which exposes different tasks. The basic execution pattern is: 

```
python -m stransfer <group name> <task name>
```

To see which _groups_ are available run `python -m stransfer`. And to see which tasks are available in each group run `python -m stranfer <group name> --help` (replacing `<group name>` with the name of the group). 

To see the description of each task you can do `python -m stransfer <group name> <task name> --help`.

For a deeper explanation of each group/task you can also check [the respective documentation page](https://tupini07.github.io/StyleTransfer/terminal_interface.html)

### Group names and their tasks

- `video_st` - Video Style Transfer
    - `convert-video` - Converts the video at `video-path` using the network pretrained with `style-name` and saves the resulting transformed video in `out-dir`.
    - `train` - Perform the training for the video style transfer network.
- `gatys_st` - Run the original Gatys style transfer (slow). (This is actually a task)
- `fast_st` - Fast Style Transfer
    - `train` - Perform the training for the fast style transfer network. A checkpoint will be created at the end of each epoch in the `data/models/` directory.
    - `convert-image` - Converts the image at `image-path` using the network pretrained with `style-name` and saves the resulting transformed image in `out-dir`.

## Download Pretrained Models

Pretrained weights can be downloaded from [here](https://drive.google.com/drive/folders/11lsETWvucCiesEaqs5fK5PaCbNMmOV4w?usp=sharing)

After downloading, put them inside `data/models/`.


## Training your own models

You can train your own `fast_st` and `video_st` models with custom styles by using the 
appropriate `train` command. 

To train the `fast_st` network the application will download the [COCO dataset](http://cocodataset.org/)
automatically. However, it will download one image at a time, which is pretty slow. A faster
way would be for you to download [this .7z file (6GB)](https://drive.google.com/file/d/1mTZGqm9fq8vGjkpNph-fhZXW2Dkt-tsH/view?usp=sharing)
and extract it inside `data/coco_dataset/images/`.

Videos will also be downloaded when training the `video_st`, but these are currently only
_4_. If you want to add more videos to the video training set then just add a URL to them
to the list in `stransfer.dataset.download_videos_dataset.videos_to_download`.


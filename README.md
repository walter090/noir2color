# noir2color

## Introduction
This repo contain the project to train and test a conditional generative adversarial network model to colorize black and white images. The implementation is based on Ian Goodfellow's [GAN paper](https://arxiv.org/abs/1406.2661).

## Requirements
This project is implemented with Tensorflow, to install the latest version of Tensorflow, follow the instructions on [this page](https://www.tensorflow.org/install/).

## Usage
### Training
To train on your own dataset, use
```commandline
python noir2color.py --bw-folder bw --colored-folder color
```
You can also set other parameters using the command line, for example
```commandline
python noir2color.py --keep-prob 0.75
```
For information about specifying other parameters, type
```commandline
python noir2color.py -h
```

### Colorize
To use a trained model, use the model_test function in colorizer.py. The function takes one or a list of black and white images and output the colorized. For details regrading the arguments for this function, check the docstring.

To colorize one single image, use
```commandline
python colorizer.py --meta dir/to/saved/model --input input_image.jpg
```
# Pixelwise Segmentation
> Pixelwise segmentation for document images

## Introduction
This repository contains the source code for training an encoder model for document image segmentation.

## Installation
Either clone the repository via `git clone https://github.com/qurator-spk/sbb_pixelwise_segmentation.git` or download and unpack the [ZIP](https://github.com/qurator-spk/sbb_pixelwise_segmentation/archive/master.zip).

### Pretrained encoder
Download our pretrained weights and add them to a ``pretrained_model`` folder:   
https://qurator-data.de/sbb_pixelwise_segmentation/pretrained_encoder/
## Usage

### Train
To train a model, run: ``python train.py with config_params.json``
      
### Ground truth format
Lables for each pixel are identified by a number. So if you have a 
binary case, ``n_classes`` should be set to ``2`` and labels should 
be ``0`` and ``1`` for each class and pixel.

In the case of multiclass, just set ``n_classes`` to the number of classes 
you have and the try to produce the labels by pixels set from ``0 , 1 ,2 .., n_classes-1``.
The labels format should be png. 
Our lables are 3 channel png images but only information of first channel is used. 
If you have an image label with height and width of 10, for a binary case the first channel should look like this:
    
    Label: [ [1, 0, 0, 1, 1, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             ...,
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] ] 
    
 This means that you have an image by `10*10*3` and `pixel[0,0]` belongs
 to class `1` and `pixel[0,1]` belongs to class `0`.
    
### Training , evaluation and output 
The train and evaluation folders should contain subfolders of images and labels.
The output folder should be an empty folder where the output model will be written to.
    
### Parameter configuration
* patches: If you want to break input images into smaller patches (input size of the model) you need to set this parameter to ``true``. In the case that the model should see the image once, like page extraction, patches should be set to ``false``.
* n_batch: Number of batches at each iteration.
* n_classes: Number of classes. In the case of binary classification this should be 2.
* n_epochs: Number of epochs.
* input_height: This indicates the height of model's input.
* input_width: This indicates the width of model's input.
* weight_decay: Weight decay of l2 regularization of model layers.
* augmentation: If you want to apply any kind of augmentation this parameter should first set to ``true``.
* flip_aug: If ``true``, different types of filp will applied on image. Type of flips is given by "flip_index" in train.py file.
    


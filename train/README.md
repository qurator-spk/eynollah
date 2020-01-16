# Pixelwise Segmentation
> Pixelwise segmentation for document images

## Introduction
This repository contains the source code for training an encoder model for document image segmentation.

## Installation
Either clone the repository via `git clone https://github.com/qurator-spk/sbb_pixelwise_segmentation.git` or download and unpack the [ZIP](https://github.com/qurator-spk/sbb_pixelwise_segmentation/archive/master.zip).

### Pretrained encoder
Download our pretrained weights and add them to a ``pretrained_model`` folder:   
https://qurator-data.de/pretrained_encoder/
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
    
If you have an image label for a binary case it should look like this:
    
    Label: [ [[1 0 0 1], [1 0 0 1] ,[1 0 0 1]], 
    [[1 0 0 1], [1 0 0 1] ,[1 0 0 1]] ,
    [[1 0 0 1], [1 0 0 1] ,[1 0 0 1]] ] 
    
 This means that you have an image by `3*4*3` and `pixel[0,0]` belongs
 to class `1` and `pixel[0,1]` belongs to class `0`.
    
### Training , evaluation and output 
The train and evaluation folders should contain subfolders of images and labels.
The output folder should be an empty folder where the output model will be written to.
    
### Patches
If you want to train your model with patches, the height and width of
the patches should be defined and also the number of batches (how many patches 
should be seen by the model in each iteration).

In the case that the model should see the image once, like page extraction,
patches should be set to ``false``.
    


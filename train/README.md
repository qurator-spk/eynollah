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
 
 A small sample of training data for binarization experiment can be found here, [Training data sample](https://qurator-data.de/~vahid.rezanezhad/binarization_training_data_sample/), which contains images and lables folders.
    
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
* flip_aug: If ``true``, different types of filp will be applied on image. Type of flips is given with "flip_index" in train.py file.
* blur_aug: If ``true``, different types of blurring will be applied on image. Type of blurrings is given with "blur_k" in train.py file.
* scaling: If ``true``, scaling will be applied on image. Scale of scaling is given with "scales" in train.py file.
* rotation_not_90: If ``true``, rotation (not 90 degree) will be applied on image. Rothation angles are given with "thetha" in train.py file.
* rotation: If ``true``, 90 degree rotation will be applied on image.
* binarization: If ``true``,Otsu thresholding will be applied to augment the input data with binarized images.
* scaling_bluring: If ``true``, combination of scaling and blurring will be applied on image.
* scaling_binarization: If ``true``, combination of scaling and binarization will be applied on image.
* scaling_flip: If ``true``, combination of scaling and flip will be applied on image.
* continue_training: If ``true``, it means that you have already trained a model and you would like to continue the training. So it is needed to provide the dir of trained model with "dir_of_start_model" and index for naming the models. For example if you have already trained for 3 epochs then your last index is 2 and if you want to continue from model_1.h5, you can set "index_start" to 3 to start naming model with index 3. 
* weighted_loss: If ``true``, this means that you want to apply weighted categorical_crossentropy as loss fucntion. Be carefull if you set to ``true``the parameter "is_loss_soft_dice" should be ``false``
* data_is_provided: If you have already provided the input data you can set this to ``true``. Be sure that the train and eval data are in "dir_output". Since when once we provide training data we resize and augment them and then we write them in sub-directories train and eval in "dir_output". 
* dir_train: This is the directory of "images" and "labels" (dir_train should include two subdirectories with names of images and labels ) for raw images and labels. Namely they are not prepared (not resized and not augmented) yet for training the model. When we run this tool these raw data will be transformed to suitable size needed for the model and they will be written in "dir_output" in train and eval directories. Each of train and eval include "images" and "labels" sub-directories.
    


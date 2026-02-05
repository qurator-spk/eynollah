import os
import sys
import json

import requests
import click

from eynollah.training.metrics import (
    soft_dice_loss,
    weighted_categorical_crossentropy
)
from eynollah.training.models import (
    PatchEncoder,
    Patches,
    machine_based_reading_order_model,
    resnet50_classifier,
    resnet50_unet,
    vit_resnet50_unet,
    vit_resnet50_unet_transformer_before_cnn,
    RESNET50_WEIGHTS_PATH,
    RESNET50_WEIGHTS_URL
)
from eynollah.training.utils import (
    data_gen,
    generate_arrays_from_folder_reading_order,
    generate_data_from_folder_evaluation,
    generate_data_from_folder_training,
    get_one_hot,
    preprocess_imgs,
    return_number_of_total_training_data
)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_USE_LEGACY_KERAS'] = '1' # avoid Keras 3 after TF 2.15
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sacred import Experiment
from tqdm import tqdm
from sklearn.metrics import f1_score

import numpy as np
import cv2

class SaveWeightsAfterSteps(ModelCheckpoint):
    def __init__(self, save_interval, save_path, _config, **kwargs):
        if save_interval:
            # batches
            super().__init__(
                os.path.join(save_path, "model_step_{batch:04d}"),
                save_freq=save_interval,
                verbose=1,
                **kwargs)
        else:
            super().__init__(
                os.path.join(save_path, "model_{epoch:02d}"),
                save_freq="epoch",
                verbose=1,
                **kwargs)
        self._config = _config

    # overwrite tf-keras (Keras 2) implementation to get our _config JSON in
    def _save_handler(self, filepath):
        super()._save_handler(filepath)
        with open(os.path.join(filepath, "config.json"), "w") as fp:
            json.dump(self._config, fp)  # encode dict into JSON
            
            
def configuration():
    try:
        for device in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(device, True)
    except:
        print("no GPU device available", file=sys.stderr)


def get_dirs_or_files(input_data):
    image_input, labels_input = os.path.join(input_data, 'images/'), os.path.join(input_data, 'labels/')
    if os.path.isdir(input_data):
        # Check if training dir exists
        assert os.path.isdir(image_input), "{} is not a directory".format(image_input)
        assert os.path.isdir(labels_input), "{} is not a directory".format(labels_input)
    return image_input, labels_input

def download_file(url, path):
    with open(path, 'wb') as f:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            for data in r.iter_content(chunk_size=4096):
                f.write(data)

ex = Experiment(save_git_info=False)


@ex.config
def config_params():
    task = "segmentation" # This parameter defines task of model which can be segmentation, enhancement or classification.
    if task in ["segmentation", "binarization", "enhancement"]:
        backbone_type = "nontransformer" # Type of image feature map network backbone. Either a vision transformer alongside a CNN we call "transformer", or only a CNN which we call "nontransformer"
        if backbone_type == "transformer":
            transformer_patchsize_x = None  # Patch size of vision transformer patches in x direction.
            transformer_patchsize_y = None # Patch size of vision transformer patches in y direction.
            transformer_num_patches_xy = None  # Number of patches for vision transformer in x and y direction respectively.
            transformer_projection_dim = 64 # Transformer projection dimension. Default value is 64.
            transformer_mlp_head_units = [128, 64] # Transformer Multilayer Perceptron (MLP) head units. Default value is [128, 64]
            transformer_layers = 8 # transformer layers. Default value is 8.
            transformer_num_heads = 4 # Transformer number of heads. Default value is 4.
            transformer_cnn_first = True # We have two types of vision transformers: either the CNN is applied first, followed by the transformer, or reversed.
    n_classes = None  # Number of classes. In the case of binary classification this should be 2.
    n_epochs = 1  # Number of epochs to train.
    n_batch = 1  # Number of images per batch at each iteration. (Try as large as fits on VRAM.)
    input_height = 224 * 1  # Height of model's input in pixels.
    input_width = 224 * 1  # Width of model's input in pixels.
    weight_decay = 1e-6  # Weight decay of l2 regularization of model layers.
    learning_rate = 1e-4  # Set the learning rate.
    if task in ["segmentation", "binarization"]:
        is_loss_soft_dice = False  # Use soft dice as loss function. When set to true, "weighted_loss" must be false.
        weighted_loss = False  # Use weighted categorical cross entropy as loss fucntion. When set to true, "is_loss_soft_dice" must be false.
    elif task == "classification":
        f1_threshold_classification = None # This threshold is used to consider models with an evaluation f1 scores bigger than it. The selected model weights undergo a weights ensembling. And avreage ensembled model will be written to output.
        classification_classes_name = None # Dictionary of classification classes names.
    patches = False  # Divides input image into smaller patches (input size of the model) when set to true. For the model to see the full image, like page extraction, set this to false.
    augmentation = False  # To apply any kind of augmentation, this parameter must be set to true.
    if augmentation:
        flip_aug = False  # If true, different types of flipping will be applied to the image. Types of flips are defined with "flip_index" in config_params.json.
        if flip_aug:
            flip_index = None  #  Flip image for augmentation.
        blur_aug = False  # If true, different types of blurring will be applied to the image. Types of blur are defined with "blur_k" in config_params.json.
        if blur_aug:
            blur_k = None  # Blur image for augmentation.
        padding_white = False # If true, white padding will be applied to the image.
        padding_black = False # If true, black padding will be applied to the image.
        scaling = False  # If true, scaling will be applied to the image. The amount of scaling is defined with "scales" in config_params.json.
        scaling_bluring = False  # If true, a combination of scaling and blurring will be applied to the image.
        scaling_binarization = False  # If true, a combination of scaling and binarization will be applied to the image.
        scaling_brightness = False  # If true, a combination of scaling and brightening will be applied to the image.
        scaling_flip = False  # If true, a combination of scaling and flipping will be applied to the image.
        if scaling or scaling_brightness or scaling_bluring or scaling_binarization or scaling_flip:
            scales = None  # Scale patches for augmentation.
        shifting = False
        degrading = False  # If true, degrading will be applied to the image. The amount of degrading is defined with "degrade_scales" in config_params.json.
        if degrading:
            degrade_scales = None  # Degrade image for augmentation.
        brightening = False  # If true, brightening will be applied to the image. The amount of brightening is defined with "brightness" in config_params.json.
        if brightening:
            brightness = None #  Brighten image for augmentation.
        binarization = False  # If true, Otsu thresholding will be applied to augment the input with binarized images.
        if binarization:
            dir_img_bin = None # Directory of training dataset subdirectory of binarized images
            add_red_textlines = False
            adding_rgb_background = False
            if adding_rgb_background:
                dir_rgb_backgrounds = None # Directory of texture images for synthetic background
            adding_rgb_foreground = False
            if adding_rgb_foreground:
                dir_rgb_foregrounds = None # Directory of texture images for synthetic foreground
            if adding_rgb_background or adding_rgb_foreground:
                number_of_backgrounds_per_image = 1
        channels_shuffling = False # Re-arrange color channels.
        if channels_shuffling:
            shuffle_indexes = None # Which channels to switch between.
        rotation = False # If true, a 90 degree rotation will be implemeneted.
        rotation_not_90 = False # If true rotation based on provided angles with thetha will be implemeneted.
        if rotation_not_90:
            thetha = None  # Rotate image by these angles for augmentation.
    dir_train = None  # Directory of training dataset with subdirectories having the names "images" and "labels".
    dir_eval = None  # Directory of validation dataset with subdirectories having the names "images" and "labels".
    dir_output = None  # Directory where the augmented training data and the model checkpoints will be saved.
    pretraining = False  # Set to true to (down)load pretrained weights of ResNet50 encoder.
    save_interval = None # frequency for writing model checkpoints (positive integer for number of batches saved under "model_step_{batch:04d}", otherwise epoch saved under "model_{epoch:02d}")
    continue_training = False  # Whether to continue training an existing model.
    if continue_training:
        dir_of_start_model = ''  # Directory of model checkpoint to load to continue training. (E.g. if you already trained for 3 epochs, set "dir_of_start_model=dir_output/model_03".)
        index_start = 0  #  Epoch counter initial value to continue training. (E.g. if you already trained for 3 epochs, set "index_start=3" to continue naming checkpoints model_04, model_05 etc.)
    data_is_provided = False  # Whether the preprocessed input data (subdirectories "images" and "labels" in both subdirectories "train" and "eval" of "dir_output") has already been generated (in the first epoch of a previous run).

@ex.main
def run(_config,
        _log,
        task,
        pretraining,
        data_is_provided,
        dir_train,
        dir_eval,
        dir_output,
        n_classes,
        n_epochs,
        n_batch,
        input_height,
        input_width,
        weight_decay,
        learning_rate,
        continue_training,
        save_interval,
        augmentation,
        # dependent config keys need a default,
        # otherwise yields sacred.utils.ConfigAddedError
        thetha=None,
        is_loss_soft_dice=False,
        weighted_loss=False,
        index_start=0,
        dir_of_start_model=None,
        backbone_type=None,
        transformer_projection_dim=None,
        transformer_mlp_head_units=None,
        transformer_layers=None,
        transformer_num_heads=None,
        transformer_cnn_first=None,
        transformer_patchsize_x=None,
        transformer_patchsize_y=None,
        transformer_num_patches_xy=None,
        f1_threshold_classification=None,
        classification_classes_name=None,
):

    if pretraining and not os.path.isfile(RESNET50_WEIGHTS_PATH):
        _log.info("downloading RESNET50 pretrained weights to %s", RESNET50_WEIGHTS_PATH)
        download_file(RESNET50_WEIGHTS_URL, RESNET50_WEIGHTS_PATH)

    # set the gpu configuration
    configuration()

    if task in ["segmentation", "enhancement", "binarization"]:
        dir_train_flowing = os.path.join(dir_output, 'train')
        dir_eval_flowing = os.path.join(dir_output, 'eval')

        dir_flow_train_imgs = os.path.join(dir_train_flowing, 'images')
        dir_flow_train_labels = os.path.join(dir_train_flowing, 'labels')

        dir_flow_eval_imgs = os.path.join(dir_eval_flowing, 'images')
        dir_flow_eval_labels = os.path.join(dir_eval_flowing, 'labels')

        if not data_is_provided:
            # first create a directory in output for both training and evaluations
            # in order to flow data from these directories.
            if os.path.isdir(dir_train_flowing):
                os.system('rm -rf ' + dir_train_flowing)
            os.makedirs(dir_train_flowing)

            if os.path.isdir(dir_eval_flowing):
                os.system('rm -rf ' + dir_eval_flowing)
            os.makedirs(dir_eval_flowing)

            os.mkdir(dir_flow_train_imgs)
            os.mkdir(dir_flow_train_labels)

            os.mkdir(dir_flow_eval_imgs)
            os.mkdir(dir_flow_eval_labels)

            dir_img, dir_seg = get_dirs_or_files(dir_train)
            dir_img_val, dir_seg_val = get_dirs_or_files(dir_eval)

            imgs_list=np.array(os.listdir(dir_img))
            segs_list=np.array(os.listdir(dir_seg))
            
            imgs_list_test=np.array(os.listdir(dir_img_val))
            segs_list_test=np.array(os.listdir(dir_seg_val))

            # writing patches into a sub-folder in order to be flowed from directory.
            preprocess_imgs(_config,
                            imgs_list,
                            segs_list,
                            dir_img,
                            dir_seg,
                            dir_flow_train_imgs,
                            dir_flow_train_labels)
            preprocess_imgs(_config,
                            imgs_list_test,
                            segs_list_test,
                            dir_img_val,
                            dir_seg_val,
                            dir_flow_eval_imgs,
                            dir_flow_eval_labels,
                            augmentation=False)

        if weighted_loss:
            weights = np.zeros(n_classes)
            if data_is_provided:
                dirs = dir_flow_train_labels
            else:
                dirs = dir_seg
            for obj in os.listdir(dirs):
                label_file = os.path.join(dirs, + obj)
                try:
                    label_obj = cv2.imread(label_file)
                    label_obj_one_hot = get_one_hot(label_obj, label_obj.shape[0], label_obj.shape[1], n_classes)
                    weights += (label_obj_one_hot.sum(axis=0)).sum(axis=0)
                except Exception:
                    _log.exception("error reading data file '%s'", label_file)

            weights = 1.00 / weights
            weights = weights / float(np.sum(weights))
            weights = weights / float(np.min(weights))
            weights = weights / float(np.sum(weights))

        if task == "enhancement":
            assert not is_loss_soft_dice, "for enhancement, soft_dice loss does not apply"
            assert not weighted_dice, "for enhancement, weighted loss does not apply"
        if continue_training:
            custom_objects = dict()
            if is_loss_soft_dice:
                custom_objects.update(soft_dice_loss=soft_dice_loss)
            elif weighted_loss:
                custom_objects.update(loss=weighted_categorical_crossentropy(weights))
            if backbone_type == 'transformer':
                custom_objects.update(PatchEncoder=PatchEncoder,
                                      Patches=Patches)
            model = load_model(dir_of_start_model, compile=False,
                               custom_objects=custom_objects)
        else:
            index_start = 0
            if backbone_type == 'nontransformer':
                model = resnet50_unet(n_classes,
                                      input_height,
                                      input_width,
                                      task,
                                      weight_decay,
                                      pretraining)
            elif backbone_type == 'transformer':
                num_patches_x = transformer_num_patches_xy[0]
                num_patches_y = transformer_num_patches_xy[1]
                num_patches = num_patches_x * num_patches_y
                
                if transformer_cnn_first:
                    model_builder = vit_resnet50_unet
                    multiple_of_32 = True
                else:
                    model_builder = vit_resnet50_unet_transformer_before_cnn
                    multiple_of_32 = False

                assert input_height == num_patches_y * transformer_patchsize_y * (32 if multiple_of_32 else 1), \
                    "transformer_patchsize_y or transformer_num_patches_xy height value error: " \
                    "input_height should be equal to " \
                    "(transformer_num_patches_xy height value * transformer_patchsize_y%s)" % \
                    " * 32" if multiple_of_32 else ""
                assert input_width == num_patches_x * transformer_patchsize_x * (32 if multiple_of_32 else 1), \
                    "transformer_patchsize_x or transformer_num_patches_xy width value error: " \
                    "input_width should be equal to " \
                    "(transformer_num_patches_xy width value * transformer_patchsize_x%s)" % \
                    " * 32" if multiple_of_32 else ""
                assert 0 == transformer_projection_dim % (transformer_patchsize_y * transformer_patchsize_x), \
                    "transformer_projection_dim error: " \
                    "The remainder when parameter transformer_projection_dim is divided by " \
                    "(transformer_patchsize_y*transformer_patchsize_x) should be zero"

                model = model_builder(
                    n_classes,
                    transformer_patchsize_x,
                    transformer_patchsize_y,
                    num_patches,
                    transformer_mlp_head_units,
                    transformer_layers,
                    transformer_num_heads,
                    transformer_projection_dim,
                    input_height,
                    input_width,
                    task,
                    weight_decay,
                    pretraining)
        
        #if you want to see the model structure just uncomment model summary.
        #model.summary()
        
        if task in ["segmentation", "binarization"]:
            if is_loss_soft_dice:                    
                loss = soft_dice_loss
            elif weighted_loss:
                loss = weighted_categorical_crossentropy(weights)
            else:
                loss = 'categorical_crossentropy'
        else: # task == "enhancement"
            loss = 'mean_squared_error'
        model.compile(loss=loss,
                      optimizer=Adam(learning_rate=learning_rate),
                      metrics=['accuracy', MeanIoU(n_classes,
                                                   name='iou',
                                                   ignore_class=0,
                                                   sparse_y_true=False,
                                                   sparse_y_pred=False)])
        
        # generating train and evaluation data
        gen_kwargs = dict(batch_size=n_batch,
                          input_height=input_height,
                          input_width=input_width,
                          n_classes=n_classes,
                          task=task)
        train_gen = data_gen(dir_flow_train_imgs, dir_flow_train_labels, **gen_kwargs)
        val_gen = data_gen(dir_flow_eval_imgs, dir_flow_eval_labels, **gen_kwargs)

        ##img_validation_patches = os.listdir(dir_flow_eval_imgs)
        ##score_best=[]
        ##score_best.append(0)
        
        callbacks = [TensorBoard(os.path.join(dir_output, 'logs'), write_graph=False),
                     SaveWeightsAfterSteps(0, dir_output, _config)]
        if save_interval:
            callbacks.append(SaveWeightsAfterSteps(save_interval, dir_output, _config))

        steps_train = len(os.listdir(dir_flow_train_imgs)) // n_batch # - 1
        steps_val = len(os.listdir(dir_flow_eval_imgs)) // n_batch
        _log.info("training on %d batches in %d epochs", steps_train, n_epochs)
        _log.info("validating on %d batches", steps_val)
        model.fit(
            train_gen,
            steps_per_epoch=steps_train,
            validation_data=val_gen,
            #validation_steps=1, # rs: only one batch??
            validation_steps=steps_val,
            epochs=n_epochs,
            callbacks=callbacks,
            initial_epoch=index_start)

        #os.system('rm -rf '+dir_train_flowing)
        #os.system('rm -rf '+dir_eval_flowing)

        #model.save(dir_output+'/'+'model'+'.h5')

    elif task=='classification':
        if continue_training:
            model = load_model(dir_of_start_model, compile=False)
        else:
            index_start = 0
            model = resnet50_classifier(n_classes,
                                        input_height,
                                        input_width,
                                        weight_decay,
                                        pretraining)

        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(learning_rate=0.001), # rs: why not learning_rate?
                      metrics=['accuracy'])
        
        list_classes = list(classification_classes_name.values())
        trainXY = generate_data_from_folder_training(
            dir_train, n_batch, input_height, input_width, n_classes, list_classes)
        testXY = generate_data_from_folder_evaluation(
            dir_eval, input_height, input_width, n_classes, list_classes)

        y_tot = np.zeros((testX.shape[0], n_classes))
        num_rows = return_number_of_total_training_data(dir_train)
        callbacks = [TensorBoard(os.path.join(dir_output, 'logs'), write_graph=False),
                     SaveWeightsAfterSteps(0, dir_output, _config,
                                           monitor='val_f1',
                                           save_best_only=True, mode='max')]
        
        history = model.fit(trainXY,
                            steps_per_epoch=num_rows / n_batch,
                            #class_weight=weights)
                            validation_data=testXY,
                            verbose=1,
                            epochs=n_epochs,
                            metrics=[F1Score(average='macro', name='f1')],
                            callbacks=callbacks,
                            initial_epoch=index_start)

        usable_checkpoints = np.flatnonzero(np.array(history['val_f1']) > f1_threshold_classification)
        if len(usable_checkpoints) >= 1:
            _log.info("averaging over usable checkpoints: %s", str(usable_checkpoints))
            all_weights = []
            for epoch in usable_checkpoints:
                cp_path = os.path.join(dir_output, 'model_{epoch:02d}'.format(epoch=epoch))
                assert os.path.isdir(cp_path)
                model = load_model(cp_path, compile=False)
                all_weights.append(model.get_weights())

            new_weights = []
            for layer_weights in zip(*all_weights):
                layer_weights = np.array([np.array(weights).mean(axis=0)
                                          for weights in zip(*layer_weights)])
                new_weights.append(layer_weights)
                
            #model = tf.keras.models.clone_model(model)
            model.set_weights(new_weights)

            cp_path = os.path.join(dir_output, 'model_ens_avg')
            model.save(cp_path)
            with open(os.path.join(cp_path, "config.json"), "w") as fp:
                json.dump(_config, fp)  # encode dict into JSON
            _log.info("ensemble model saved under '%s'", cp_path)
            
    elif task=='reading_order':
        if continue_training:
            model = load_model(dir_of_start_model, compile=False)
        else:
            index_start = 0
            model = machine_based_reading_order_model(n_classes,
                                                      input_height,
                                                      input_width,
                                                      weight_decay,
                                                      pretraining)
        
        dir_flow_train_imgs = os.path.join(dir_train, 'images')
        dir_flow_train_labels = os.path.join(dir_train, 'labels')
        
        classes = os.listdir(dir_flow_train_labels)
        if augmentation:
            num_rows = len(classes)*(len(thetha) + 1)
        else:
            num_rows = len(classes)
        #ls_test = os.listdir(dir_flow_train_labels)

        #f1score_tot = [0]
        model.compile(loss="binary_crossentropy",
                      #optimizer=SGD(learning_rate=0.01, momentum=0.9),
                      optimizer=Adam(learning_rate=0.0001), # rs: why not learning_rate?
                      metrics=['accuracy'])
        
        callbacks = [TensorBoard(os.path.join(dir_output, 'logs'), write_graph=False),
                     SaveWeightsAfterSteps(0, dir_output, _config)]
        if save_interval:
            callbacks.append(SaveWeightsAfterSteps(save_interval, dir_output, _config))

        trainXY = generate_arrays_from_folder_reading_order(
            dir_flow_train_labels, dir_flow_train_imgs,
            n_batch, input_height, input_width, n_classes,
            thetha, augmentation)

        history = model.fit(trainXY,
                            steps_per_epoch=num_rows / n_batch,
                            verbose=1,
                            epochs=n_epochs,
                            callbacks=callbacks,
                            initial_epoch=index_start)
        '''
        if f1score>f1score_tot[0]:
            f1score_tot[0] = f1score
            model_dir = os.path.join(dir_out,'model_best')
            model.save(model_dir)
        '''

    

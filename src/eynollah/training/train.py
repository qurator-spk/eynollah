import os
import sys
import io
import json

from tqdm import tqdm
import requests

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_USE_LEGACY_KERAS'] = '1' # avoid Keras 3 after TF 2.15
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.metrics import MeanIoU, F1Score
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.layers import StringLookup
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.backend import one_hot
from sacred import Experiment
from sacred.config import create_captured_function

import numpy as np
import cv2
from matplotlib import pyplot as plt # for plot_confusion_matrix

from .metrics import (
    soft_dice_loss,
    weighted_categorical_crossentropy,
    get as get_metric,
    metrics_superposition,
    ConfusionMatrix,
    connected_components_loss,
)
from .models import (
    PatchEncoder,
    Patches,
    machine_based_reading_order_model,
    resnet50_classifier,
    resnet50_unet,
    vit_resnet50_unet,
    vit_resnet50_unet_transformer_before_cnn,
    cnn_rnn_ocr_model,
    RESNET50_WEIGHTS_PATH,
    RESNET50_WEIGHTS_URL
)
from .utils import (
    generate_arrays_from_folder_reading_order,
    get_one_hot,
    preprocess_imgs,
)
from .weights_ensembling import run_ensembling


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
        #tf.keras.mixed_precision.set_global_policy('mixed_float16')
        #tf.keras.backend.set_epsilon(1e-4) # avoid NaN from smaller defaults
    except:
        print("no GPU device available", file=sys.stderr)

def plot_layout_tf(in_: tf.Tensor, out:tf.Tensor) -> tf.Tensor:
    """
    Implements training.inference.SBBPredict.visualize_model_output for TF
    (effectively plotting the layout segmentation map on the input image).

    In doing so, also converts:
    - from Eynollah's BGR/float on the input side
    - to std RGB/int format on the output side
    """
    # in_: [B, H, W, 3] (BGR float)
    image = in_[..., ::-1] * 255
    # out: [B, H, W, C]
    lab = tf.math.argmax(out, axis=-1)
    # lab: [B, H, W]
    colors = tf.constant([[255, 255, 255],
                          [255, 0, 0],
                          [255, 125, 0],
                          [255, 0, 125],
                          [125, 125, 125],
                          [125, 125, 0],
                          [0, 125, 255],
                          [0, 125, 0],
                          [125, 125, 125],
                          [0, 125, 255],
                          [125, 0, 125],
                          [0, 255, 0],
                          [0, 0, 255],
                          [0, 255, 255],
                          [255, 125, 125],
                          [255, 0, 255]])
    layout = tf.gather(colors, lab)
    # layout: [B, H, W, 3]
    image = tf.cast(image, tf.float32)
    layout = tf.cast(layout, tf.float32)
    #weighted = image * 0.5 + layout * 0.1 (too dark)
    weighted = image * 0.9 + layout * 0.1
    return tf.cast(weighted, tf.uint8)

def plot_confusion_matrix(cm, name="Confusion Matrix"):
    """
    Plot the confusion matrix with matplotlib and tensorflow
    """
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    im = ax.imshow(cm, vmin=0.0, vmax=1.0, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xlim=[-0.5, cm.shape[1] - 0.5],
           ylim=[-0.5, cm.shape[0] - 0.5],
           #xticklabels=labels,
           #yticklabels=labels,
           title=name,
           ylabel='True class',
           xlabel='Predicted class')
    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], ".2f"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    # convert to PNG
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image

# plot predictions on train and test set during every epoch
class TensorBoardPlotter(TensorBoard):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_call = None
    def on_epoch_begin(self, epoch, logs=None):
        super().on_epoch_begin(epoch, logs=logs)
        # override the model's call(), so we don't have to invest extra cycles
        # to predict our samples (plotting itself can be neglected)
        self.model_call = self.model.call
        def new_call(inputs, **kwargs):
            outputs = self.model_call(inputs, **kwargs)
            images = plot_layout_tf(inputs, outputs)
            self.plot(images, training=kwargs.get('training', None), epoch=epoch)
            with tf.control_dependencies(None):
                return outputs
        self.model.call = new_call
        # force rebuild of tf.function (so Python binding for epoch gets re-evaluated)
        self.model.train_function = self.model.make_train_function(True)
        self.model.test_function = self.model.make_test_function(True)
    def on_epoch_end(self, epoch, logs=None):
        # re-instate (so ModelCheckpoint does not see our override call)
        self.model.call = self.model_call
        super().on_epoch_end(epoch, logs=logs)
    def plot(self, images, training=None, epoch=0):
        if training:
            writer = self._train_writer
            mode, step = "train", self._train_step.value()
        else:
            writer = self._val_writer
            mode, step = "test", self._val_step.value()
        # skip most samples, because TF's EncodePNG is so costly,
        # and now ends up in the middle of our pipeline, thus causing stalls
        # (cannot use max_outputs, as batch size may be too small)
        if not tf.cast(step % 3, tf.bool):
            with writer.as_default():
                # used to be family kwarg for tf.summary.image name prefix
                family = "epoch_%03d/" % (1 + epoch)
                name = family + mode
                tf.summary.image(name, images, step=step, max_outputs=len(images))
    def on_train_batch_end(self, batch, logs=None):
        if logs is not None:
            logs = dict(logs)
            # cannot be logged as scalar:
            logs.pop('confusion_matrix', None)
        super().on_train_batch_end(batch, logs)
    def on_test_end(self, logs=None):
        if logs is not None:
            logs = dict(logs)
            # cannot be logged as scalar:
            logs.pop('confusion_matrix', None)
        super().on_test_end(logs)
    def _log_epoch_metrics(self, epoch, logs):
        if not logs:
            return
        logs = dict(logs)
        # cannot be logged as scalar:
        train_matrix = logs.pop('confusion_matrix', None)
        val_matrix = logs.pop('val_confusion_matrix', None)
        super()._log_epoch_metrics(epoch, logs)
        # now plot confusion_matrix
        with tf.summary.record_if(True):
            if train_matrix is not None:
                train_image = plot_confusion_matrix(train_matrix)
                with self._train_writer.as_default():
                    tf.summary.image("confusion_matrix", train_image, step=epoch)
            if val_matrix is not None:
                val_image = plot_confusion_matrix(val_matrix)
                with self._val_writer.as_default():
                    tf.summary.image("confusion_matrix", val_image, step=epoch)

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
    if task == 'cnn-rnn-ocr':
        max_len = None # Maximum sequence length (characters per line) for OCR output.
        characters_txt_file = None # Path of JSON file defining character set needed of OCR model.
    input_height = 224 * 1  # Height of model's input in pixels.
    input_width = 224 * 1  # Width of model's input in pixels.
    weight_decay = 1e-6  # Weight decay of l2 regularization of model layers.
    learning_rate = 1e-4  # Set the learning rate.
    if task in ["segmentation", "binarization"]:
        is_loss_soft_dice = False  # Use soft dice as loss function. When set to true, "weighted_loss" must be false.
        weighted_loss = False  # Use weighted categorical cross entropy as loss fucntion. When set to true, "is_loss_soft_dice" must be false.
        add_ncc_loss = 0 # Add regression loss for number of connected components. When non-zero, use this as weight for the nCC term.
    elif task == "classification":
        f1_threshold_classification = None # This threshold is used to consider models with an evaluation f1 scores bigger than it. The selected model weights undergo a weights ensembling. And avreage ensembled model will be written to output.
        classification_classes_name = None # Dictionary of classification classes names.
    patches = False  # Divides input image into smaller patches (input size of the model) when set to true. For the model to see the full image, like page extraction, set this to false.
    augmentation = False  # To apply any kind of augmentation, this parameter must be set to true.
    if augmentation:
        flip_aug = False  # Whether different types of flipping will be applied to the image. Requires "flip_index" setting.
        blur_aug = False  # Whether images will be blurred. Requires "blur_k" setting.
        if blur_aug:
            blur_k = None  # Method of blurring (gauss, median or blur).
        padding_white = False # If true, white padding will be applied to the image.
        if padding_white and task == 'cnn-rnn-ocr':
            white_padds = None # List of padding sizes.
            padd_colors = None # List of padding colors, but only "white" or "black" or both.
        padding_black = False # If true, black padding will be applied to the image.
        scaling = False  # Whether images will be scaled up or down. Requires "scales" setting.
        scaling_bluring = False  # Whether a combination of scaling and blurring will be applied to the image.
        scaling_binarization = False  # Whether a combination of scaling and binarization will be applied to the image.
        scaling_brightness = False  # Whether a combination of scaling and brightening will be applied to the image.
        scaling_flip = False  # Whether a combination of scaling and flipping will be applied to the image.
        if scaling or scaling_brightness or scaling_bluring or scaling_binarization or scaling_flip:
            scales = None  # Scale patches for augmentation.
        if flip_aug or scaling_flip:
            flip_index = None  # List of codes (as in cv2.flip) for flip augmentation.
        shifting = False
        brightening = False  # Whether images will be brightened. Requires "brightness" setting.
        if brightening:
            brightness = None #  List of intensity factors for brightening.
        binarization = False  # Whether binary images will be used, too. (Will use Otsu thresholding unless supplying precomputed images in "dir_img_bin".)
        if binarization:
            dir_img_bin = None # Directory of training dataset subdirectory of binarized images
            add_red_textlines = False
            adding_rgb_background = False # Whether texture images will be added as artificial background.
            if adding_rgb_background:
                dir_rgb_backgrounds = None # Directory of texture images for synthetic background
            adding_rgb_foreground = False # Whether texture images will be added as artificial foreground.
            if adding_rgb_foreground:
                dir_rgb_foregrounds = None # Directory of texture images for synthetic foreground
            if adding_rgb_background or adding_rgb_foreground:
                number_of_backgrounds_per_image = 1
            if task == 'cnn-rnn-ocr':
                image_inversion = False # Whether the binarized images will be inverted.
                textline_skewing_bin = False # Whether binarized textline images will be rotated.
                textline_left_in_depth_bin = False # Whether left side of binary textline image will be displayed in depth.
                textline_right_in_depth_bin = False # Whether right side of binary textline image will be displayed in depth.
                textline_up_in_depth_bin = False # Whether upper side of binary textline image will be displayed in depth.
                textline_down_in_depth_bin = False # Whether lower side of binary textline image will be displayed in depth.
                pepper_bin_aug = False # Whether pepper noise will be added to binary textline images.
                bin_deg = False # Whether a combination of degrading and binarization will be applied to the image.
        degrading = False  # Whether images will be artificially degraded. Requires the "degrade_scales" setting.
        if degrading or binarization and task == 'cnn-rnn-ocr' and bin_deg:
            degrade_scales = None  # List of quality factors for degradation.
        channels_shuffling = False # Re-arrange color channels.
        if channels_shuffling:
            shuffle_indexes = None # List of channels to switch between.
        rotation = False # Whether images will be rotated by 90 degrees.
        rotation_not_90 = False # Whether images will be rotated arbitrarily (skewed). Requires "thetha" setting.
        if rotation_not_90:
            thetha = None  # List of rotation angles in degrees.
        if task == 'cnn-rnn-ocr':
            white_noise_strap = False # Whether white noise will be applied on some straps on the textline image.
            textline_skewing = False # Whether textline images will be skewed for augmentation.
            if textline_skewing or binarization and textline_skewing_bin:
                skewing_amplitudes = None # List of skewing angles in degrees like [5, 8]
            textline_left_in_depth = False # If true, left side of textline image will be displayed in depth.
            textline_right_in_depth = False # If true, right side of textline image will be displayed in depth.
            textline_up_in_depth = False # If true, upper side of textline image will be displayed in depth.
            textline_down_in_depth = False # If true, lower side of textline image will be displayed in depth.
            pepper_aug = False # Whether pepper noise will be added to textline images.
            if pepper_aug or binarization and pepper_bin_aug:
                pepper_indexes = None # List of pepper noise factors, e.g. [0.01, 0.005].
            color_padding_rotation = False # Whether images will be rotated with color padding. Requires "thetha_padd" setting.
            if color_padding_rotation:
                thetha_padd = None # List of angles (in degrees) used for rotation alongside padding.
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
        ## if rotation_not_90
        thetha=None,
        is_loss_soft_dice=False,
        weighted_loss=False,
        add_ncc_loss=None,
        ## if continue_training
        index_start=0,
        dir_of_start_model=None,
        backbone_type=None,
        ## if backbone_type=transformer
        transformer_projection_dim=None,
        transformer_mlp_head_units=None,
        transformer_layers=None,
        transformer_num_heads=None,
        transformer_cnn_first=None,
        transformer_patchsize_x=None,
        transformer_patchsize_y=None,
        transformer_num_patches_xy=None,
        ## if task=classification
        f1_threshold_classification=None,
        classification_classes_name=None,
        ## if task=cnn-rnn-ocr
        characters_txt_file=None,
        color_padding_rotation=False,
        thetha_padd=None,
        bin_deg=False,
        image_inversion=False,
        white_noise_strap=False,
        textline_skewing=False,
        textline_skewing_bin=False,
        textline_left_in_depth=False,
        textline_left_in_depth_bin=False,
        textline_right_in_depth=False,
        textline_right_in_depth_bin=False,
        textline_up_in_depth=False,
        textline_up_in_depth_bin=False,
        textline_down_in_depth=False,
        textline_down_in_depth_bin=False,
        pepper_aug=False,
        pepper_bin_aug=False,
        pepper_indexes=None,
        padd_colors=None,
        white_padds=None,
        skewing_amplitudes=None,
        max_len=None,
):
    """
    run configured experiment via sacred
    """

    if continue_training:
        assert n_epochs > index_start, "with continue_training, n_epochs must be greater than index_start"

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

            # writing patches into a sub-folder in order to be flowed from directory.
            def gen(dir_img, dir_lab, dir_flow_imgs, dir_flow_labs, augmentation=True):
                indexer = 0
                for img, lab in tqdm(preprocess_imgs(_config,
                                                     dir_img,
                                                     dir_lab,
                                                     augmentation=augmentation),
                                     desc="data_is_provided"):
                    fname = 'img_%d.png' % indexer
                    cv2.imwrite(os.path.join(dir_flow_imgs, fname), img)
                    cv2.imwrite(os.path.join(dir_flow_labs, fname), lab)
                    indexer += 1
            gen(*get_dirs_or_files(dir_train),
                dir_flow_train_imgs,
                dir_flow_train_labels)
            gen(*get_dirs_or_files(dir_eval),
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
            assert not weighted_loss, "for enhancement, weighted loss does not apply"
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
            else:
                num_patches_x = transformer_num_patches_xy[0]
                num_patches_y = transformer_num_patches_xy[1]
                num_patches = num_patches_x * num_patches_y

                if transformer_cnn_first:
                    model_builder = vit_resnet50_unet
                    multiple = 32
                else:
                    model_builder = vit_resnet50_unet_transformer_before_cnn
                    multiple = 1

                assert input_height == (
                    num_patches_y * transformer_patchsize_y * multiple), (
                    "transformer_patchsize_y or transformer_num_patches_xy height value error: "
                    "input_height should be equal to "
                    "(transformer_num_patches_xy height value * transformer_patchsize_y * %d)" % multiple)
                assert input_width == (
                    num_patches_x * transformer_patchsize_x * multiple), (
                        "transformer_patchsize_x or transformer_num_patches_xy width value error: "
                        "input_width should be equal to "
                        "(transformer_num_patches_xy width value * transformer_patchsize_x * %d)" % multiple)
                assert 0 == (transformer_projection_dim %
                             (transformer_patchsize_y * transformer_patchsize_x)), (
                                 "transformer_projection_dim error: "
                                 "The remainder when parameter transformer_projection_dim is divided by "
                                 "(transformer_patchsize_y*transformer_patchsize_x) should be zero")

                model_builder = create_captured_function(model_builder)
                model_builder.config = _config
                model_builder.logger = _log
                model = model_builder(num_patches)

        assert model is not None
        #if you want to see the model structure just uncomment model summary.
        #model.summary()

        metrics = ['categorical_accuracy']
        if task in ["segmentation", "binarization"]:
            if is_loss_soft_dice:
                loss = soft_dice_loss
            elif weighted_loss:
                loss = weighted_categorical_crossentropy(weights)
            else:
                loss = get_metric('categorical_crossentropy')
            if add_ncc_loss:
                loss = metrics_superposition(loss, connected_components_loss(n_classes - 1),
                                             weights=[1 - add_ncc_loss, add_ncc_loss])
            metrics.append(connected_components_loss(n_classes - 1))
            metrics.append(MeanIoU(n_classes,
                                   name='iou',
                                   ignore_class=0,
                                   sparse_y_true=False,
                                   sparse_y_pred=False))
            metrics.append(ConfusionMatrix(n_classes))
        else: # task == "enhancement"
            loss = 'mean_squared_error'
        model.compile(loss=loss,
                      #jit_compile=True,
                      optimizer=Adam(learning_rate=learning_rate),
                      metrics=metrics)

        def _to_cv2float(img):
            # rgb→bgr and uint8→float, as expected by Eynollah models
            return tf.cast(tf.reverse(img, [-1]), tf.float32) / 255
        def _to_intrgb(img):
            # bgr→rgb and float→uint8 for plotting
            return tf.reverse(tf.cast(img * 255, tf.uint8), [-1])
        def _to_categorical(seg):
            seg = tf.cast(seg * 255, tf.int8)
            # gt_gen_utils/pagexml2label uses peculiar pseudo-RGB/index colors
            #seg = tf.image.rgb_to_grayscale(seg)
            seg = tf.gather(seg, [0], axis=-1)
            seg = tf.squeeze(seg, axis=-1)
            return one_hot(seg, n_classes)
        def get_dataset(dir_imgs, dir_labs, shuffle=None):
            gen_kwargs = dict(labels=None,
                              label_mode=None,
                              batch_size=None, # batch after zip below
                              image_size=(input_height, input_width),
                              color_mode='rgb',
                              shuffle=shuffle is not None,
                              seed=shuffle,
                              interpolation='nearest',
                              crop_to_aspect_ratio=False,
                              # Keras 3 only...
                              #pad_to_aspect_ratio=False,
                              #data_format='channel_last',
                              #verbose=False,
            )
            img_gen = image_dataset_from_directory(dir_imgs, **gen_kwargs)
            lab_gen = image_dataset_from_directory(dir_labs, **gen_kwargs)
            img_gen = img_gen.map(_to_cv2float, num_parallel_calls=tf.data.AUTOTUNE)
            lab_gen = lab_gen.map(_to_cv2float, num_parallel_calls=tf.data.AUTOTUNE)
            if task in ["segmentation", "binarization"]:
                lab_gen = lab_gen.map(_to_categorical, num_parallel_calls=tf.data.AUTOTUNE)
            ds = tf.data.Dataset.zip(img_gen, lab_gen)
            return ds.batch(n_batch, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)
        train_gen = get_dataset(dir_flow_train_imgs, dir_flow_train_labels, shuffle=np.random.randint(1e6))
        valdn_gen = get_dataset(dir_flow_eval_imgs, dir_flow_eval_labels)
        train_steps = len(os.listdir(dir_flow_train_imgs)) // n_batch
        valdn_steps = len(os.listdir(dir_flow_eval_imgs)) // n_batch
        _log.info("training on %d batches in %d epochs", train_steps, n_epochs)
        _log.info("validating on %d batches", valdn_steps)

        callbacks = [TensorBoardPlotter(os.path.join(dir_output, 'logs'), profile_batch=(10, 20)),
                     SaveWeightsAfterSteps(0, dir_output, _config),
        ]
        if save_interval:
            callbacks.append(SaveWeightsAfterSteps(save_interval, dir_output, _config))
        train_gen = train_gen.shuffle(train_steps // 1000, reshuffle_each_iteration=True)
        valdn_gen = valdn_gen.shuffle(valdn_steps // 10, reshuffle_each_iteration=False)
        model.fit(
            train_gen.prefetch(tf.data.AUTOTUNE),
            steps_per_epoch=train_steps,
            validation_data=valdn_gen.prefetch(tf.data.AUTOTUNE),
            validation_steps=valdn_steps,
            verbose=1,
            epochs=n_epochs,
            callbacks=callbacks,
            initial_epoch=index_start)

    elif task=="cnn-rnn-ocr":

        with open(characters_txt_file, 'r') as char_txt_f:
            characters = json.load(char_txt_f)
        padding_token = len(characters) + 5
        # Mapping characters to integers.
        char_to_num = StringLookup(vocabulary=list(characters), mask_token=None)
        n_classes = len(char_to_num.get_vocabulary()) + 2

        if continue_training:
            model = load_model(dir_of_start_model)
        else:
            index_start = 0
            model = cnn_rnn_ocr_model(image_height=input_height,
                                      image_width=input_width,
                                      n_classes=n_classes,
                                      max_seq=max_len)
        #print(model.summary())

        # todo: use Dataset.map() on Dataset.list_files()
        def get_dataset(dir_img, dir_lab):
            def gen():
                return preprocess_imgs(_config,
                                       dir_img,
                                       dir_lab,
                                       # extra+overrides
                                       char_to_num=char_to_num,
                                       padding_token=padding_token
                )
            return (tf.data.Dataset.from_generator(gen, (tf.float32, tf.int64))
                    .padded_batch(n_batch,
                                  padded_shapes=([input_height, input_width, 3], [None]),
                                  padding_values=(None, tf.constant(padding_token, dtype=tf.int64)),
                                  drop_remainder=True,
                                  #num_parallel_calls=tf.data.AUTOTUNE,
                    )
                    .map(lambda x, y: {"image": x, "label": y})
                    .prefetch(tf.data.AUTOTUNE)
            )
        train_ds = get_dataset(*get_dirs_or_files(dir_train))
        valdn_ds = get_dataset(*get_dirs_or_files(dir_eval))

        #initial_learning_rate = 1e-4
        #decay_steps = int (n_epochs * ( len_dataset / n_batch ))
        #alpha = 0.01
        #lr_schedule = 1e-4
        #tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate, decay_steps, alpha)
        opt = Adam(learning_rate=learning_rate)
        model.compile(optimizer=opt) # rs: loss seems to be (ctc_batch_cost) in last layer

        callbacks = [TensorBoard(os.path.join(dir_output, 'logs'), write_graph=False),
                     EarlyStopping(verbose=1, patience=3, restore_best_weights=False, start_from_epoch=3),
                     SaveWeightsAfterSteps(0, dir_output, _config)]
        if save_interval:
            callbacks.append(SaveWeightsAfterSteps(save_interval, dir_output, _config))
        model.fit(
            train_ds.shuffle(200),
            validation_data=valdn_ds,
            verbose=1,
            epochs=n_epochs,
            callbacks=callbacks,
            initial_epoch=index_start)

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
                      metrics=['accuracy', F1Score(average='macro', name='f1')])

        list_classes = list(classification_classes_name.values())
        data_args = dict(label_mode="categorical",
                         class_names=list_classes,
                         batch_size=n_batch,
                         image_size=(input_height, input_width),
                         interpolation="nearest")
        trainXY = image_dataset_from_directory(dir_train, shuffle=True, **data_args)
        testXY = image_dataset_from_directory(dir_eval, shuffle=False, **data_args)
        callbacks = [TensorBoard(os.path.join(dir_output, 'logs'), write_graph=False),
                     SaveWeightsAfterSteps(0, dir_output, _config,
                                           monitor='val_f1',
                                           #save_best_only=True, # we need all for ensembling
                                           mode='max')]

        history = model.fit(trainXY,
                            #class_weight=weights)
                            validation_data=testXY,
                            verbose=1,
                            epochs=n_epochs,
                            callbacks=callbacks,
                            initial_epoch=index_start)

        usable_checkpoints = np.flatnonzero(np.array(history.history['val_f1']) >
                                            f1_threshold_classification)
        if len(usable_checkpoints) >= 1:
            _log.info("averaging over usable checkpoints: %s", str(usable_checkpoints))
            usable_checkpoints = [os.path.join(dir_output, 'model_{epoch:02d}'.format(epoch=epoch + 1))
                                  for epoch in usable_checkpoints]
            ens_path = os.path.join(dir_output, 'model_ens_avg')
            run_ensembling(usable_checkpoints, ens_path)
            _log.info("ensemble model saved under '%s'", ens_path)

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

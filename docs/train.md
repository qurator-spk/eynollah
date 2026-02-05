# Training documentation

This document aims to assist users in preparing training datasets, training models, and
performing inference with trained models.  We cover various use cases including
pixel-wise segmentation, image classification, image enhancement, and
machine-based reading order detection. For each use case, we provide guidance
on how to generate the corresponding training dataset.

The following three tasks can all be accomplished using the code in the
[`train`](https://github.com/qurator-spk/eynollah/tree/main/train) directory:

* [Generate training dataset](#generate-training-dataset)
* [Train a model](#train-a-model)
* [Inference with the trained model](#inference-with-the-trained-model)

## Training, evaluation and output 

The train and evaluation folders should contain subfolders of `images` and `labels`.

The output folder should be an empty folder where the output model will be written to.

## Generate training dataset

The script `generate_gt_for_training.py` is used for generating training datasets. As the results of the following
command demonstrates, the dataset generator provides several subcommands:

```sh
eynollah-training generate-gt --help
```

The three most important subcommands are:

* image-enhancement
* machine-based-reading-order
* pagexml2label

### image-enhancement

Generating a training dataset for image enhancement is quite straightforward. All that is needed is a set of
high-resolution images. The training dataset can then be generated using the following command:

```sh
eynollah-training image-enhancement \
  -dis "dir of high resolution images" \
  -dois "dir where degraded images will be written" \
  -dols "dir where the corresponding high resolution image will be written as label" \
  -scs "degrading scales json file"
```

The scales JSON file is a dictionary with a key named `scales` and values representing scales smaller than 1. Images are
downscaled based on these scales and then upscaled again to their original size. This process causes the images to lose
resolution at different scales. The degraded images are used as input images, and the original high-resolution images
serve as labels. The enhancement model can be trained with this generated dataset. The scales JSON file looks like this:

```yaml
{
    "scales": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
}
```

### machine-based-reading-order

For machine-based reading order, we aim to determine the reading priority between two sets of text regions. The model's
input is a three-channel image: the first and last channels contain information about each of the two text regions,
while the middle channel encodes prominent layout elements necessary for reading order, such as separators and headers.
To generate the training dataset, our script requires a PAGE XML file that specifies the image layout with the correct
reading order.

For output images, it is necessary to specify the width and height. Additionally, a minimum text region size can be set
to filter out regions smaller than this minimum size. This minimum size is defined as the ratio of the text region area
to the image area, with a default value of zero. To run the dataset generator, use the following command:

```shell
eynollah-training generate-gt machine-based-reading-order \
  -dx "dir of GT xml files" \
  -domi "dir where output images will be written" \
"" -docl "dir where the labels will be written" \
  -ih "height" \
  -iw "width" \
  -min "min area ratio"
```

### pagexml2label

`pagexml2label` is designed to generate labels from PAGE XML GT files for various pixel-wise segmentation use cases,
including:
- `printspace` (i.e. page frame),
- `layout` (i.e. regions),
- `textline`,
- `word`, and
- `glyph`.

To train a pixel-wise segmentation model, we require images along with their corresponding labels. Our training script
expects a PNG image where each pixel corresponds to a label, represented by an integer. The background is always labeled
as zero, while other elements are assigned different integers. For instance, if we have ground truth data with four
elements including the background, the classes would be labeled as 0, 1, 2, and 3 respectively.

In binary segmentation scenarios such as textline or page extraction, the background is encoded as 0, and the desired
element is automatically encoded as 1 in the PNG label.

To specify the desired use case and the elements to be extracted in the PNG labels, a custom JSON file can be passed.
For example, in the case of textline detection, the JSON contents could be this:

```yaml
{
"use_case": "textline"
}
```

In the case of layout segmentation, the config JSON file might look like this:

```yaml
{
"use_case": "layout",
"textregions": {"rest_as_paragraph": 1, "drop-capital": 1, "header": 2, "heading": 2, "marginalia": 3},
"imageregion": 4,
"separatorregion": 5,
"graphicregions": {"rest_as_decoration": 6, "stamp": 7}
}
```

The same example if `PrintSpace` (or `Border`) should be represented as a unique class:

```yaml
{
"use_case": "layout",
"textregions": {"rest_as_paragraph": 1, "drop-capital": 1, "header": 2, "heading": 2, "marginalia": 3},
"imageregion": 4,
"separatorregion": 5,
"graphicregions": {"rest_as_decoration": 6, "stamp": 7}
"printspace_as_class_in_layout": 8
}
```

In the `layout` use-case, it is beneficial to first understand the structure of the PAGE XML file and its elements.
For a given page image, the visible segments are annotated in XML with their polygon coordinates and types.
On the region level, available segment types include `TextRegion`, `SeparatorRegion`, `ImageRegion`, `GraphicRegion`,
`NoiseRegion` and `TableRegion`.

Moreover, text regions and graphic regions in particular are subdivided via `@type`:
- The allowed subtypes for text regions are `paragraph`, `heading`, `marginalia`, `drop-capital`, `header`, `footnote`,
`footnote-continued`, `signature-mark`, `page-number` and `catch-word`. 
- The known subtypes for graphic regions are `handwritten-annotation`, `decoration`, `stamp` and `signature`.

These types and subtypes must be mapped to classes for the segmentation model. However, sometimes these fine-grained
distinctions are not useful or the existing annotations are not very usable (too scarce or too unreliable). 
In that case, instead of these subtypes with a specific mapping, they can be pooled together by using the two special
types:
- `rest_as_paragraph` (mapping missing TextRegion subtypes and `paragraph`)
- `rest_as_decoration` (mapping missing GraphicRegion subtypes and `decoration`)

(That way, users can extract all known types from the labels and be confident that no subtypes are overlooked.)

In the custom JSON example shown above, `header` and `heading` are extracted as the same class, 
while `marginalia` is modelled as a different class. All other text region types, including `drop-capital`,
are grouped into the same class. For graphic regions, `stamp` has its own class, while all other types
are classified together. `ImageRegion` and `SeparatorRegion` will also represented with a class label in the
training data. However, other regions like `NoiseRegion` or `TableRegion` will not be included in the PNG files,
even if they were present in the PAGE XML.

The tool expects various command-line options:

```sh
eynollah-training generate-gt pagexml2label \
  -dx "dir of input PAGE XML files" \
  -do "dir of output label PNG files" \
  -cfg "custom config JSON file" \
  -to "output type (2d or 3d)"
```

As output type, use
- `2d` for training,
- `3d` to just visualise the labels.

We have also defined an artificial class that can be added to (rendered around) the boundary
of text region types or text lines in order to make separation of neighbouring segments more
reliable. The key is called `artificial_class_on_boundary`, and it takes a list of text region
types to be applied to.

Our example JSON config file could then look like this:

```yaml
{
    "use_case": "layout",
    "textregions": {
        "paragraph": 1,
        "drop-capital": 1,
        "header": 2,
        "heading": 2,
        "marginalia": 3
    },
    "imageregion": 4,
    "separatorregion": 5,
    "graphicregions": {
        "rest_as_decoration": 6
    },
    "artificial_class_on_boundary": ["paragraph", "header", "heading", "marginalia"],
    "artificial_class_label": 7
}
```

This implies that the artificial class label (denoted by 7) will be present in the generated PNG files
and will only be added around segments labeled `paragraph`, `header`, `heading` or `marginalia`. (This
class will be handled specially during decoding at inference, and not show up in final results.)

For `printspace`, `textline`, `word`, and `glyph` segmentation use-cases, there is no `artificial_class_on_boundary` key,
but `artificial_class_label` is available. If specified in the config file, then its value should be set at 2, because
these elements represent binary classification problems (with background represented as 0, and segments as 1, respectively).

For example, the JSON config for textline detection could look as follows:

```yaml
{
    "use_case": "textline",
    "artificial_class_label": 2
}
```

If the coordinates of `PrintSpace` (or `Border`) are present in the PAGE XML ground truth files,
and one wishes to crop images to only cover the print space bounding box, this can be achieved
by passing the `-ps` option. Note that in this scenario, the directory of the original images
must also be provided, to ensure that the images are cropped in sync with the labels. The command
line would then resemble this:

```sh
eynollah-training generate-gt pagexml2label \
  -dx "dir of input PAGE XML files" \
  -do "dir of output label PNG files" \
  -cfg "custom config JSON file" \
  -to "output type (2d or 3d)" \
  -ps \
  -di "dir of input original images" \
  -doi "dir of output cropped images"
```

## Train a model

### classification

For the image classification use-case, we have not provided a ground truth generator, as it is unnecessary.
All we require is a training directory with subdirectories, each containing images of its respective classes. We need
separate directories for training and evaluation, and the class names (subdirectories) must be consistent across both
directories. Additionally, the class names should be specified in the config JSON file, as shown in the following
example. If, for instance, we aim to classify "apple" and "orange," with a total of 2 classes, the
`classification_classes_name` key in the config file should appear as follows:

```yaml
{
    "backbone_type" : "nontransformer",
    "task": "classification",
    "n_classes" : 2,
    "n_epochs" : 10,
    "input_height" : 448,
    "input_width" : 448,
    "weight_decay" : 1e-6,
    "n_batch" : 4,
    "learning_rate": 1e-4,
    "f1_threshold_classification": 0.8,
    "pretraining" : true,
    "classification_classes_name" : {"0":"apple",  "1":"orange"},
    "dir_train": "./train",
    "dir_eval": "./eval",
    "dir_output": "./output"
}
```

Then `dir_train` should be like this:

```
.
└── train             # train directory
   ├── apple          # directory of images for apple class
   └── orange         # directory of images for orange class
```

And `dir_eval` analogously:

```
.
└── eval              # evaluation directory
   ├── apple          # directory of images for apple class
   └── orange         # directory of images for orange class

```

The classification model can be trained using the following command line:

```sh
eynollah-training train with config_classification.json
```

As evident in the example JSON file above, for classification, we utilize a "f1_threshold_classification" parameter.
This parameter is employed to gather all models with an evaluation f1 score surpassing this threshold. Subsequently,
an ensemble of these model weights is executed, and a model is saved in the output directory as "model_ens_avg".
Additionally, the weight of the best model based on the evaluation f1 score is saved as "model_best".

### reading order
An example config json file for machine based reading order should be like this:

```yaml
{
    "backbone_type" : "nontransformer",
    "task": "reading_order",
    "n_classes" : 1,
    "n_epochs" : 5,
    "input_height" : 672,
    "input_width" : 448,
    "weight_decay" : 1e-6,
    "n_batch" : 4,
    "learning_rate": 1e-4,
    "pretraining" : true,
    "dir_train": "./train",
    "dir_eval": "./eval",
    "dir_output": "./output"
}
```

The "dir_train" should be like this:

```
.
└── train             # train directory
   ├── images          # directory of images
   └── labels         # directory of labels
```

And the "dir_eval" the same structure as train directory:

```
.
└── eval             # evaluation directory
   ├── images          # directory of images
   └── labels         # directory of labels
```

The reading-order model can be trained like the classification case command line.

### Segmentation (Textline, Binarization, Page extraction and layout) and enhancement

#### Parameter configuration for segmentation or enhancement usecases

The following parameter configuration can be applied to all segmentation use cases and enhancements. The augmentation,
its sub-parameters, and continued training are defined only for segmentation use cases and enhancements, not for
classification and machine-based reading order, as you can see in their example config files.

* `task`: The task parameter must be one of the following values:
  - `binarization`,
  - `enhancement`,
  - `segmentation`,
  - `classification`,
  - `reading_order`.
* `backbone_type`: For the tasks `segmentation` (such as text line, and region layout detection),
  `binarization` and `enhancement`, we offer two backbone options:
  - `nontransformer` (only a CNN ResNet-50).
  - `transformer` (first apply a CNN, followed by a transformer)
* `transformer_cnn_first`: Whether to apply the CNN first (followed by the transformer) when using `transformer` backbone.
* `transformer_num_patches_xy`: Number of patches for vision transformer in x and y direction respectively.
* `transformer_patchsize_x`: Patch size of vision transformer patches in x direction.
* `transformer_patchsize_y`: Patch size of vision transformer patches in y direction.
* `transformer_projection_dim`: Transformer projection dimension. Default value is 64.
* `transformer_mlp_head_units`: Transformer Multilayer Perceptron (MLP) head units. Default value is [128, 64].
* `transformer_layers`: transformer layers. Default value is 8.
* `transformer_num_heads`: Transformer number of heads. Default value is 4.
* `patches`: Whether to break up (tile) input images into smaller patches (input size of the model).
  If `false`, the model will see the image once (resized to the input size of the model).  
  Should be set to `false` for cases like page extraction.
* `n_batch`: Number of batches at each iteration.
* `n_classes`: Number of classes. In the case of binary classification this should be 2. In the case of reading_order it
  should set to 1. And for the case of layout detection just the unique number of classes should be given.
* `n_epochs`: Number of epochs (iterations over the data) to train.
* `input_height`: the image height for the model's input.
* `input_width`: the image width for the model's input.
* `weight_decay`: Weight decay of l2 regularization of model layers.
* `weighted_loss`: If `true`, this means that you want to apply weighted categorical crossentropy as loss function.  
   (Mutually exclusive with `is_loss_soft_dice`, and only applies for `segmentation` and `binarization` tasks.)
* `pretraining`: Set to `true` to (download and) initialise pretrained weights of ResNet50 encoder.
* `dir_train`: Path to directory of raw training data (as extracted via `pagexml2labels`, i.e. with subdirectories
  `images` and `labels` for input images and output labels.  
   (These are not prepared for training the model, yet. Upon first run, the raw data will be transformed to suitable size
    needed for the model, and written in `dir_output` under `train` and `eval` subdirectories. See `data_is_provided`.)
* `dir_eval`: Ditto for raw evaluation data.
* `dir_output`: Directory to write model checkpoints, logs (for Tensorboard) and precomputed images to.
* `data_is_provided`: If you have already trained at least one complete epoch (using the same data settings) before,
  you can set  this to `true` to avoid computing the resized / patched / augmented image files again.  
  Be sure that there are subdirectories `train` and `eval` data are in `dir_output` (each with subdirectories `images`
  and `labels`, respectively).
* `continue_training`: If `true`, continue training a model checkpoint from a previous run.  
  This requires providing the directory of the model checkpoint to load via `dir_of_start_model`
  and setting `index_start` counter for naming new checkpoints.
  For example if you have already trained for 3 epochs, then your last index is 2, so if you want
  to continue with `model_04`, `model_05` etc., set `index_start=3`.
* `index_start`: Starting index for saving models in the case that `continue_training` is `true`.  
   (Existing checkpoints above this will be overwritten.)
* `dir_of_start_model`: Directory containing existing model checkpoint to initialise model weights from when `continue_training=true`.  
   (Can be an epoch-interval checkpoint, or batch-interval checkpoint from `save_interval`.)
* `augmentation`: If you want to apply any kind of augmentation this parameter should first set to `true`.  
   The remaining settings pertain to that...
* `flip_aug`: If `true`, different types of flipping over the image arrays. Requires `flip_index` parameter.
* `flip_index`: List of flip codes (as in `cv2.flip`, i.e. 0 for vertical, positive for horizontal shift, negative for vertical and horizontal shift).
* `blur_aug`: If `true`, different types of blurring will be applied on image. Requires `blur_k` parameter.
* `blur_k`: Method of blurring (`gauss`, `median` or `blur`).
* `scaling`: If `true`, scaling will be applied on image. Requires `scales` parameter.
* `scales`: List of scale factors for scaling.
* `scaling_bluring`: If `true`, combination of scaling and blurring will be applied on image.
* `scaling_binarization`: If `true`, combination of scaling and binarization will be applied on image.
* `scaling_flip`: If `true`, combination of scaling and flip will be applied on image.
* `degrading`: If `true`, degrading will be applied to the image. Requires `degrade_scales` parameter.
* `degrade_scales`: List of intensity factors for degrading.
* `brightening`: If `true`, brightening will be applied to the image. Requires `brightness` parameter.
* `brightness`: List of intensity factors for brightening.
* `binarization`: If `true`, Otsu thresholding will be applied to augment the input data with binarized images.
* `dir_img_bin`: With `binarization`, use this directory to read precomputed binarized images instead of ad-hoc Otsu.  
   (Base names should correspond to the files in `dir_train/images`.)
* `rotation`: If `true`, 90° rotation will be applied on images.
* `rotation_not_90`: If `true`, random rotation (other than 90°) will be applied on image. Requires `thetha` parameter.
* `thetha`: List of rotation angles (in degrees).

In case of segmentation and enhancement the train and evaluation data should be organised as follows.

The "dir_train" directory should be like this:

```
.
└── train             # train directory
   ├── images          # directory of images
   └── labels         # directory of labels
```

And the "dir_eval" the same structure as train directory:

```
.
└── eval             # evaluation directory
   ├── images          # directory of images
   └── labels         # directory of labels
```

After configuring the JSON file for segmentation or enhancement,
training can be initiated by running the following command line,
similar to classification and reading-order model training:

```sh
eynollah-training train with config_classification.json
```

#### Binarization

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
    

An example config json file for binarization can be like this:

```yaml
{
    "backbone_type" : "transformer",
    "task": "binarization",
    "n_classes" : 2,
    "n_epochs" : 4,
    "input_height" : 224,
    "input_width" : 672,
    "weight_decay" : 1e-6,
    "n_batch" : 1,
    "learning_rate": 1e-4,
    "patches" : true,
    "pretraining" : true,
    "augmentation" : true,
    "flip_aug" : false,
    "blur_aug" : false,
    "scaling" : true,
    "degrading": false,
    "brightening": false,
    "binarization" : false,
    "scaling_bluring" : false,
    "scaling_binarization" : false,
    "scaling_flip" : false,
    "rotation": false,
    "rotation_not_90": false,
    "transformer_num_patches_xy": [7, 7],
    "transformer_patchsize_x": 3,
    "transformer_patchsize_y": 1,
    "transformer_projection_dim": 192,
    "transformer_mlp_head_units": [128, 64],
    "transformer_layers": 8,
    "transformer_num_heads": 4,
    "transformer_cnn_first": true,
    "blur_k" : ["blur","guass","median"],
    "scales" : [0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.4],
    "brightness" : [1.3, 1.5, 1.7, 2],
    "degrade_scales" : [0.2, 0.4],
    "flip_index" : [0, 1, -1],
    "thetha" : [10, -10],
    "continue_training": false,
    "index_start" : 0,
    "dir_of_start_model" : " ",
    "weighted_loss": false,
    "is_loss_soft_dice": false,
    "data_is_provided": false,
    "dir_train": "./train",
    "dir_eval": "./eval",
    "dir_output": "./output"
}
```

#### Textline

```yaml
{
    "backbone_type" : "nontransformer",
    "task": "segmentation",
    "n_classes" : 2,
    "n_epochs" : 4,
    "input_height" : 448,
    "input_width" : 224,
    "weight_decay" : 1e-6,
    "n_batch" : 1,
    "learning_rate": 1e-4,
    "patches" : true,
    "pretraining" : true,
    "augmentation" : true,
    "flip_aug" : false,
    "blur_aug" : false,
    "scaling" : true,
    "degrading": false,
    "brightening": false,
    "binarization" : false,
    "scaling_bluring" : false,
    "scaling_binarization" : false,
    "scaling_flip" : false,
    "rotation": false,
    "rotation_not_90": false,
    "blur_k" : ["blur","guass","median"],
    "scales" : [0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.4],
    "brightness" : [1.3, 1.5, 1.7, 2],
    "degrade_scales" : [0.2, 0.4],
    "flip_index" : [0, 1, -1],
    "thetha" : [10, -10],
    "continue_training": false,
    "index_start" : 0,
    "dir_of_start_model" : " ",
    "weighted_loss": false,
    "is_loss_soft_dice": false,
    "data_is_provided": false,
    "dir_train": "./train",
    "dir_eval": "./eval",
    "dir_output": "./output"
}
```

#### Enhancement

```yaml
{
    "backbone_type" : "nontransformer",
    "task": "enhancement",
    "n_classes" : 3,
    "n_epochs" : 4,
    "input_height" : 448,
    "input_width" : 224,
    "weight_decay" : 1e-6,
    "n_batch" : 4,
    "learning_rate": 1e-4,
    "patches" : true,
    "pretraining" : true,
    "augmentation" : true,
    "flip_aug" : false,
    "blur_aug" : false,
    "scaling" : true,
    "degrading": false,
    "brightening": false,
    "binarization" : false,
    "scaling_bluring" : false,
    "scaling_binarization" : false,
    "scaling_flip" : false,
    "rotation": false,
    "rotation_not_90": false,
    "blur_k" : ["blur","guass","median"],
    "scales" : [0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.4],
    "brightness" : [1.3, 1.5, 1.7, 2],
    "degrade_scales" : [0.2, 0.4],
    "flip_index" : [0, 1, -1],
    "thetha" : [10, -10],
    "continue_training": false,
    "index_start" : 0,
    "dir_of_start_model" : " ",
    "weighted_loss": false,
    "is_loss_soft_dice": false,
    "data_is_provided": false,
    "dir_train": "./train",
    "dir_eval": "./eval",
    "dir_output": "./output"
}
```

It's important to mention that the value of n_classes for enhancement should be 3, as the model's output is a 3-channel
image.

#### Page extraction

```yaml
{
    "backbone_type" : "nontransformer",
    "task": "segmentation",
    "n_classes" : 2,
    "n_epochs" : 4,
    "input_height" : 448,
    "input_width" : 224,
    "weight_decay" : 1e-6,
    "n_batch" : 1,
    "learning_rate": 1e-4,
    "patches" : false,
    "pretraining" : true,
    "augmentation" : false,
    "flip_aug" : false,
    "blur_aug" : false,
    "scaling" : true,
    "degrading": false,
    "brightening": false,
    "binarization" : false,
    "scaling_bluring" : false,
    "scaling_binarization" : false,
    "scaling_flip" : false,
    "rotation": false,
    "rotation_not_90": false,
    "blur_k" : ["blur","guass","median"],
    "scales" : [0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.4],
    "brightness" : [1.3, 1.5, 1.7, 2],
    "degrade_scales" : [0.2, 0.4],
    "flip_index" : [0, 1, -1],
    "thetha" : [10, -10],
    "continue_training": false,
    "index_start" : 0,
    "dir_of_start_model" : " ",
    "weighted_loss": false,
    "is_loss_soft_dice": false,
    "data_is_provided": false,
    "dir_train": "./train",
    "dir_eval": "./eval",
    "dir_output": "./output"
}
```

For page segmentation (or print space or border segmentation), the model needs to view the input image in its
entirety,hence the patches parameter should be set to false.

#### layout segmentation

An example config json file for layout segmentation with 5 classes (including background) can be like this:

```yaml
{
    "backbone_type" : "transformer",
    "task": "segmentation",
    "n_classes" : 5,
    "n_epochs" : 4,
    "input_height" : 448,
    "input_width" : 224,
    "weight_decay" : 1e-6,
    "n_batch" : 1,
    "learning_rate": 1e-4,
    "patches" : true,
    "pretraining" : true,
    "augmentation" : true,
    "flip_aug" : false,
    "blur_aug" : false,
    "scaling" : true,
    "degrading": false,
    "brightening": false,
    "binarization" : false,
    "scaling_bluring" : false,
    "scaling_binarization" : false,
    "scaling_flip" : false,
    "rotation": false,
    "rotation_not_90": false,
    "transformer_num_patches_xy": [7, 14],
    "transformer_patchsize_x": 1,
    "transformer_patchsize_y": 1,
    "transformer_projection_dim": 64,
    "transformer_mlp_head_units": [128, 64],
    "transformer_layers": 8,
    "transformer_num_heads": 4,
    "transformer_cnn_first": true,
    "blur_k" : ["blur","guass","median"],
    "scales" : [0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.4],
    "brightness" : [1.3, 1.5, 1.7, 2],
    "degrade_scales" : [0.2, 0.4],
    "flip_index" : [0, 1, -1],
    "thetha" : [10, -10],
    "continue_training": false,
    "index_start" : 0,
    "dir_of_start_model" : " ",
    "weighted_loss": false,
    "is_loss_soft_dice": false,
    "data_is_provided": false,
    "dir_train": "./train",
    "dir_eval": "./eval",
    "dir_output": "./output"
}
```
## Inference with the trained model

### classification

For conducting inference with a trained model, you simply need to execute the following command line, specifying the
directory of the model and the image on which to perform inference:

```sh
eynollah-training inference -m "model dir" -i "image"
```

This will straightforwardly return the class of the image.

### machine based reading order

To infer the reading order using a reading order model, we need a PAGE XML file containing layout information but
without the reading order. We simply need to provide the model directory, the XML file, and the output directory. The
new XML file with the added reading order will be written to the output directory with the same name. We need to run:

```sh
eynollah-training inference \
  -m "model dir" \
  -xml "page xml file" \
  -o "output dir to write new xml with reading order"
```

### Segmentation (Textline, Binarization, Page extraction and layout) and enhancement

For conducting inference with a trained model for segmentation and enhancement you need to run the following command line:

```sh
eynollah-training inference \
  -m "model dir" \
  -i "image" \
  -p \
  -s "output image"
```

Note that in the case of page extraction the -p flag is not needed.

For segmentation or binarization tasks, if a ground truth (GT) label is available, the IoU evaluation metric can be
calculated for the output. To do this, you need to provide the GT label using the argument -gt.

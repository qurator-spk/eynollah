# Training Eynollah Models

This guide describes how to prepare training datasets, train and fine-tune models, evaluate training results, and deploy custom models with [Eynollah](https://github.com/qurator-spk/eynollah).

Eynollah provides a collection of deep learning models for document image processing, including document layout analysis, textline detection, binarization, image enhancement, reading order detection, and optical character recognition (OCR).

Model training is managed through the `eynollah-training` command-line interface. Depending on the task, the training workflow uses either TensorFlow/Keras or PyTorch-based models.

## 1. Overview

### 1.1 Supported training tasks

The training framework covers the following tasks:

| Task | Description | Typical training data |
|---|---|---|
| Layout segmentation | Detection and classification of document regions | Document images and pixel-wise labels |
| Text-line segmentation | Detection of text lines | Document images and text-line labels |
| Page extraction | Detection of page boundaries and print space | Document images and page-boundary labels |
| Binarization | Separation of foreground and background pixels | Document images and binary labels |
| Image enhancement | Restoration of degraded or low-resolution images | Degraded and reference image pairs |
| Image classification | Classification of document images | Images organized by class |
| Reading-order detection | Prediction of reading-order relationships | PAGE-XML annotations with reading order |
| OCR | Recognition of text from document images | Text-line images and transcription data |

The availability and maturity of individual training workflows depend on the model architecture and software version. In particular, TrOCR training integration was introduced in v0.9.0 and was still described as untested in that release.

### 1.2 General training workflow

Training a model generally involves the following steps:

1. Select the task and model architecture.
2. Prepare training and validation datasets.
3. Generate pixel-wise labels or other task-specific ground truth, if necessary.
4. Configure the training parameters.
5. Initialize a model, optionally using pretrained weights.
6. Train the model and monitor its performance.
7. Evaluate the resulting checkpoints.
8. Select and, where appropriate, ensemble model weights.
9. Convert the model into a supported inference format.
10. Test and integrate the model into Eynollah.

### 1.3 Training models versus inference models

An important distinction is that the model used during training is not necessarily the model format used during inference.

Eynollah distributes separate training and inference model packages.

- **Training models:** Original framework models and weights used for training or fine-tuning.
- **Inference models:** Optimized models used for prediction, including ONNX models.

Since v0.9.0, ONNX/TensorRT is the default inference backend for the relevant model types. Consequently, a trained model may need to be converted before deployment.

See [Models documentation](models.md) for information about available pretrained models and their intended applications.

## 2. Installation and requirements

### 2.1 Install Eynollah

Clone the repository and install Eynollah with its training dependencies:

```bash
git clone https://github.com/qurator-spk/eynollah.git
cd eynollah
pip install '.[training]'
```

For development installations, consult the repository installation instructions.

### 2.2 Install and obtain pretrained models

Pretrained weights can be used to initialize a model before fine-tuning.

Download the appropriate training model package from the official [Eynollah model distribution](https://zenodo.org/records/17727267).

Use the training model package rather than the ONNX inference package when you intend to continue training or fine-tune a model.

The model documentation describes the available models, their architectures, and their intended applications.

### 2.3 Check the training interface

The `eynollah-training` command provides the main interface for training-related operations.

To inspect the available commands and their arguments, use:

```bash
eynollah-training --help
```

For command-specific options, use the corresponding `--help` option.

## 3. Preparing training datasets

Training and validation datasets must be prepared separately. Their organization depends on the task.

### 3.1 General directory structure

For image segmentation and enhancement tasks, the expected input structure is:

```text
dataset/
├── train/
│   ├── images/
│   └── labels/
└── eval/
    ├── images/
    └── labels/
```

The image and label files must correspond to each other. Their dimensions, naming conventions, and annotation formats must be consistent with the selected training task.

For classification, use one subdirectory per class instead:

```text
dataset/
├── train/
│   ├── class_1/
│   └── class_2/
└── eval/
    ├── class_1/
    └── class_2/
```

### 3.2 PAGE-XML as ground truth

PAGE-XML is an important source of ground-truth annotations for Eynollah.

Depending on the task, PAGE-XML annotations can be converted into pixel-wise segmentation maps or used to generate training examples for reading-order detection.

Before generating labels, check that:

- PAGE-XML files reference the correct source images.
- Coordinates correspond to the original image dimensions.
- Region and text-line annotations are sufficiently complete.
- Region types and reading-order information are consistent.
- Training and validation datasets do not contain duplicate pages.

### 3.3 Generating pixel-wise labels

The `pagexml2label` functionality generates PNG labels from PAGE-XML files.

Example:

```bash
eynollah-training generate-gt pagexml2label \
    -dx /path/to/pagexml \
    -do /path/to/labels \
    -cfg config.json \
    -to 2d
```

Use `2d` for training labels and `3d` for visualizing the generated annotations.

The configuration file determines which PAGE-XML elements are included and how they are mapped to segmentation classes.

#### Layout segmentation

For layout analysis, the configuration maps PAGE-XML region types to integer class labels.

Example:

```json
{
    "use_case": "layout",
    "textregions": {
        "rest_as_paragraph": 1,
        "header": 2,
        "heading": 2,
        "marginalia": 3
    },
    "imageregion": 4,
    "separatorregion": 5,
    "graphicregions": {
        "rest_as_decoration": 6
    }
}
```

The background is represented by class `0`. The other classes are assigned consecutive integer values.

The special mapping `rest_as_paragraph` groups otherwise unspecified text-region types with paragraph regions. Similarly, `rest_as_decoration` groups otherwise unspecified graphic-region types with decorations.

#### Artificial boundary classes

Artificial classes can be used to improve the separation of neighboring regions or text lines.

For example:

```json
{
    "use_case": "layout",
    "textregions": {
        "paragraph": 1,
        "header": 2,
        "heading": 2,
        "marginalia": 3
    },
    "imageregion": 4,
    "separatorregion": 5,
    "artificial_class_on_boundary": [
        "paragraph",
        "header",
        "heading",
        "marginalia"
    ],
    "artificial_class_label": 6
}
```

The artificial class is used internally during segmentation and is handled specially during inference.

#### Page extraction and print-space segmentation

For page extraction, the model must consider the complete page image. The corresponding training configuration should therefore use `patches: false`.

When cropping training images to the print space, the source images and labels must be cropped consistently.

Consult the `pagexml2label` command help for the options controlling print-space extraction and handling of missing annotations.

### 3.4 Machine-based reading-order dataset

Reading-order training uses PAGE-XML files containing the correct reading order.

The dataset generator creates examples from relationships between text regions. Its inputs include the PAGE-XML files and the dimensions of the generated training images.

Use:

```bash
eynollah-training generate-gt machine-based-reading-order --help
```

The generated training and evaluation datasets must contain corresponding images and labels.

### 3.5 Image-enhancement dataset

Image-enhancement training uses pairs of degraded and reference images.

A dataset can be generated from high-resolution source images by applying different degradation scales.

The original high-resolution images serve as labels, while the degraded images are used as inputs.

Use:

```bash
eynollah-training generate-gt image-enhancement --help
```

### 3.6 OCR training data

OCR training requires text-line images and their corresponding transcriptions.

The training data preparation process depends on the OCR architecture:

- CNN-RNN models use the Keras-based training implementation.
- TrOCR uses the corresponding Transformer-based implementation.

Consult the OCR training implementation and model-specific configuration examples for the required data format.

## 4. Configuring models

Training parameters are defined in configuration files.

The parameters depend on the task and model architecture. Common configuration categories include:

| Category | Examples |
|---|---|
| Task | Segmentation, binarization, enhancement, classification, reading order |
| Architecture | CNN-based or Transformer-based backbone |
| Input | Image height, width and number of classes |
| Optimization | Learning rate, batch size and weight decay |
| Training | Number of epochs and checkpoint settings |
| Initialization | Pretrained weights and starting checkpoint |
| Augmentation | Scaling, rotation, flipping, blurring and degradation |
| Loss | Weighted loss and soft Dice options |
| Data | Training and evaluation directories |
| Output | Checkpoints, logs and intermediate data |

### 4.1 Model architectures

For segmentation, binarization and enhancement, the training framework supports CNN-based and Transformer-based architectures.

The configuration parameter `backbone_type` determines the architecture:

- `nontransformer`: CNN-based architecture.
- `transformer`: CNN and Transformer-based architecture.

Transformer-specific parameters include the number of patches, patch dimensions, projection dimensions, attention heads and Transformer layers.

### 4.2 Data augmentation

Augmentation can improve model robustness when the training data contains sufficient variation.

Available augmentation options include:

- Image flipping.
- Scaling.
- Blurring.
- Brightness modification.
- Image degradation.
- Binarization.
- Rotation.

The selected transformations should reflect realistic variations in the target document collection.

### 4.3 Loss functions

Loss-function configuration depends on the task.

The segmentation framework includes weighted categorical cross-entropy and soft Dice-related options. These settings should be configured according to the class distribution and segmentation objective.

Since v0.8.0, connected-component-based loss calculation has also been introduced to improve training.

## 5. Training models

### 5.1 General training procedure

Once the training and evaluation datasets have been prepared and the configuration file has been created, start training using:

```bash
eynollah-training train with config.json
```

The training output directory stores model checkpoints, training logs and, where applicable, preprocessed training data.

### 5.2 Fine-tuning pretrained models

Fine-tuning allows an existing model to be adapted to a new document collection.

The training configuration must identify the appropriate starting model and enable continuation of training where supported.

Check the model architecture and configuration compatibility before loading pretrained weights.

### 5.3 Segmentation and image-processing models

The segmentation training workflow covers:

- Layout segmentation.
- Text-line segmentation.
- Page extraction.
- Binarization.
- Image enhancement.

These tasks share several configuration parameters but differ in label structure, number of classes, image dimensions and augmentation requirements.

### 5.4 Classification and reading-order models

Classification datasets are organized by class directories.

Reading-order models use generated image and label pairs derived from PAGE-XML annotations.

These tasks have configuration requirements that differ from pixel-wise segmentation. Therefore, task-specific configuration examples should be maintained separately.

### 5.5 OCR models

CNN-RNN OCR training and TrOCR training are separate workflows.

They differ in architecture, data handling, optimization and inference conversion.

The CNN-RNN implementation is based on Keras and uses CTC-based recognition. TrOCR is based on a Transformer architecture.

TrOCR training support was introduced in v0.9.0 and should be considered experimental until the implementation and its end-to-end workflow have been validated.

## 6. Monitoring and evaluating training

### 6.1 Checkpoints

Training generates model checkpoints at defined intervals.

Checkpoints should be evaluated on the validation dataset rather than selected solely on training loss.

Retain the best-performing checkpoint and, where supported, compare it with the ensemble model.

### 6.2 TensorBoard

Since v0.8.0, training integrates with TensorBoard.

TensorBoard provides access to training progress, plots and intermediate evaluation visualizations.

The training output directory contains the relevant log data.

### 6.3 Evaluation metrics

Evaluation should use metrics appropriate to the task:

- Segmentation: Intersection over Union (IoU) and related pixel-wise metrics.
- Classification: F1 score and classification accuracy.
- OCR: Recognition accuracy and error rates.
- Reading order: Correctness of predicted reading-order relationships.
- Image enhancement: Image-quality measures and visual assessment.

For segmentation and binarization, the standalone inference interface supports comparison against ground-truth labels where applicable.

## 7. Model conversion and export

A trained model may need conversion before it can be used with the optimized Eynollah inference pipeline.

Since v0.9.0, model conversion is provided through:

```bash
eynollah-training convert --help
```

The conversion functionality supports several formats, including:

- Keras HDF5.
- Native Keras format.
- TensorFlow SavedModel.
- TensorFlow Serving export.
- ONNX.

Conversion requirements depend on the model architecture and supported operations.

Always test the converted model against the original training model to ensure that its predictions remain consistent.

## 8. Using trained models

### 8.1 Standalone inference

The training interface also provides standalone inference functionality.

Inspect the available options using:

```bash
eynollah-training inference --help
```

The inference process depends on the task. For example, classification takes an image as input, whereas reading-order inference requires PAGE-XML containing the document layout.

### 8.2 Integration into Eynollah

Trained models can be used in the main Eynollah application.

Since v0.8.0, individual models can be overridden, allowing a custom model to replace a corresponding model in the distributed model collection.

Consult the main [Models documentation](models.md) and the CLI help for the correct model categories and override syntax.

### 8.3 Model validation

Before deploying a custom model:

1. Test it on images not used during training.
2. Compare its results with the original model.
3. Inspect PAGE-XML output where applicable.
4. Check prediction quality across different document types.
5. Verify inference performance and resource requirements.

## 9. Troubleshooting and best practices

- Keep training and validation datasets separate.
- Verify image-label correspondence before training.
- Inspect generated segmentation labels visually.
- Avoid inconsistent PAGE-XML annotations.
- Select image dimensions and patch sizes appropriate to the task.
- Monitor validation performance to identify overfitting.
- Use pretrained models when appropriate.
- Preserve the original training checkpoints.
- Validate converted inference models independently.
- Consult command-specific help because CLI options can change between releases.

## 10. Further documentation

- [Eynollah repository](https://github.com/qurator-spk/eynollah)
- [Models documentation](models.md)
- [Changelog](../CHANGELOG.md)
- [Pretrained model distribution](https://zenodo.org/records/17727267)
- [Eynollah model collection on Hugging Face](https://huggingface.co/SBB)

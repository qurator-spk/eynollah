# Eynollah Models

Eynollah is a document layout analysis system designed primarily for digitised historical documents. It combines multiple deep learning models with image processing and heuristic methods to detect, classify and segment document images.

Eynollah supports document layout analysis, textline segmentation, binarization, image enhancement, reading order detection and text recognition (OCR).

This document describes the available model families, their intended purposes, alternative model variants, and the distinction between models used for inference and for training or fine-tuning.

## 1. Overview

### 1.1 The Eynollah model suite

Eynollah uses multiple models in a processing pipeline.

The models perform complementary tasks, including:

- Detection of page boundaries and cropping.
- Classification of the number of columns.
- Enhancement of low-resolution images.
- Semantic segmentation of layout regions.
- Segmentation of textlines.
- Segmentation of tables and illustrations.
- Reading order detection.
- Text recognition.

The individual models are combined with heuristics to derive the final result.

The general processing workflow is illustrated below:

![Eynollah processing flow](processing-flow.svg)

The exact sequence and selection of models depend on the processing options and document characteristics.

### 1.2 Deep learning and heuristics

An important characteristic of Eynollah is the combination of pixel-wise segmentation and heuristic post-processing.

As described in [Rezanezhad et al. (2023)](https://doi.org/10.1145/3604951.3605513), the original layout analysis approach uses convolutional neural networks to detect document elements at pixel level. 
Heuristics complement the neural networks, particularly for detecting marginalia, extracting textline geometries and determining reading order.

This combination addresses challenges commonly encountered in historical documents, such as:

- Different numbers of columns.
- Drop capitals and headings.
- Illustrations and tables.
- Skewed and curved text lines.
- Vertical text.
- Noise, annotations and degraded image quality.

The individual models can also be used independently for other document processing applications.

## 2. Model distributions and formats

Pretrained Eynollah models are distributed separately from the software package.

Since v0.9.0, two types of model packages are provided: inference models and training models.

They serve different purposes and should not be used interchangeably.

### 2.1 Inference models

Inference models are optimized for faster execution of predictions with Eynollah.

The current inference distribution contains models in ONNX format and is intended for use with ONNX Runtime and TensorRT.

Download the `models_inference_...zip` archive packages from [Zenodo](https://doi.org/10.5281/zenodo.21381102):

| Model      | Variant     | Download                                                                                                       |
|------------|:-----------:|----------------------------------------------------------------------------------------------------------------|
| `all`      |  Inference  | [models_inference_all_v0_9_1](https://zenodo.org/records/21381102/files/models_inference_all_v0_9_1.zip)       |
| `layout`   |  Inference  | [models_inference_layout_v0_9_1](https://zenodo.org/records/21381102/files/models_inference_layout_v0_9_1.zip) |
| `ocr`      |  Inference  | [models_inference_ocr_v0_9_1](https://zenodo.org/records/21381102/files/models_inference_ocr_v0_9_1.zip)       |
| `extra`    |  Inference  | [models_inference_extra_v0_9_1](https://zenodo.org/records/21381102/files/models_inference_extra_v0_9_1.zip)   |

Use these models when:

- Running Eynollah on document images.
- Deploying Eynollah in production.
- Using the standard ONNX/TensorRT inference backend.
- Integrating Eynollah into OCR-D processing workflows.

Since v0.9.0, ONNX/TensorRT is the default inference backend for the relevant models.

TensorRT may require an initial warm-up phase and a persistent cache directory.

### 2.2 Training and fine-tuning models

Training models contain the original framework models and weights needed to continue training or fine-tune existing models.

Download the `models_training...zip` archive packages from [Zenodo](https://doi.org/10.5281/zenodo.21381102):

| Model      | Variant     | Download                                                                                                       |
|------------|:-----------:|----------------------------------------------------------------------------------------------------------------|
| `all`      |  Training   | [models_training_all_v0_9_1](https://zenodo.org/records/21381102/files/models_training_all_v0_9_1.zip)         |
| `layout`   |  Training   | [models_training_layout_v0_9_1](https://zenodo.org/records/21381102/files/models_training_layout_v0_9_1.zip)   |
| `ocr`      |  Training   | [models_training_ocr_v0_9_1](https://zenodo.org/records/21381102/files/models_training_ocr_v0_9_1.zip)         |
| `extra`    |  Training   | [models_training_extra_v0_9_1](https://zenodo.org/records/21381102/files/models_training_extra_v0_9_1.zip)     |

These packages contain the models based on the corresponding original frameworks, primarily:

- TensorFlow/Keras.
- PyTorch for Transformer-based models.

Use these models when:

- Fine-tuning an existing model on a new document collection.
- Continuing training from pretrained weights.
- Experimenting with model architectures.
- Developing new models for specific document types.

Training requires the appropriate optional dependencies. See [Training Eynollah Models](train.md) for further information.

### 2.3 Relationship between inference and training models

The inference and training packages represent different distributions of the same model families.

| Property | Inference models | Training models |
|---|---|---|
| Main purpose | Prediction | Training and fine-tuning |
| Framework | ONNX | TensorFlow/Keras or PyTorch |
| Execution | ONNX Runtime / TensorRT | Original training framework |
| Optimization | Inference performance | Training flexibility |
| Typical use | Production processing | Model development |
| Distribution | `models_inference_...zip` | `models_training...zip` |

### 2.4 Model compatibility

Use the model package version corresponding to the Eynollah software release whenever possible.

Model compatibility depends on the architecture, model configuration, input dimensions, preprocessing and output representation.

When using a model from another release:

1. Check its architecture and expected input format.
2. Check the associated model configuration.
3. Verify that its output classes are compatible with the current processing pipeline.
4. Test its predictions before using it in production.

## 3. Model selection and customization

### 3.1 Model zoo

Since v0.7.0, Eynollah includes a model zoo that provides a central place to describe and load models.

Models are organized according to their functional categories and variants.

This makes it possible to select or replace individual models without necessarily replacing the entire model collection.

### 3.2 Model variants

Some model families provide several variants trained with different augmentation strategies or designed for different document characteristics.

Examples include:

| Model family | Variant | Main characteristic |
|---|---|---|
| Main region detection | Standard | General-purpose early layout detection |
| Main region detection | Scaling augmentation | Better handling of differently scaled document elements |
| Main region detection | Rotation augmentation | Better handling of vertically oriented text regions |
| Main region detection | Ensembled | Combines weights from multiple training epochs |
| Full region detection | 1–2 columns | Detection of headings and drop capitals in one- and two-column documents |
| Full region detection | 3 or more columns | Detection of headings and drop capitals in documents with more columns |
| Textline detection | Standard | Textline segmentation followed by heuristic geometry extraction |
| Textline detection | Light | Simpler and faster textline extraction using an artificial boundary class |

These variants should not be regarded as interchangeable models. Their suitability depends on the document collection and the desired balance between processing speed and prediction quality.

### 3.3 Choosing a model

The following recommendations provide a starting point:

| Document characteristics | Suggested model or variant |
|---|---|
| General historical documents | Standard main-region model |
| Documents containing elements at very different scales | Main-region model with scaling augmentation |
| Documents with vertically oriented text regions | Main-region model with rotation augmentation |
| Applications prioritising robustness | Ensembled main-region model |
| One- or two-column layouts | Full-region 1-2-column model |
| Three or more columns | Full-region 3-plus-column model |
| Applications prioritising speed | Light textline model |
| Applications requiring more detailed textline geometry processing | Standard textline model |

The final choice should be validated against representative documents from the intended collection.

### 3.4 Overriding individual models

Since v0.8.0, individual models can be overridden.

This allows a custom model to replace one component of the standard model collection without replacing all the other models.

The generic CLI options include:

```bash
eynollah -m <model_directory> ...
```

and:

```bash
eynollah -mv <model_category> <model_variant> <model_path> ...
```

The layout CLI also provides the `--model_version` option for selecting alternative model versions.

The root level model directory and individual overrides should be used according to the requirements of the selected processing task.

See [Usage documentation](use.md) for the relevant command-line options.

## 4. Layout analysis models

Layout analysis is the central functionality of Eynollah.

It identifies the principal structural elements of a document image and produces PAGE-XML output containing their pixel-wise segmentation and classifications.

### 4.1 Page extraction / border detection

Model: [eynollah-page-extraction](https://huggingface.co/SBB/eynollah-page-extraction)

This model detects the boundaries of the physical page.

Scanned documents frequently contain dark margins, scanner borders or other unwanted background noise. These can negatively affect subsequent layout analysis and OCR processing.

The page-extraction model uses pixel-wise segmentation to identify the page boundaries and supports contour-based page extraction.

The model introduced in v0.5.0 improves handling of skewed pages by producing page contours rather than relying exclusively on rectangular bounding boxes.

### 4.2 Column classification

Model: [eynollah-column-classifier](https://huggingface.co/SBB/eynollah-column-classifier)

The column classifier predicts the number of columns in a document.

It was trained to distinguish six classes:

- One column.
- Two columns.
- Three columns.
- Four columns.
- Five columns.
- Six or more columns.

The prediction is used to adapt subsequent layout analysis steps to the document structure.

### 4.3 Main region detection

Model: [eynollah-main-regions](https://huggingface.co/SBB/eynollah-main-regions)

Main-region detection performs the initial pixel-wise segmentation of document images.

It identifies the main layout classes, including:

- Background.
- Text regions.
- Images.
- Separators.

The model can use an artificial class during training to facilitate text-region detection. This allows the network to identify text regions efficiently, including when processing downscaled images.

#### Main-region model variants

**Standard**

[eynollah-main-regions](https://huggingface.co/SBB/eynollah-main-regions)

General-purpose model for main layout detection.

**Scaling augmentation**

[eynollah-main-regions-aug-scaling](https://huggingface.co/SBB/eynollah-main-regions-aug-scaling)

Uses scaling augmentation to improve detection of document elements with substantially different sizes.

**Rotation augmentation**

[eynollah-main-regions-aug-rotation](https://huggingface.co/SBB/eynollah-main-regions-aug-rotation)

Uses rotation augmentation to improve the segmentation of vertically oriented text regions.

**Ensembled**

[eynollah-main-regions-ensembled](https://huggingface.co/SBB/eynollah-main-regions-ensembled)

Combines weights from multiple training epochs to improve robustness and prediction stability.

### 4.4 Full layout detection

Full layout detection extends main layout detection by identifying additional document structures, particularly headings and drop capitals.

These elements can be difficult to distinguish from ordinary text regions in historical documents.

The full layout models use additional pixel-wise segmentation to identify these classes.

Two model variants are provided:

**One- and two-column documents**

[eynollah-full-regions-1column](https://huggingface.co/SBB/eynollah-full-regions-1column)

Detects headings and drop capitals in documents containing one or two columns.

**Documents with three or more columns**

[eynollah-full-regions-3pluscolumn](https://huggingface.co/SBB/eynollah-full-regions-3pluscolumn)

Detects headings and drop capitals in documents containing three or more columns.

The appropriate model should be selected according to the column structure of the document.

### 4.5 Text-line detection

Textline detection identifies individual lines of text within the detected text regions.

Two variants are available.

**Standard textline detection**

[eynollah-textline](https://huggingface.co/SBB/eynollah-textline)

The model performs pixel-wise segmentation into textline and background classes.

The segmentation is subsequently processed using heuristics to derive bounding boxes or contours.

The processing pipeline includes deskewing and additional operations to handle curved textlines.

For curved textlines, the text region is divided into vertical strips. The segmentation within each strip is deskewed, textlines are separated, and the resulting geometries are transformed back into the original orientation.

**Light textline detection**

[eynollah-textline_light](https://huggingface.co/SBB/eynollah-textline_light)

This variant uses an artificial boundary class in addition to the textline class.

The artificial class helps prevent connections between adjacent textlines and improves their separation.

Unlike the standard approach, the light variant does not require additional heuristics for extracting textline contours.

It provides a simpler and faster alternative for applications where its segmentation characteristics are suitable.

### 4.6 Table detection

Model: [eynollah-tables](https://huggingface.co/SBB/eynollah-tables)

The table detection model performs pixel-wise segmentation to identify tables in document images.

Traditional tables are often composed primarily of text rather than explicit graphical borders. Consequently, segmentation alone is insufficient in many cases.

Additional heuristics are used to identify and delineate tables.

### 4.7 Illustration detection

Model: [eynollah-image-extraction](https://huggingface.co/SBB/eynollah-image-extraction)

This model detects and extracts illustrations from historical document images.

It performs image segmentation specifically for illustration extraction and complements the general layout-detection models.

## 5. Image-processing models

### 5.1 Binarization

Model: [eynollah-binarization](https://huggingface.co/SBB/eynollah-binarization)

Binarization separates foreground and background pixels, producing a black-and-white document image.

The model performs pixel-wise segmentation and is particularly useful for degraded or low-quality documents.

Binarization can improve the subsequent processing of text and layout elements.

Eynollah supports binarization as a separate processing task and as part of the layout-analysis workflow.

### 5.2 Image enhancement

Model: [eynollah-enhancement](https://huggingface.co/SBB/eynollah-enhancement)

The image-enhancement model aims to improve the resolution of document images.

It is particularly useful for documents with insufficient pixel density, where text and other layout elements are difficult to identify reliably.

Enhancement can be integrated into the layout-analysis pipeline or executed as a separate processing task.

The decision to apply enhancement should take the original image resolution and document characteristics into account.

## 6. Reading order detection

Reading order detection determines the sequence in which document regions and textlines should be read.

Eynollah supports heuristic and machine-based approaches.

### 6.1 Heuristic reading order

The heuristic approach uses structural and geometric information to determine the reading sequence.

It considers factors including:

- Column positions.
- Separators.
- Headings.
- Floating elements.
- Marginalia.
- Relative positions of text regions.

The heuristics are an integral part of the Eynollah processing pipeline.

### 6.2 Machine-based reading order

A trainable reading order detection model is available for predicting relationships between text regions.

The model classifies pairwise relationships, after which a sorting algorithm derives the overall reading sequence.

The machine-based reading order detection model uses rotation augmentation.

The model can be used as a separate processing task or integrated into layout analysis.

The selection between heuristic and machine-based reading order detection should be evaluated using representative documents, particularly for complex multi-column layouts.

## 7. OCR models

Eynollah integrates two OCR model architectures:

- CNN-RNN.
- TrOCR.

These models are separate from the original layout-analysis model suite. They recognise text from extracted textline images and can be used in conjunction with Eynollah's layout analysis output.

### 7.1 CNN-RNN OCR

The CNN-RNN architecture combines convolutional neural networks for visual feature extraction with recurrent neural networks for sequence modelling.

The architecture uses CTC-based text recognition.

CNN-RNN models are generally faster and require fewer computational resources than Transformers.

They provide confidence scores that can be used to filter uncertain predictions.

The current model distribution includes CNN-RNN OCR models trained on different selections of Fraktur and Antiqua material.

#### Available models

**Fraktur-oriented model**

`model_eynollah_ocr_cnnrnn_20250805`

This model was trained on data containing a relatively large proportion of German Fraktur script.

**General-purpose model (default)**

`model_eynollah_ocr_cnnrnn_20250904`

Compared with the Fraktur-oriented model, this model was trained using a larger proportion of Antiqua material and achieves better results on the reported evaluation datasets.

The two models should be evaluated on representative documents before selecting one for a particular collection.

### 7.2 TrOCR

**Model:** `model_eynollah_ocr_trocr_20250919`

TrOCR is a Transformer-based OCR architecture.

Unlike CNN-RNN, it uses a Transformer approach for visual and textual sequence modelling.

The model is computationally more demanding and generally slower during inference.

However, it can be advantageous for strongly degraded documents and other difficult recognition cases.

The TrOCR implementation was integrated into Eynollah in v0.5.0. Training support and standalone inference were subsequently integrated in v0.9.0.

TrOCR training was described as untested in the v0.9.0 release notes. Users intending to fine-tune this model should therefore validate the training workflow independently.

### 7.3 OCR model comparison

The following results are reported based on the currently available OCR models.

Character Error Rate (CER) and Word Error Rate (WER) are used to evaluate recognition quality. Lower values indicate better results.

| Model | Dataset | Input | CER | WER |
|---|---|---|---:|---:|
| CNN-RNN 20250805 | OCR-D-GT-VD-SBB | BIN | 0.02147 | 0.05685 |
| CNN-RNN 20250805 | OCR-D-GT-VD-SBB | RGB | 0.01636 | 0.06285 |
| CNN-RNN 20250904 | OCR-D-GT-VD-SBB | BIN | 0.01635 | 0.05410 |
| CNN-RNN 20250904 | OCR-D-GT-VD-SBB | RGB | 0.01471 | 0.05813 |
| CNN-RNN 20250904 | BLN600 | RGB | 0.04409 | 0.08879 |
| CNN-RNN 20250904 | BLN600 | Enhanced | 0.03599 | 0.06244 |
| TrOCR 20250919 | OCR-D-GT-VD-SBB | BIN | 0.01841 | 0.05589 |
| TrOCR 20250919 | OCR-D-GT-VD-SBB | RGB | 0.01552 | 0.06177 |
| TrOCR 20250919 | BLN600 | RGB | 0.06347 | 0.13853 |

The results should be interpreted with regard to the dataset and input type.

In particular:

- RGB and binarized inputs do not necessarily produce equivalent results.
- Image enhancement can improve recognition on some degraded documents.
- The CNN-RNN models generally provide faster inference.
- TrOCR may be useful for difficult document images, but its computational requirements are higher.

These values are dataset-specific and should not be interpreted as guaranteed performance on other document collections.

### 7.4 OCR model selection

The following recommendations can be used as a starting point:

| Requirement | Suggested model |
|---|---|
| General-purpose OCR | CNN-RNN 20250904 |
| Documents dominated by Fraktur script | CNN-RNN 20250805 |
| Faster processing speed | CNN-RNN |
| Strongly degraded documents | TrOCR |
| Documents benefiting from enhancement | Evaluate enhanced input with CNN-RNN |

The best model depends on the script, print quality, historical period, language and characteristics of the training data.

## 8. Evaluation and limitations

### 8.1 Model evaluation

The quality of an Eynollah model should be assessed on data representative of the intended application.

Relevant evaluation measures depend on the task:

| Task | Possible evaluation measures |
|---|---|
| Layout segmentation | Intersection over Union (IoU), precision and recall |
| Text-line detection | Precision, recall and geometric overlap |
| Page extraction | Boundary and contour accuracy |
| Reading order | Pairwise ordering accuracy |
| Binarization | Pixel-level measures and visual assessment |
| Image enhancement | Image-quality measures and visual assessment |
| OCR | Character Error Rate and Word Error Rate |

### 8.2 Limitations

Eynollah is primarily designed for document layout analysis, with a focus on historical printed documents.

Potential limitations include:

- Unusual or highly complex page layouts.
- Manuscripts or document types substantially different from the training data.
- Very low-resolution or severely degraded images.
- Complex tables and illustrations.
- Unusual scripts and text orientations.
- Document structures that differ considerably from the training annotations.

For OCR, model performance also depends on the language, script, typeface, image quality and availability of representative training data.

Users should validate the results before applying Eynollah to large document collections.

## 9. Training and fine-tuning

Eynollah supports the training and fine-tuning of its model families.

The training workflow uses the original framework models rather than the optimized ONNX inference distributions.

Since v0.6.0, training is managed through the dedicated `eynollah-training` CLI.

Since v0.7.0, the model zoo provides a central mechanism for describing and loading models.

Since v0.8.0, training also includes TensorBoard integration and improved evaluation and loss calculation.

For details on:

- Preparing training and validation datasets.
- Generating ground truth labels.
- Configuring model architectures.
- Fine-tuning pretrained models.
- Monitoring training.
- Evaluating checkpoints.
- Converting trained models.

See [Training Eynollah Models](train.md).

### 9.1 Converting trained models for inference

Trained models may need to be converted into an appropriate inference format.

Since v0.9.0, Eynollah provides the `eynollah-training convert` CLI for model conversion.

Supported formats include:

- Keras HDF5.
- Native Keras.
- TensorFlow SavedModel.
- TensorFlow Serving.
- ONNX.

Conversion compatibility depends on the model architecture and supported operations.

Converted models should be tested against the original training models before deployment.

## 10. Model downloads and further information

### 10.1 Model distributions

Pretrained model packages are available through:

- [Zenodo model distribution](https://zenodo.org/records/17727267)
- [Eynollah models on Hugging Face](https://huggingface.co/collections/SBB/eynollah-models)

Use the inference distribution for optimized prediction and the training distribution for fine-tuning.

### 10.2 Individual model cards

The individual model cards provide further information about the original models:

| Model | Model card |
|---|---|
| Image enhancement | [eynollah-enhancement](https://huggingface.co/SBB/eynollah-enhancement) |
| Page extraction | [eynollah-page-extraction](https://huggingface.co/SBB/eynollah-page-extraction) |
| Column classification | [eynollah-column-classifier](https://huggingface.co/SBB/eynollah-column-classifier) |
| Binarization | [eynollah-binarization](https://huggingface.co/SBB/eynollah-binarization) |
| Main region detection | [eynollah-main-regions](https://huggingface.co/SBB/eynollah-main-regions) |
| Main region detection with scaling augmentation | [eynollah-main-regions-aug-scaling](https://huggingface.co/SBB/eynollah-main-regions-aug-scaling) |
| Main region detection with rotation augmentation | [eynollah-main-regions-aug-rotation](https://huggingface.co/SBB/eynollah-main-regions-aug-rotation) |
| Ensembled main region detection | [eynollah-main-regions-ensembled](https://huggingface.co/SBB/eynollah-main-regions-ensembled) |
| Full region detection, 1–2 columns | [eynollah-full-regions-1column](https://huggingface.co/SBB/eynollah-full-regions-1column) |
| Full region detection, 3+ columns | [eynollah-full-regions-3pluscolumn](https://huggingface.co/SBB/eynollah-full-regions-3pluscolumn) |
| Text-line detection | [eynollah-textline](https://huggingface.co/SBB/eynollah-textline) |
| Light text-line detection | [eynollah-textline_light](https://huggingface.co/SBB/eynollah-textline_light) |
| Table detection | [eynollah-tables](https://huggingface.co/SBB/eynollah-tables) |
| Illustration extraction | [eynollah-image-extraction](https://huggingface.co/SBB/eynollah-image-extraction) |

### 10.3 Further documentation

- [Eynollah repository](https://github.com/qurator-spk/eynollah)
- [Training documentation](train.md)
- [Usage documentation](use.md)
- [Docker documentation](docker.md)

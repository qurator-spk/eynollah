# Eynollah
> Document Layout Analysis, Binarization and OCR with Deep Learning Models and Heuristics

[![Python Versions](https://img.shields.io/pypi/pyversions/eynollah.svg)](https://pypi.python.org/pypi/eynollah)
[![PyPI Version](https://img.shields.io/pypi/v/eynollah)](https://pypi.org/project/eynollah/)
[![GH Actions Test](https://github.com/qurator-spk/eynollah/actions/workflows/test-eynollah.yml/badge.svg)](https://github.com/qurator-spk/eynollah/actions/workflows/test-eynollah.yml)
[![GH Actions Deploy](https://github.com/qurator-spk/eynollah/actions/workflows/build-docker.yml/badge.svg)](https://github.com/qurator-spk/eynollah/actions/workflows/build-docker.yml)
[![License: ASL](https://img.shields.io/pypi/l/eynollah)](https://opensource.org/license/apache-2-0/)
[![DOI](https://img.shields.io/badge/DOI-10.1145%2F3604951.3605513-red)](https://doi.org/10.1145/3604951.3605513)

![](https://user-images.githubusercontent.com/952378/102350683-8a74db80-3fa5-11eb-8c7e-f743f7d6eae2.jpg)

## Features
* Document layout analysis with support for 10 segmentation classes: 
  * background, [page border](https://ocr-d.de/en/gt-guidelines/trans/lyRand.html), [text region](https://ocr-d.de/en/gt-guidelines/trans/lytextregion.html#textregionen__textregion_), [text line](https://ocr-d.de/en/gt-guidelines/pagexml/pagecontent_xsd_Complex_Type_pc_TextLineType.html), [header](https://ocr-d.de/en/gt-guidelines/trans/lyUeberschrift.html), [image](https://ocr-d.de/en/gt-guidelines/trans/lyBildbereiche.html), [separator](https://ocr-d.de/en/gt-guidelines/trans/lySeparatoren.html),
  [marginalia](https://ocr-d.de/en/gt-guidelines/trans/lyMarginalie.html), [initial](https://ocr-d.de/en/gt-guidelines/trans/lyInitiale.html), [table](https://ocr-d.de/en/gt-guidelines/trans/lyTabellen.html)
* Textline segmentation to bounding boxes or polygons (including for curved lines and vertical text)
* Binarization
* Text recognition
* Detection of reading order (left-to-right or right-to-left)
* Output in [PAGE-XML](https://github.com/PRImA-Research-Lab/PAGE-XML)
* [OCR-D](https://github.com/qurator-spk/eynollah/tree/main/docs/ocrd.md) interface

:warning: Development is focused on achieving the optimal quality for a wide variety of historical documents using a 
combination of multiple deep learning models and heuristics; therefore processing can be slow.

## Installation
Python `3.9-3.11` with ONNX Runtime on Linux are currently supported.

For GPU support, Nvidia drivers supporting CUDA 12 must be installed.
The runtime dependencies will pull in ONNX, TensorRT and CUDA runtime
libraries (including cuDNN) from PyPI.

You can either install from PyPI

```
pip install eynollah
```

or clone the repository, enter it and install (editable) with

```
git clone git@github.com:qurator-spk/eynollah.git
cd eynollah; pip install -e .
```

To also install the dependencies for the OCR engines:

```
pip install "eynollah[OCR]"
```

:warning: Requirements for OCR are more involved, as they need Tensorflow (with `tf-keras`) and/or
Torch (with `transformers`), which may also have conflicting CUDA dependencies. 
An ONNX conversion for these models may be achieved soon.

## Models
Pretrained models can be downloaded from [Zenodo](https://doi.org/10.5281/zenodo.17194823). 
For model documentation and model cards, see [`models.md`](https://github.com/qurator-spk/eynollah/tree/main/docs/models.md).

Two model variants are provided:
- For faster runtime inference, download the ONNX model variants distributed as `models_inference_...zip`
- For finetuning and training, download the original (Tensorflow / Torch) models distributed as `models_training...zip`

To train your own model with Eynollah, see [`train.md`](https://github.com/qurator-spk/eynollah/tree/main/docs/train.md).

## Usage
Eynollah supports five use cases: 
1. [layout analysis (segmentation)](#layout-analysis), 
2. [binarization](#binarization), 
3. [image enhancement](#image-enhancement), 
4. [text recognition (OCR)](#ocr), and 
5. [reading order detection](#reading-order-detection).

The **generic options** shared by all subcommands are:
```sh
  -m <directory containing model files>
  -mv <model category> <model variant> <model path>
  -D <device specifier>
  -l <log level>
```

### Layout Analysis
Detect layout elements and their reading order.

The command-line interface for layout analysis can be called like this:

```sh
eynollah [GENERIC_OPTIONS] layout \
  -i <single image file> | -di <directory containing image files> \
  -o <output directory> \
     [OPTIONS]
```

The following options can be used to further configure the processing:

| option            | description                                                                        |
|-------------------|:-----------------------------------------------------------------------------------|
| `-fl`             | apply full layout analysis including all steps and classes (recommended)           |
| `-tab`            | apply table detection                                                              |
| `-ae`             | apply image enhancement (the resulting image is saved to the output directory)     |
| `-as`             | apply automatic scaling                                                            |
| `-cl`             | apply contour detection for curved text lines, deskewing all regions independently |
| `-ib`             | apply binarization (the resulting bitonal image is saved to the output directory)  |
| `-ep`             | enable plotting (MUST always be used with `-sl`, `-sd`, `-sa`, `-si` or `-ae`)     |
| `-ho`             | ignore headers for reading order detection                                         |
| `-si <directory>` | save detected image regions to this directory                                      |
| `-sd <directory>` | save deskewed image to this directory                                              |
| `-sl <directory>` | save layout prediction as plot to this directory                                   |
| `-sp <directory>` | save cropped page image to this directory                                          |
| `-sa <directory>` | save all (plot, enhanced/binary image, layout) to this directory                   |
| `-thart`          | confidence threshold of artificial boundary class during textline detection        |
| `-tharl`          | confidence threshold of artificial boundary class during region detection          |
| `-ncu`            | set upper limit of columns in document image                                       |
| `-ncl`            | set lower limit of columns in document image                                       |
| `-slro`           | skip layout detection and reading order                                            |
| `-romb`           | apply machine learning based reading order detection                               |
| `-ipe`            | ignore page extraction                                                             |
| `-j`              | number of CPU jobs to run parallel (useful with -di)                               |
| `-H`              | when to halt when some jobs fail                                                   |

### Binarization
Perform document image binarization. 

The command-line interface for binarization can be called like this:

```sh
eynollah [GENERIC_OPTIONS] binarization \
  -i <single image file> | -di <directory containing image files> \
  -o <output directory> \
  [OPTIONS]
```

### Image Enhancement
Enlarge and enhance images. Useful when the scan quality is low.

The command-line interface for image enhancement can be called like this:

```sh
eynollah [GENERIC_OPTIONS] enhancement \
  -i <single image file> | -di <directory containing image files> \
  -o <output directory> \
  [OPTIONS]
```

The following options can be used to further configure the processing:

| option            | description                                                                                 |
|-------------------|:--------------------------------------------------------------------------------------------|
| `-sos`            | save the enhanced image in original image size                                              |
| `-ncu`            | upper limit of columns in document image                                                    |
| `-ncl`            | lower limit of columns in document image                                                    |

### OCR
Perform text recognition using either a CNN-RNN model (default) or a TrOCR model.
Needs a PAGE-XML input file containing textline segmentation information.

The command-line interface for OCR can be called like this:

```sh
eynollah [GENERIC_OPTIONS] ocr \
  -i <single image file> | -di <directory containing image files> \
  -xml <page-xml input file> | -dx <directory containing page-xml files> \
  -o <output directory> \
```

The following options can be used to further configure the processing:

| option            | description                                                                                |
|-------------------|:-------------------------------------------------------------------------------------------|
| `-trocr`          | use TrOCR model instead of CNN-RNN model                                                   |
| `-dib`            | directory of binarized images (file type must be '.png'), prediction with both RGB and bin |
| `-doit`           | directory for output images rendered with the predicted text                               |
| `-nmtc`           | cropped textline images will not be masked with textline contour                           |
| `-bs`             | OCR inference batch size. Default batch size is 2 for TrOCR and 8 for CNN-RNN models       |
| `-min_conf`       | minimum OCR confidence value. OCR with textline confidence lower than this will be ignored |

### Reading Order Detection
Detect reading order using a machine learning based model.
Needs a PAGE-XML input file containing textline segmentation information.

The command-line interface for machine learning based reading 
order detection can be called like this:

```sh
eynollah [GENERIC_OPTIONS] machine-based-reading-order \
  -i <single image file> | -di <directory containing image files> \
  -xml <xml file name> | -dx <directory containing xml files> \
  -o <output directory> 
```

## Documentation
Additional documentation:
* For using Eynollah with Docker, see [`docker.md`](https://github.com/qurator-spk/eynollah/tree/main/docs/docker.md).
* For using Eynollah with OCR-D, see [`ocrd.md`](https://github.com/qurator-spk/eynollah/tree/main/docs/ocrd.md).
* For advanced use, see [`use.md`](https://github.com/qurator-spk/eynollah/tree/main/docs/use.md).
* For model documentation and model cards, see [`models.md`](https://github.com/qurator-spk/eynollah/tree/main/docs/models.md).
* For model training or fine-tuning, see [`train.md`](https://github.com/qurator-spk/eynollah/tree/main/docs/train.md).

## How to cite
```bibtex
@inproceedings{hip23rezanezhad,
  title     = {Document Layout Analysis with Deep Learning and Heuristics},
  author    = {Rezanezhad, Vahid and Baierer, Konstantin and Gerber, Mike and Labusch, Kai and Neudecker, Clemens},
  booktitle = {Proceedings of the 7th International Workshop on Historical Document Imaging and Processing {HIP} 2023,
               San José, CA, USA, August 25-26, 2023},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  year      = {2023},
  pages     = {73--78},
  url       = {https://doi.org/10.1145/3604951.3605513}
}
```

# Eynollah

> Document Layout Analysis, Binarization and OCR with Deep Learning and Heuristics

[![Python Versions](https://img.shields.io/pypi/pyversions/eynollah.svg)](https://pypi.python.org/pypi/eynollah)
[![PyPI Version](https://img.shields.io/pypi/v/eynollah)](https://pypi.org/project/eynollah/)
[![GH Actions Test](https://github.com/qurator-spk/eynollah/actions/workflows/test-eynollah.yml/badge.svg)](https://github.com/qurator-spk/eynollah/actions/workflows/test-eynollah.yml)
[![GH Actions Deploy](https://github.com/qurator-spk/eynollah/actions/workflows/build-docker.yml/badge.svg)](https://github.com/qurator-spk/eynollah/actions/workflows/build-docker.yml)
[![License: ASL](https://img.shields.io/pypi/l/eynollah)](https://opensource.org/license/apache-2-0/)
[![DOI](https://img.shields.io/badge/DOI-10.1145%2F3604951.3605513-red)](https://doi.org/10.1145/3604951.3605513)

![](https://user-images.githubusercontent.com/952378/102350683-8a74db80-3fa5-11eb-8c7e-f743f7d6eae2.jpg)

## Features
* Document layout analysis using pixelwise segmentation models with support for 10 segmentation classes: 
  * background, [page border](https://ocr-d.de/en/gt-guidelines/trans/lyRand.html), [text region](https://ocr-d.de/en/gt-guidelines/trans/lytextregion.html#textregionen__textregion_), [text line](https://ocr-d.de/en/gt-guidelines/pagexml/pagecontent_xsd_Complex_Type_pc_TextLineType.html), [header](https://ocr-d.de/en/gt-guidelines/trans/lyUeberschrift.html), [image](https://ocr-d.de/en/gt-guidelines/trans/lyBildbereiche.html), [separator](https://ocr-d.de/en/gt-guidelines/trans/lySeparatoren.html), [marginalia](https://ocr-d.de/en/gt-guidelines/trans/lyMarginalie.html), [initial](https://ocr-d.de/en/gt-guidelines/trans/lyInitiale.html), [table](https://ocr-d.de/en/gt-guidelines/trans/lyTabellen.html)
* Textline segmentation to bounding boxes or polygons (contours) including for curved lines and vertical text
* Document image binarization with pixelwise segmentation or hybrid CNN-Transformer models
* Text recognition (OCR) with CNN-RNN or TrOCR models
* Detection of reading order (left-to-right or right-to-left) using heuristics or trainable models
* Output in [PAGE-XML](https://github.com/PRImA-Research-Lab/PAGE-XML)
* [OCR-D](https://github.com/qurator-spk/eynollah#use-as-ocr-d-processor) interface

:warning: Development is focused on achieving the best quality of results for a wide variety of historical 
documents using a combination of multiple deep learning models and heuristics; therefore processing can be slow.

## Installation

Python `3.8-3.11` with ONNX Runtime on Linux are currently supported.

For GPU support, NVidia drivers supporting CUDA 12 must be installed.
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

Alternatively, you can run `make install` or `make install-dev` for editable installation.

To also install the dependencies for the OCR engines:

```
pip install "eynollah[OCR]"
# or
make install EXTRAS=OCR
```

> **Note**: Requirements for OCR are more involved, 
> as they may need Tensorflow (with tf-keras) and/or
> Torch (with transformers). Those two frameworks may
> also have conflicting CUDA dependencies. An ONNX
> conversion for these models may be achieved soon.
> :construction:

### Docker

Use

```
docker pull ghcr.io/qurator-spk/eynollah:latest
```

When using Eynollah with Docker, see [`docker.md`](https://github.com/qurator-spk/eynollah/tree/main/docs/docker.md).

## Models

Pretrained models can be downloaded from [Zenodo](https://doi.org/10.5281/zenodo.17194823) or [Hugging Face](https://huggingface.co/SBB?search_models=eynollah).

For fast runtime inference, download the ONNX models distributed as `models_inference_...zip`. 

For finetuning training, download the original (Tensorflow / Torch) models distributed as `models_training...zip`
(and install the `[training]` extra).

For model documentation and model cards, see [`models.md`](https://github.com/qurator-spk/eynollah/tree/main/docs/models.md).

## Training

To train your own model with Eynollah, see [`train.md`](https://github.com/qurator-spk/eynollah/tree/main/docs/train.md) and use the tools in the [`train`](https://github.com/qurator-spk/eynollah/tree/main/train) folder.

## Usage

Eynollah supports five use cases: 
1. [layout analysis (segmentation)](#layout-analysis), 
2. [binarization](#binarization), 
3. [image enhancement](#image-enhancement), 
4. [text recognition (OCR)](#ocr), and 
5. [reading order detection](#reading-order-detection).

Some example outputs can be found in [`examples.md`](https://github.com/qurator-spk/eynollah/tree/main/docs/examples.md).

The **generic options** shared by all subcommands are:
```sh
  -m <directory containing model files>
  -mv <model category> <model variant> <model path>
  -D <device specifier>
  -l <log level>
```

### Layout Analysis

Detects layout elements, i.e. regions of various types and text lines,
and determines their reading order using either heuristic methods or a
[pretrained model](https://github.com/qurator-spk/eynollah#machine-based-reading-order).

The command-line interface for layout analysis can be called like this:

```sh
eynollah [GENERIC_OPTIONS] layout \
  -i <single image file> | -di <directory containing image files> \
  -o <output directory> \
     [OPTIONS]
```

The following options can be used to further configure the processing:

| option            | description                                                                                 |
|-------------------|:--------------------------------------------------------------------------------------------|
| `-fl`             | full layout analysis including all steps and segmentation classes (recommended)             |
| `-tab`            | apply table detection                                                                       |
| `-ae`             | apply enhancement (the resulting image is saved to the output directory)                    |
| `-as`             | apply scaling                                                                               |
| `-cl`             | apply contour detection for curved text lines, deskewing all regions independently          |
| `-ib`             | apply binarization (the resulting image is saved to the output directory)                   |
| `-ep`             | enable plotting (MUST always be used with `-sl`, `-sd`, `-sa`, `-si` or `-ae`)              |
| `-ho`             | ignore headers for reading order dectection                                                 |
| `-si <directory>` | save image regions detected to this directory                                               |
| `-sd <directory>` | save deskewed image to this directory                                                       |
| `-sl <directory>` | save layout prediction as plot to this directory                                            |
| `-sp <directory>` | save cropped page image to this directory                                                   |
| `-sa <directory>` | save all (plot, enhanced/binary image, layout) to this directory                            |
| `-thart`          | confidence threshold of artifical boundary class during textline detection                  |
| `-tharl`          | confidence threshold of artifical boundary class during region detection                    |
| `-ncu`            | upper limit of columns in document image                                                    |
| `-ncl`            | lower limit of columns in document image                                                    |
| `-slro`           | skip layout detection and reading order                                                     |
| `-romb`           | apply machine based reading order detection                                                 |
| `-ipe`            | ignore page extraction                                                                      |
| `-j`              | number of CPU jobs to run parallel (useful with -di)                                        |
| `-H`              | when to halt when some jobs fail                                                            |

The default is to only perform layout detection of main regions
(background, text, images, separators and marginals).

The best output quality is achieved when RGB images are used as input
rather than greyscale or binarized images.

Additional documentation can be found in
[`usage.md`](https://github.com/qurator-spk/eynollah/tree/main/docs/usage.md).

### Binarization

Performs document image binarization (thresholding)
using pretrained pixelwise segmentation models. 

The command-line interface for binarization can be called like this:

```sh
eynollah [GENERIC_OPTIONS] binarization \
  -i <single image file> | -di <directory containing image files> \
  -o <output directory> \
  [OPTIONS]
```

### Image Enhancement

This enlarges and enhances images. Useful in case the scan quality is low.

```sh
eynollah [GENERIC_OPTIONS] enhancement \
  -i <single image file> | -di <directory containing image files> \
  -o <output directory> \
  [OPTIONS]
```

| option            | description                                                                                 |
|-------------------|:--------------------------------------------------------------------------------------------|
| `-sos`            | save the enhanced image in original image size                                              |
| `-ncu`            | upper limit of columns in document image                                                    |
| `-ncl`            | lower limit of columns in document image                                                    |

### OCR

Performs text recognition using either a CNN-RNN model or a Transformer model.
Needs a PAGE-XML input file.

The command-line interface for OCR can be called like this:

```sh
eynollah [GENERIC_OPTIONS] ocr \
  -i <single image file> | -di <directory containing image files> \
  -dx <directory of xmls> \
  -o <output directory> \
```

The following options can be used to further configure the ocr processing:

| option            | description                                                                                |
|-------------------|:-------------------------------------------------------------------------------------------|
| `-trocr`          | use transformer OCR model instead of CNN-RNN model                                         |
| `-dib`            | directory of binarized images (file type must be '.png'), prediction with both RGB and bin |
| `-doit`           | directory for output images rendered with the predicted text                               |
| `-nmtc`           | cropped textline images will not be masked with textline contour                           |
| `-bs`             | ocr inference batch size. Default batch size is 2 for trocr and 8 for cnn_rnn models       |
| `-min_conf`       | minimum OCR confidence value. OCR with textline conf lower than this will be ignored       |


### Reading Order Detection

Reading order can be detected either during layout analysis,
or as a separate module, which requires a PAGE-XML input file.

The command-line interface for machine based reading order can be called like this:

```sh
eynollah [GENERIC_OPTIONS] machine-based-reading-order \
  -i <single image file> | -di <directory containing image files> \
  -xml <xml file name> | -dx <directory containing xml files> \
  -o <output directory> 
```

## Use as OCR-D processor

See [`ocrd.md`](https://github.com/qurator-spk/eynollah/tree/main/docs/ocrd.md).

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

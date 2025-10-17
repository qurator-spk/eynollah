# Training eynollah

This README explains the technical details of how to set up and run training, for detailed information on parameterization, see [`docs/train.md`](../docs/train.md)

## Introduction

This folder contains the source code for training an encoder model for document image segmentation.

## Installation

Clone the repository and install eynollah along with the dependencies necessary for training:

```sh
git clone https://github.com/qurator-spk/eynollah
cd eynollah
pip install '.[training]'
```

### Pretrained encoder

Download our pretrained weights and add them to a `train/pretrained_model` folder:   

```sh
cd train
wget -O pretrained_model.tar.gz https://zenodo.org/records/17243320/files/pretrained_model_v0_5_1.tar.gz?download=1
tar xf pretrained_model.tar.gz
```

### Binarization training data

A small sample of training data for binarization experiment can be found [on
zenodo](https://zenodo.org/records/17243320/files/training_data_sample_binarization_v0_5_1.tar.gz?download=1),
which contains `images` and `labels` folders.

### Helpful tools

* [`pagexml2img`](https://github.com/qurator-spk/page2img)
> Tool to extract 2-D or 3-D RGB images from PAGE-XML data. In the former case, the output will be 1 2-D image array which each class has filled with a pixel value. In the case of a 3-D RGB image, 
each class will be defined with a RGB value and beside images, a text file of classes will also be produced.
* [`cocoSegmentationToPng`](https://github.com/nightrome/cocostuffapi/blob/17acf33aef3c6cc2d6aca46dcf084266c2778cf0/PythonAPI/pycocotools/cocostuffhelper.py#L130)
> Convert COCO GT or results for a single image to a segmentation map and write it to disk.
* [`ocrd-segment-extract-pages`](https://github.com/OCR-D/ocrd_segment/blob/master/ocrd_segment/extract_pages.py)
> Extract region classes and their colours in mask (pseg) images. Allows the color map as free dict parameter, and comes with a default that mimics PageViewer's coloring for quick debugging; it also warns when regions do overlap.

### Train using Docker

Build the Docker image:

```bash
cd train
docker build -t model-training .
```

Run Docker image 

```bash
cd train
docker run --gpus all -v $PWD:/entry_point_dir model-training
```

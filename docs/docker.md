## Inference with Docker

    docker pull ghcr.io/qurator-spk/eynollah:latest

### 1. ocrd resource manager
(just once, to get the models and install them into a named volume for later re-use)

    vol_models=ocrd-resources:/usr/local/share/ocrd-resources
    docker run --rm -v $vol_models ocrd/eynollah ocrd resmgr download ocrd-eynollah-segment default

Now, each time you want to use Eynollah, pass the same resources volume again.
Also, bind-mount some data directory, e.g. current working directory $PWD (/data is default working directory in the container).

Either use standalone CLI (2) or OCR-D CLI (3):

### 2. standalone CLI 
(follow self-help, cf. readme)

    docker run --rm -v $vol_models -v $PWD:/data ocrd/eynollah eynollah binarization --help
    docker run --rm -v $vol_models -v $PWD:/data ocrd/eynollah eynollah layout --help
    docker run --rm -v $vol_models -v $PWD:/data ocrd/eynollah eynollah ocr --help

### 3. OCR-D CLI 
(follow self-help, cf. readme and https://ocr-d.de/en/spec/cli)

    docker run --rm -v $vol_models -v $PWD:/data ocrd/eynollah ocrd-eynollah-segment -h
    docker run --rm -v $vol_models -v $PWD:/data ocrd/eynollah ocrd-sbb-binarize -h

Alternatively, just "log in" to the container once and use the commands there:

    docker run --rm -v $vol_models -v $PWD:/data -it ocrd/eynollah bash

## Training with Docker

Build the Docker image

    cd train
    docker build -t model-training .

Run the Docker image

    cd train
    docker run --gpus all -v $PWD:/entry_point_dir model-training

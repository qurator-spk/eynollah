# Running Eynollah with Docker

This guide explains how to run [Eynollah](https://github.com/qurator-spk/eynollah) in a Docker container. It covers installation, model management, GPU acceleration, standalone inference, and integration into OCR-D workflows.

Docker provides a convenient way to run Eynollah without installing its Python dependencies and machine-learning frameworks directly on the host system.

## 1. Overview

Eynollah provides deep-learning-based document image processing, including:

- Document layout analysis and text-line detection.
- Image binarization.
- Image enhancement.
- Reading-order detection.
- OCR (subject to the availability of the required OCR dependencies).

The Docker image is published at:

`ghcr.io/qurator-spk/eynollah`

### 1.1 Docker image and supported functionality

Since v0.9.0, the official Docker image is based on `ocrd/core-cuda-onnx`.

This image provides the ONNX/TensorRT-based inference environment for layout processing. The optional OCR dependencies are not installed in the standard image.

Consequently, the standard Docker image should primarily be used for layout analysis and the associated inference functionality.

OCR engines based on CNN-RNN and TrOCR require additional dependencies, including TensorFlow and/or PyTorch. These are not included in the standard Docker image because of potential conflicts between their CUDA dependencies.

### 1.2 Standalone CLI and OCR-D interface

Eynollah provides two interfaces:

1. **Standalone CLI:** Run Eynollah directly on images or directories of images.
2. **OCR-D CLI:** Run Eynollah as a processor within an OCR-D workflow.

Both interfaces are available in the Docker environment for the supported functionality.

## 2. Requirements

### 2.1 Docker

Install Docker Engine or Docker Desktop and ensure that the Docker daemon is running.

See the official [Docker installation documentation](https://docs.docker.com/engine/install/).

### 2.2 GPU support

The standard Docker image uses ONNX Runtime and TensorRT for GPU-accelerated inference.

For GPU execution, the host system requires:

- An NVIDIA GPU.
- NVIDIA drivers supporting CUDA 12.
- NVIDIA Container Toolkit configured for Docker.

See the [NVIDIA Container Toolkit installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

Check whether Docker can access the GPU:

```bash
docker run --rm --gpus all \
    nvidia/cuda:12.0.0-base-ubuntu22.04 \
    nvidia-smi
```

If GPU access is unavailable, Eynollah can also be run on the CPU, although inference will generally be slower.

## 3. Obtaining the Docker image

Pull the latest published image:

```bash
docker pull ghcr.io/qurator-spk/eynollah:latest
```

For reproducible workflows, use a release-specific image tag rather than `latest`.

Check the available command-line interface:

```bash
docker run --rm ghcr.io/qurator-spk/eynollah:latest \
    eynollah --help
```

The default working directory inside the container is `/data`.

For more information about the available commands, see the [Eynollah usage documentation](use.md).

## 4. Model and resource management

Eynollah requires pretrained models for inference. These models are distributed separately from the Docker image.

The current model distribution provides separate packages for training and inference.

For Docker-based inference, use the inference model package, which contains the optimized ONNX models.

See the [model documentation](models.md) for available models and the [official model distribution](https://huggingface.co/SBB?search_models=eynollah).

### 4.1 Persistent Docker volumes

Model files and other resources should be stored outside the container so that they remain available when containers are removed.

Docker named volumes provide a convenient way to persist OCR-D resources and configuration data.

For example, define a named volume:

```bash
vol_models=ocrd-resources:/usr/local/share/ocrd-resources
```

The volume can be initialized using the OCR-D resource manager:

```bash
docker run --rm \
    -v $vol_models \
    ghcr.io/qurator-spk/eynollah:latest \
    ocrd resmgr download ocrd-eynollah-segment default
```

This downloads the default OCR-D resource package.

The same volume must be mounted whenever the container is started so that the downloaded resources remain available.

For custom model distributions, mount the directory containing the model files into the container and provide its location through the Eynollah model-directory option (`-m`).

### 4.2 Custom models and model overrides

Eynollah supports custom model directories and individual model overrides.

To use a custom model directory, mount it into the container:

```bash
docker run --rm \
    -v $PWD/models:/models:ro \
    -v $PWD/data:/data \
    ghcr.io/qurator-spk/eynollah:latest \
    eynollah -m /models layout \
        -i /data/input.tif \
        -o /data/output
```

The mounted model directory must contain the appropriate model files and directory structure.

Since v0.8.0, individual models can be overridden. This allows custom models to replace selected models from the standard model distribution.

Consult the [model documentation](models.md) for the available model categories and variants.

## 5. Running Eynollah with Docker

The following examples use the official Docker image and mount the current host directory into `/data`.

### 5.1 Standalone CLI

First, display the available options:

```bash
docker run --rm \
    -v $PWD:/data \
    ghcr.io/qurator-spk/eynollah:latest \
    eynollah --help
```

For more detailed information about a specific command:

```bash
docker run --rm \
    -v $PWD:/data \
    ghcr.io/qurator-spk/eynollah:latest \
    eynollah layout --help
```

### 5.2 Layout analysis

To process a single image:

```bash
docker run --rm \
    --gpus all \
    -v $vol_models \
    -v $PWD:/data \
    ghcr.io/qurator-spk/eynollah:latest \
    eynollah layout \
        -i /data/input.tif \
        -o /data/output
```

To process a directory of images:

```bash
docker run --rm \
    --gpus all \
    -v $vol_models \
    -v $PWD:/data \
    ghcr.io/qurator-spk/eynollah:latest \
    eynollah layout \
        -di /data/input \
        -o /data/output
```

The resulting PAGE-XML files and additional output images are written to the mounted output directory.

### 5.3 Binarization

To inspect the available binarization options:

```bash
docker run --rm \
    -v $PWD:/data \
    ghcr.io/qurator-spk/eynollah:latest \
    eynollah binarization --help
```

Use this command to determine whether the desired operation is supported by the selected image version.

### 5.4 Image enhancement

The enhancement CLI can be inspected in the same way:

```bash
docker run --rm \
    -v $PWD:/data \
    ghcr.io/qurator-spk/eynollah:latest \
    eynollah enhancement --help
```

### 5.5 Reading-order detection

Reading-order detection can also be performed as a separate operation.

```bash
docker run --rm \
    -v $PWD:/data \
    ghcr.io/qurator-spk/eynollah:latest \
    eynollah machine-based-reading-order --help
```

The input must include the corresponding PAGE-XML annotations.

### 5.6 GPU and CPU execution

The Docker image supports GPU-accelerated inference.

To enable GPU access, add:

```bash
--gpus all
```

For systems without an available NVIDIA GPU, omit this option and explicitly select CPU execution where necessary.

The `-D` option controls device selection in the Eynollah CLI.

Inspect the command-specific help for the accepted device values.

### 5.7 Persistent TensorRT cache

Since v0.9.0, ONNX/TensorRT is the default inference backend for the relevant models.

TensorRT requires an initial warm-up phase and uses a persistent cache directory configured through `$XDG_CONFIG_HOME`.

The standard Docker image sets this variable to:

```text
/usr/local/share/ocrd-resources
```

The resource volume described in Section 4 can therefore also preserve this configuration and cache data between container executions.

Using the same volume across runs avoids unnecessary repeated initialization.

## 6. Using Eynollah as an OCR-D processor

The Docker image also provides the OCR-D command-line interface.

OCR-D allows Eynollah to be integrated into document-processing workflows involving other OCR-D processors.

See the [OCR-D CLI specification](https://ocr-d.de/en/spec/cli).

### 6.1 Layout analysis

To inspect the Eynollah OCR-D processor:

```bash
docker run --rm \
    -v $vol_models \
    -v $PWD:/data \
    ghcr.io/qurator-spk/eynollah:latest \
    ocrd-eynollah-segment --help
```

A typical processor invocation is:

```bash
docker run --rm \
    --gpus all \
    -v $vol_models \
    -v $PWD:/data \
    ghcr.io/qurator-spk/eynollah:latest \
    ocrd-eynollah-segment \
        -I OCR-D-IMG \
        -O OCR-D-SEG \
        -P models eynollah_layout_v0_9_0
```

The example assumes that the OCR-D workspace and the required input file group have already been prepared.

### 6.2 Binarization

The OCR-D binarization processor can be inspected using:

```bash
docker run --rm \
    -v $vol_models \
    -v $PWD:/data \
    ghcr.io/qurator-spk/eynollah:latest \
    ocrd-sbb-binarize --help
```

For example:

```bash
docker run --rm \
    --gpus all \
    -v $vol_models \
    -v $PWD:/data \
    ghcr.io/qurator-spk/eynollah:latest \
    ocrd-sbb-binarize \
        -I OCR-D-IMG \
        -O OCR-D-BIN \
        -P model default-2021-03-09
```

Consult the OCR-D processor documentation for the available parameters and model identifiers.

### 6.3 OCR-D resources

The OCR-D resource manager can be used to download and manage processor resources:

```bash
docker run --rm \
    -v $vol_models \
    ghcr.io/qurator-spk/eynollah:latest \
    ocrd resmgr --help
```

Using a persistent volume avoids downloading the same resources every time a container is started.

## 7. OCR support and limitations

The standard Docker image introduced in v0.9.0 does not include the optional OCR dependencies.

The OCR implementations use different machine-learning frameworks:

- CNN-RNN OCR requires TensorFlow/Keras.
- TrOCR requires PyTorch and the corresponding Transformers dependencies.

Installing both frameworks in the same environment can lead to conflicts involving CUDA and cuDNN versions.

For this reason, OCR is not included in the standard Docker image.

Users who require OCR should use an environment in which the appropriate optional dependencies have been installed.

For native Python installations, these dependencies can be installed using:

```bash
pip install "eynollah[OCR]"
```

For more information about OCR usage and available models, see the [main Eynollah documentation](../README.md) and [model documentation](models.md).

## 8. Building a custom Docker image

The official Docker image can be rebuilt from the repository.

Clone the repository:

```bash
git clone https://github.com/qurator-spk/eynollah.git
cd eynollah
```

Build the Docker image:

```bash
make docker
```

By default, the build uses the CUDA-enabled ONNX base image:

```text
ocrd/core-cuda-onnx:v3.13.2
```

The base image can be changed through the `DOCKER_BASE_IMAGE` variable:

```bash
make docker \
    DOCKER_BASE_IMAGE=docker.io/ocrd/core-cuda-onnx:v3.13.2
```

The output image is tagged according to the Git release tag when available, or as `latest` otherwise.

To change the image name:

```bash
make docker DOCKER_TAG=my-eynollah
```

For further build options, inspect the repository Makefile.

### 8.1 Custom dependencies

The standard Docker image intentionally excludes the optional OCR dependencies.

If OCR functionality is required, a separate image should be built with a compatible base environment and the necessary OCR dependencies.

The TensorFlow and PyTorch CUDA dependencies should be checked carefully before combining them in a single image.

For this reason, the standard ONNX inference image should not be modified simply by adding the OCR extra without validating framework compatibility.

## 9. Troubleshooting

### 9.1 Docker cannot access the GPU

Check the NVIDIA drivers on the host:

```bash
nvidia-smi
```

Check Docker GPU access:

```bash
docker run --rm --gpus all \
    nvidia/cuda:12.0.0-base-ubuntu22.04 \
    nvidia-smi
```

If the second command fails, check the NVIDIA Container Toolkit installation and Docker runtime configuration.

### 9.2 Models cannot be found

Check that the model volume is mounted correctly:

```bash
docker volume inspect ocrd-resources
```

Also verify that the requested model identifier or model directory corresponds to the selected Eynollah release.

### 9.3 Repeated TensorRT initialization

TensorRT may need to initialize and compile optimized execution plans during the first inference run.

Ensure that the configuration/cache directory is persistent between container executions.

### 9.4 Permission problems

Files created inside the container may have different ownership from those on the host.

If necessary, configure the container user and permissions appropriately.

### 9.5 Insufficient GPU memory

Reduce the workload by processing fewer pages in parallel or using a smaller number of concurrent jobs.

Device selection and processing options can be inspected using:

```bash
docker run --rm \
    ghcr.io/qurator-spk/eynollah:latest \
    eynollah --help
```

## 10. Version compatibility

Docker images and pretrained models should be selected consistently.

In particular:

| Version | Relevant Docker considerations                                                                                        |
|---------|-----------------------------------------------------------------------------------------------------------------------|
| v0.5.0  | Existing standalone and OCR-D workflows                                                                               |
| v0.6.0  | Dedicated training CLI introduced; inference remains separate                                                         |
| v0.7.0  | Model zoo introduced                                                                                                  |
| v0.8.0  | Model overrides and improved GPU configuration                                                                        |
| v0.9.0  | ONNX/TensorRT inference becomes the default; Docker image changes to `ocrd/core-cuda-onnx`; OCR dependencies excluded |
| v0.9.1  | Requires OCR-D core 3.13.2 or later                                                                                   |
| v0.9.2  | Docker workflow remains based on the v0.9 ONNX image                                                                  |

For reproducible processing, use a release-specific Docker image and the corresponding inference model package.

Do not assume that model packages from older releases are compatible with newer Docker images.

## 11. Further documentation

- [Eynollah repository](https://github.com/qurator-spk/eynollah)
- [Eynollah installation and usage](../README.md)
- [Model documentation](models.md)
- [OCR-D processor documentation](ocrd.md)
- [Training documentation](train.md)
- [Eynollah changelog](../CHANGELOG.md)
- [Docker documentation](https://docs.docker.com/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/)
- [OCR-D CLI specification](https://ocr-d.de/en/spec/cli)

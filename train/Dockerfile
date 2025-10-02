# Use NVIDIA base image
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Set the working directory
WORKDIR /app


# Set environment variable for GitPython
ENV GIT_PYTHON_REFRESH=quiet

# Install Python and pip
RUN apt-get update && apt-get install -y --fix-broken && \
    apt-get install -y \
    python3 \
    python3-pip \
    python3-distutils \
    python3-setuptools \
    python3-wheel && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Specify the entry point
CMD ["python3", "train.py", "with", "config_params_docker.json"]

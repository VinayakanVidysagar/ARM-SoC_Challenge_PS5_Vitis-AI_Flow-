Vitis AI PyTorch Tutorial for ZCU104 / ZCU102

This guide provides step-by-step instructions to set up the AMD Vitis AI toolchain using Docker, quantize a pre-trained PyTorch ResNet18 model, compile it for the DPU, and deploy it on a ZCU104 or ZCU102 board.
Overview

The Vitis AI environment requires a tightly controlled combination of software dependencies (specific Python, CUDA, deep-learning frameworks, etc.). Installing these directly on a host system can lead to conflicts. Docker provides an isolated, pre-validated container (Vitis 2022.2) that ensures compatibility and enables GPU acceleration when available. All steps below assume you are working with a ZCU104 board (adjust architecture files accordingly for ZCU102).
Prerequisites

    Linux host machine (Ubuntu 20.04 recommended)

    Docker Engine installed (see official documentation)

    User added to docker group

    SD card (≥16 GB) and reader

    Ethernet connection between host and target board

    (Optional) USB webcam for real-time inference

Install Docker

Follow the official Docker installation guide for your operating system. After installation, add your user to the docker group:
bash

sudo usermod -aG docker $USER
newgrp docker

Verify Docker Installation

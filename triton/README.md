# Triton

## Check NVIDIA Driver

> nvidia-smi

## Install CUDA Toolkit

fetch NVIDIA apt

```sh
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
```

CUDA 12.6 (better compatible with Triton)

> sudo apt-get -y install cuda-toolkit-12-6

Add to PATH:

```sh
echo 'export PATH=/usr/local/cuda-12.6/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

Check CUDA Toolkit:

> nvcc --version

## Install Triton

> pip install triton

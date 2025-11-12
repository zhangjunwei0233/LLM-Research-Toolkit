# This environment is setup for using vllm


## Introduction
In order to be performant, vLLM has to compile many cuda kernels. The compilation
unfortunately introduces binary incompatibility with other CUDA versions and PyTorch
versions, even for the same PyTorch version with different building configurations.

Therefore, it is recommended to install vLLM with a **fresh new** environment, letting
vllm decide witch pytorch verison to use. If either you have a different CUDA version
or you want to use an existing PyTorch installation, you need to build vLLM from source.
In this case, you will need to have `cuda-toolkit` installed.

## Setup

1. create a new python env using `uv`

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
```

2. install vllm and its dependencies automatically

```bash
uv pip install vllm --torch-backend=auto
```

3. run shell script to verify install

```bash
python ./check_setup.py
```
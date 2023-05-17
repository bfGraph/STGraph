# Installation Guide

This comprehensive guide will walk you through the installation process for Seastar, along with its essential prerequisites, enabling you to train GNN models on your local machine. Let's get started.

## Seastar Installation

To install Seastar on your local machine, execute the following command:

```
git clone https://github.com/bfGraph/Seastar.git
cd Seastar
```

### Seastar Virtual Environment

It is highly recommended to create a dedicated Python virtual environment for running and developing with Seastar. To create a virtual environment named `seastar`, you can use either conda or venv.

Using conda:

```
conda create --name seastar
conda activate seastar
```

### Installing Python Packages

To install the necessary Python packages for Seastar, run the following command. Ensure that you have activated the `seastar` virtual environment before installing these packages.

```
pip install -r requirements.txt
```

By executing the above instructions, you will have successfully installed Seastar along with all the required packages.

## CUDA Python

Seastar leverages CUDA Python, which encompasses a standardized collection of low-level interfaces, granting complete coverage of and access to the CUDA host APIs within Python. Prior to installing CUDA Python, ensure that your system meets the following requirements:

**Note:**
> Seastar has undergone testing with Python CUDA version 11.7 and above. This installation guide is based on the latest release of Python CUDA, namely version 12.1.0. You can also install Python CUDA for versions 11.7 and above.

### System Requirements

1. **Driver**: Linux (450.80.02 or later) / Windows (456.38 or later)
2. **CUDA Toolkit**: Versions 12.0 to 12.1
3. **Python**: Versions 3.8 to 3.11

### Installation from PyPI

Execute the following command to install CUDA Python via PyPI:

```
pip install cuda-python
```

### Verifying the Installation

To validate the successful installation of Python CUDA, run the provided Python script located within the Seastar directory:

```
cd seastar/compiler/code_gen/
python3 cuda_check.py
```

If the installation was successful, you should observe the following output. However, please note that the specific output may differ depending on the GPU present in your machine:

```

       Device Property   Value
───────────────────────────────────────────────────
     Number of Devices   1
                  Name   NVIDIA GeForce MX350
    Compute Capability   6.1
  Multiprocessor Count   5
    Concurrent Threads   10240
             GPU Clock   1468.0 MHz
          Memory Clock   3504.0 MHz
          Total Memory   2047.875 MiB
           Free Memory   1641.7216796875 MiB

Note: If either the Total Memory or Free Memory shows 0,
      it indicates that no context has been loaded into the device.
```

With this, you have now completed the installation of Seastar and verified the presence of CUDA Python on your system.

## Running Seastar

To ensure the successful installation of Seastar, let's proceed with running Seastar by training a T-GCN model on the EnglandCOVID dataset.

```
cd benchmarking/tgcn/seastar-dynamic/
python3 train.py --type naive --num_epochs 10
```

Upon executing the above command, you should observe the following output:

![Seastar Verification Output](assets/Seastar%20verification%20output.png)

If you encounter any errors while attempting to train the T-GCN model, kindly raise an issue, and our team will promptly assist you.
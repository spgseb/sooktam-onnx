# ONNX runtime execution code for Sooktam-2

## References
1. [ONNX Runtime](https://onnxruntime.ai/)
1. [NCasT4_v3-series virtual machines](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/ncast4v3-series?tabs=sizebasic)
1. [TensorRT Execution Provider](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html)
1. [CUDA Execution Provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
1. [Real Time Factor (RTF)](https://openvoice-tech.net/index.php/Real-time-factor)

## Pre-requisites
1. Hardware: GPU-enabled machine
    - Tested on NCasT4_v3-series virtual machine (VM) on the Azure cloud.
    - 1 NVIDIA Tesla T4 GPU, 4 AMD EPYC 7V12 CPUs, 28 GB RAM
2. Software: Ubuntu with CUDA-12 and python3
    - Tested on Ubuntu 22.04.1 with CUDA 12.2.140
    - libnvinfer-10 installed (python3-libnvinfer-dev)
    - Python version 3.10.12

## How to setup
1. Repo is a fork created from  https://github.com/Bharatgen-Tech/sooktam-onnx.git)  
1. Clone the repository (```$WORK``` represents the directory to be used)
    ```
        cd $WORK
        git lfs install
        git clone https://github.com/spgseb/sooktam-onnx  
        cd sooktam-onnx  
        git checkout spg  
    ```
1. Create a virtual environment (```$VENV``` represents the directory to be used) and install the python packages
    ```
        rm -rf $VENV/sooktam2-onnx
		python3 -m venv $VENV/sooktam2-onnx
        source $VENV/sooktam2-onnx/bin/activate
		pip install -r requirements.txt
    ```

## How to run
1. Activate the virtual environment
    ```
        cd $WORK
        cd sooktam-onnx/F5_TTS
        source $VENV/sooktam2-onnx/bin/activate
    ```

1. Run the code
    ```
        python3 F5-TTS-ONNX-Inference-GPU.py
    ```
    - This converts the text to speech. The speech file is stored in ```../outputs/generated_audio_gpu.wav```
    - The code uses the *TensorrtExecutionProvider*. The first run compiles the code, so it may be substantially slower. The *RTF* is printed to gauge the real-time performance.
    - The compiled output is cached in the files ```../outputs/TensorrtExecutionProvider_*```. To force recompile, delete these files.
    - To use the *CUDAExecutionProvider*, set the ```trt_flag``` in the file to ```False```

1. Run the regression to measure execution speed
    ```
        python3 F5-TTS-ONNX-LoadTest-Hindi-GPU.py
    ```
    - This runs multiple files for testing.
    - Speech files are stored in ```../outputs```
    - A *.csv* and a *.json* file with measurements are also stored there.




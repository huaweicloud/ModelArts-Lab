# dawnbench_inference_imagenet
## run inference task on ImageNet

The following instructions show how to achieve the performance that we submitted to DAWNBench step by step.
1. install CUDA 10 and CUDNN 7, TensorRT 6 and TensorFlow 1.13
```
    download CUDA 10.0.130 for Ubuntu 16.04  (https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda-repo-ubuntu1604-10-0-local-10.0.130-410.48_1.0-1_amd64)
    download and install CUDNN 7.6.3.30 for Ubuntu 16.04 (https://developer.nvidia.com/compute/machine-learning/cudnn/secure/7.6.5.32/Production/10.0_20191031/Ubuntu16_04-x64/libcudnn7_7.6.5.32-1%2Bcuda10.0_amd64.deb)
    download and install TensorRT 6.0.1.5 for Ubuntu 16.04 (https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/6.0/GA_6.0.1.5/local_repos/nv-tensorrt-repo-ubuntu1604-cuda10.0-trt6.0.1.5-ga-20190913_1-1_amd64.deb)
```
2. install dependencies
```
	opencv==3.4.3
	libjpeg-turbo==2.0.1
	python==3.5.2 
	numpy==1.17.3 
	tensorflow==1.13.1 
```
3. Download ModelArts-AIBOX from https://modelarts-labs.obs.cn-north-1.myhuaweicloud.com/tools/modelarts_aibox-0.13.0-py3-none-any.whl. 
4. Run
```bash
	pip install modelarts_aibox-0.13.0-py3-none-any.whl
	export LD_LIBRARY_PATH=/usr/local/lib/python3.5/dist-packages/modelarts/aibox:/usr/local/lib/python3.5/dist-packages/modelarts/aibox/operator/:/usr/local/cuda/lib64/:${LD_LIBRARY_PATH}
	nvidia-smi -pm 1
```
5. Clone this repo. 
6. Modify 'DATA_DIR, UFF_FILE, CALIB_FILE' in inference_aibox.py if necessary.
7. Run `python inference_aibox.py`.

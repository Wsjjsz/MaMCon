# MaMCon: Protein Contact Prediction
![test.png](..%2F..%2FNova_fig%2FMaMCon_Nova%2Ftest.png)
# Installation
Clone the repository and install it:
https://github.com/Wsjjsz/MaMCon.git
# MaMCon Environment Configuration:
> conda create -n MaMCon python=3.10.13
> 
> conda activate MaMCon
> 
> conda install cudatoolkit==11.8 -c nvidia
> 
> pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
> 
> conda install -c "nvidia/label/cuda-11.8.0" cuda-nvcc
> 
> conda install packaging
> 
> pip install causal-conv1d==1.4.0  # "Choose the version number based on the actual situation, or install the latest version without specifying a version."
> 
> pip install mamba-ssm==2.2.2  # "Choose the version number based on the actual situation, requires CUDA > 12.2."
> 
> pip install numpy==1.26.3
> 
> pip install esm
> 
> pip install Bio
> 
> pip install matplotlib
> 
> pip install pandas==2.2.2
> 
# Usage
* Please use the script for prediction:
> python prediction.py

* Please use the script for train:
> python train.py
<div align="center">
      <h1>SimpleClip</h1>
</div>

<div align="center">
    <p align="center">
          <em> Open-source / Simple / Lightweight / Easy-to-use / Extensible </em>
    </p>
</div>

<hr>

# Introduction

**This repository provides pytorch training examples for clip model.**

# Training GPU server

# Environments

**1、Python and Pytorch Supported Version: Python>=3.12, Pytorch>=2.5.1.**

**2、(optional)Add HF_HOME dir HF_ENDPOINT dir in .bashrc and .zshrc:**
```
# Add HF_HOME dir and HF_ENDPOINT dir in .bashrc and .zshrc files:
export HF_HOME=/root/autodl-tmp/cache
export HF_ENDPOINT=https://hf-mirror.com
```
```
source .bashrc
source .zshrc
```

**3、Create a conda environment:**
```
conda create -n SimpleClip python=3.12
```

**4、Install PyTorch:**
```
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
```
To install a different PyTorch version, find command from here:

https://pytorch.org/get-started/previous-versions/

**5、Install other Packages:**
```
pip install -r requirements.txt
```

# Download my pretrained models and experiments records


# Prepare datasets


# How to train or test a model


# Reference

```
https://github.com/openai/CLIP
https://github.com/mlfoundations/open_clip
```

# Citation

If you find my work useful in your research, please consider citing:
```
@inproceedings{zgcr,
 title={SimpleAICV-pytorch-training-examples},
 author={zgcr},
 year={2020-2030}
}
```
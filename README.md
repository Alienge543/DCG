# DCG


> [**Distributional Consistency-Guided Prompt Learning for Vision-Language Models**]() <br>


<hr />


## Installation

```bash
# Installation borrowed from https://github.com/muzairkhattak/multimodal-prompt-learning
# Create a conda environment
conda create -y -n DCG python=3.10

# Activate the environment
conda activate DCG

# Install torch (requires version >= 1.8.1) and torchvision
# Please refer to https://pytorch.org/ if you need a different cuda version
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html


# Clone this repo
git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
cd Dassl.pytorch/

# Install dependencies
pip install -r requirements.txt

# Install this library (no need to re-build if the source code is modified)
python setup.py develop
cd ..

# Clone CoPrompt code base
git clone https://github.com/Alienge543/DCG.git

cd DCG
# Install requirements

pip install -r requirements.txt

# Update setuptools package
pip install setuptools==59.5.0

```

## Dataset

Please follow the [CoOp](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md) repo to prepare the datasets. Set your data directory in the `scripts/base2new_train_DCG.sh` and `scripts/base2new_train_DCG.sh` file.

## Running the experiments

```bash
exp_name=DCG
trainer=DCG
train_bash=scripts/base2new_train_DCG.sh
test_bash=scripts/base2new_test_DCG.sh

export PYTHONPATH="$PYTHONPATH:$PWD"


for seed in 1 
do
  bash $train_bash caltech101 $seed $exp_name
  bash $test_bash caltech101 $seed $exp_name 
done
```

                                        

## Acknowledgements

Our code is based on [MaPLe](https://github.com/muzairkhattak/multimodal-prompt-learning) and [CoPrompt](https://github.com/ShuvenduRoy/CoPrompt.git) repository. We thank the authors for releasing their code. If you use our model and code, please consider citing these works as well.



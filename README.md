# Adversarially Guided Actor-Critic (AGAC)

This repository contains an implementation of AGAC, as introduced
in [Adversarially Guided Actor-Critic](https://openreview.net/forum?id=_mQp5cr_iNy) (ICLR 2021).


This is the original **TensorFlow** implementation.  
Find the **PyTorch** implementation [here](https://github.com/yfletberliac/adversarially-guided-actor-critic/tree/main/agac_torch).

## Installation

```
# create a new conda environment
conda create -n agac python=3.7
conda activate agac 

# install dependencies
git clone git@github.com:yfletberliac/adversarially-guided-actor-critic.git
cd adversarially-guided-actor-critic
pip install -r requirements.txt
```

## Train AGAC on MiniGrid

```
python run_minigrid.py
```

## Train AGAC on Vizdoom
(follow [Vizdoom install](https://github.com/yfletberliac/adversarially-guided-actor-critic#vizdoom-install) first)
```
python run_vizdoom.py
```

### Vizdoom install

#### Ubuntu

```
sudo apt-get install cmake libboost-all-dev libgtk2.0-dev libsdl2-dev python-numpy
pip install -e git://github.com/yfletberliac/vizdoomgym.git#egg=vizdoomgym
```

#### macOS

```
brew install cmake boost sdl2
pip install vizdoom==1.1.8
pip install pyglet==1.5.11 -e git://github.com/yfletberliac/vizdoomgym.git#egg=vizdoomgym
```

## Citation

```
@inproceedings{
flet-berliac2021adversarially,
title={Adversarially Guided Actor-Critic},
author={Yannis Flet-Berliac and Johan Ferret and Olivier Pietquin and Philippe Preux and Matthieu Geist},
booktitle={International Conference on Learning Representations},
year={2021},
}
```

## Acknowledgements

The code is an adaptation of [Stable Baselines](https://github.com/hill-a/stable-baselines).  
Thank you to [@cibeah](https://github.com/cibeah) for the PyTorch implementation.

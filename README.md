# Adversarially Guided Actor-Critic

This is an implementation of the method proposed
in [Adversarially Guided Actor-Critic](https://openreview.net/forum?id=_mQp5cr_iNy), published at ICLR 2021.

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

## Train AGAC on Vizdoom (follow "Vizdoom install" first)

```
python run_vizdoom.py
```

### Vizdoom install

#### Ubuntu

```
sudo apt install cmake libboost-all-dev libsdl2-dev libfreetype6-dev libgl1-mesa-dev libglu1-mesa-dev libpng-dev libjpeg-dev libbz2-dev libfluidsynth-dev libgme-dev libopenal-dev zlib1g-dev timidity tar nasm
pip install -e git://github.com/shakenes/vizdoomgym.git#egg=vizdoomgym
```

#### macOS

```
brew install cmake boost sdl2
pip install pyglet==1.5.11 -e git://github.com/shakenes/vizdoomgym.git#egg=vizdoomgym
```

NB: if error "TypeError: 'Box' object is not iterable", that means
this [PR](https://github.com/shakenes/vizdoomgym/pull/8/files) has not yet been merged. Making the corresponding 7-line
change fixes the issue.

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
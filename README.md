# Installation
## Git
```bash
git@github.com:esevegne/astro-nn.git
```
## Conda and environments
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
```
```bash
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh
```
```bash
conda env create -f environment.yml
conda activate astro-nn
```
## Jupyter Online
```bash
tmux new -s my_sess
cd ~/Workspace
conda activate astro-nn
jupyter notebook --no-browser --port=8080 &
```
# Repo structure
This repo is structured is 5 main file.
## data_prep.py
This file is made for importing the data necessary for training.
"sats" is a list of strings like ["mms1","mms2","mms3"] which specify which spacecraft to import.
"times" is a list of 2-tuple for starting time and ending time of each period you want to import.
"data_path" is a string to specify the path where to save the data in hdf5 format.
data_prep.ipynb does a similar job but in an interactive notebook.

## model.py
model.py is a file where I save the models structure I want to use to have a cleaner code in training.ipynb.

## utils.py
Is a script with usefull routines for analysis or training.

## training.ipynb
Core file of the code, edit the model_config.toml to train the model with the choosen parameters.

## analysis.ipynb
Code to analysis a model and to retrieve the results presented in the paper.

# Data availability
Data files (.hdf5) are not pushed to github because they are heavy. If needed they can be requested on demand or recreated with the correct time periods and data_prep.py

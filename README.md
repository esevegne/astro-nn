# Installation
## GIT

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

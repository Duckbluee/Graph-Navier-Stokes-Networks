# Graph-Navier-Stokes-Networks
Anonymous code release for double-blind review.

## Set environment
The environment can be set up using either `environment.yml` file or manually installing the dependencies.
### Using an environment.yml file
```
conda env create -f environment.yml
```
### Manually install
```
conda create -n gread python=3.9
conda activate gread
pip install torch==1.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install torch-scatter torch-sparse torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
pip install torchdiffeq ogb wandb deeprobust==0.2.4
```
or

`conda env create -f env.yml`

## How to run
To run each experiment, navigate into `src`. Then, run the following command:
`python Reproduce_Cora.py`/`python Reproduce_Texas.py`

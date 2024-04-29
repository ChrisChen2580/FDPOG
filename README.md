# FDPOG
The code will be continuously updated
## Prerequisites

- Install pip
- Python 3.8

- PyTorch 1.11.0

- functorch 0.1.1

```bash
pip install functorch==0.1.1
```

- opacus 1.1

```bash
conda install -c conda-forge opacus=1.1
```

- matplotlib 3.4.3

```bash
conda install -c conda-forge matplotlib=3.4.3
```

- Other requirements

```bash
pip install pandas tbb regex tqdm tensorboardX=2.2
pip install tensorboard==2.9
```

Scripts to reproduce experiments located at FDPOG/experiment_scripts, results saved to FDPOG/runs.

## MNIST

```
bash ./experiment_scripts/mnist_script.sh
```

when compared with AUTO-S, it needs to use gradient normalization, please refer to [Automatic clipping: Differentially private deep learning made easier and stronger](https://proceedings.neurips.cc/paper_files/paper/2023/hash/8249b30d877c91611fd8c7aa6ac2b5fe-Abstract-Conference.html)


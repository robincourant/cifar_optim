# CIFAR10 Optimization

## Requirements :

- python 3.8
- pip version ≥ 20.0

## Installation :

```
pip install -r requirements.txt
```

## Usage

Example of command to launch the main:
```
python src/main.py minicifar_scratch naive_convnet --epochs 15
```

To monitor metrics with `tensorboard`:
```
tensorboard --logdir=logs
```
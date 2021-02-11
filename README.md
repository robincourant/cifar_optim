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

To monitor metrics with `tensorboard.dev` (online):
```
tensorboard dev upload --logdir logs  # Load a directory
tensorboard dev list  # Get experiment list (especially IDs)
tensorboard dev delete --experiment_id EXPERIMENT_ID # Delete an experiement
```
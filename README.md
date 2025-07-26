# L3M

Repository for the Lightcone Large Language Model (L3M) [2506.14757](https://arxiv.org/abs/2506.14757)

# Regression

To run the regression task, execute

```
python src/experiments/regression/training_script.py --config CONFIG_FILE --output_dir runs/regression
```

after replacing the CONFIG_FILE with the path to the config file of the experiment. An example file is provided in in config/regression/example.json.

# Generation

To run the generation task, execute

```
python src/experiments/generation/training_script.py --config CONFIG_FILE --output_dir runs/generation
```

after replacing the CONFIG_FILE with the path to the config file of the experiment. An example file is provided in config/generation/example.json and config/generation/example_lora.json.

# Dataset

Your dataset must be a directory, where each lightcone is stored as a .npz file. It must have the keys
- "image": the brightness temperature distribution must be stored as a 3d array with shape [spatial x, spatial y, time], where the time direction evolves from low redshift to high redshift.
- "labels": the simulation parameters must be stored as a 1d array in the order [m, Om, log LX, E0, log Tvir, zeta].

# Pretrained Qwen2.5 models

You must download the pretrained Qwen2.5 models from huggingface into a folder. This older must contain the weights and the config file. This folder must then be specifig in the config file of the run.

# Config files

The documentation of the config file can be found in src/utils/exp_config.py, src/experiments/regression/reg_exp_config.py and src/experiments/generation/gen_exp_config.py.

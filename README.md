# DiffusionPDE

## Dependencies
See the list of dependencies in `environment.yml`
Install to a new conda environement called `ddpm` using `conda env create -f environment.yml`. Activate the environment using `conda activate ddpm`

## Dataset generation
I have written a highly paralelized solver for 2D darcy flow in which we can control the underlying dimension of the parameterization. Generate datasets using `data_generation/darcy_flow/generate_darcy.py`

## Training
To train a diffusion model, see the configuration file in `configs/darcy_flow.yaml`
I have designed the code to be very flexible. Any SDE formulations can be easily and simply adapted along with any denoising model architectures. This is very useful for rapidly prototyping and experimenting.
New datasets can be easily implemented, but they require a pytorch-lighting data module to be created
The training utilizes parallelized strategies in pytorch-lightning, utilizing many efficiency improvements such as automatic mixed precision (AMP), DDP for small models, and DeepSpeed for large models which must be sharded across gpus.

Train the model by creating a configuration file and running `python train.py --logdir /path/to/logs --config /path/to/config.yaml0 --name experiment_name`

To view training, use tensorboard via `tensorboard --logdir /path/to/logdir`. This shows all training losses as well as samples from the reverse sde during training


'''
Training script for score-matching generative models
Author: Christian Jacobsen, University of Michigan 2023
'''

import torch
import datetime
import os
import os.path as osp
import glob
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
from omegaconf import OmegaConf
from utils import instantiate_from_config


def get_parser():
    parser = argparse.ArgumentParser(description='training inputs')
    parser.add_argument('--name', type=str, help='postfix for logdir')
    parser.add_argument('--logdir', type=str, help='path to log to')
    parser.add_argument('--config', type=str, help='path to config file')
    parser.add_argument('--seed', type=int, default=101, help='global seed for reproducibility')
    parser.add_argument('--resume', '-r', type=str, default="", help='resume from logdir or log file')
    parser.add_argument('--unet_weights', type=str, help="load unet weights only from *.pth file")
    return parser

def main():
    parser = get_parser()
    known, unknown = parser.parse_known_args()

    if len(unknown) > 0:
        raise Exception('Unknown input argument(s): ', unknown)

    config = OmegaConf.load(known.config)
    

    # separate the configurations
    lightning_config = config.pop("lightning", OmegaConf.create())
    trainer_config = lightning_config.pop("trainer", OmegaConf.create())
    model_config = config.pop("model", OmegaConf.create())
    data_config = config.pop("data", OmegaConf.create())

    if known.name and known.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot both be specified."
        ) 

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    if known.resume:
        if not os.path.exists(known.resume):
            raise ValueError("Cannot find {}".format(known.resume))
        if os.path.isfile(known.resume):
            paths = known.resume.split("/")
            logdir = "/".join(paths[:-2])
            ckpt = known.resume
        else:
            raise Exception('Not a checkpoint file to resume from: ', known.resume)

        known.resume_from_checkpoint = ckpt

    else:
        if known.name:
            name = "_" + known.name
        else:
            name = ""
        nowname = now + name
        logdir = os.path.join(known.logdir, nowname)    

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(known.seed)

    # initialize the model
    model = instantiate_from_config(model_config)

    # add callback which sets up log directory
    callbacks_cfg = {
        "setup_callback": {
            "target": "loggers.SetupCallback",
            "params": {
                "resume": known.resume,
                "now": now,
                "logdir": logdir,
                "ckptdir": ckptdir,
                "cfgdir": cfgdir,
                "model_config": model_config,
                "lightning_config": lightning_config,
                "trainer_config": trainer_config,
                "data_config": data_config,
            }
        },
        "image_logger": {
            "target": "loggers.ImageLogger",
            "params": {
                "max_images": 12,
                "clamp": False
            }
        },
        "cuda_callback": {
            "target": "loggers.CUDACallback"
        },
    }

    checkpoint_callback_0 = ModelCheckpoint(dirpath=osp.join(ckptdir), every_n_epochs=lightning_config.save_every_n_epochs, save_on_train_epoch_end=True, filename="last")
    #checkpoint_callback_1 = ModelCheckpoint(dirpath=osp.join(ckptdir), save_top_k=5, monitor="train/loss_epoch")
    checkpoint_callback_1 = ModelCheckpoint(
                                        dirpath=osp.join(ckptdir),
                                        save_top_k=1,  # Keeps only the best checkpoint
                                        monitor="val/loss_score", 
                                        mode="min",  
                                        filename="best_val", 
                                        verbose=True
                                    )

    trainer_kwargs = dict()
    trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]
    trainer_kwargs["callbacks"].append(checkpoint_callback_0)
    trainer_kwargs["callbacks"].append(checkpoint_callback_1)

    torch.set_float32_matmul_precision('medium')
    
    # set the correct gpu devices
    try:
        gpus = trainer_config.devices.split(',')
    except:
        gpus = [str(trainer_config.devices)]

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(trainer_config.devices)
    trainer = Trainer(max_epochs=lightning_config.epochs,
                      accelerator=trainer_config.accelerator,
                      strategy=trainer_config.strategy,
                      devices=len(gpus),
                      precision=trainer_config.precision,
                      default_root_dir=logdir,
                      **trainer_kwargs)

    dataloader = instantiate_from_config(data_config)

    # load unet weights?
    if known.unet_weights is not None:
        print('Loading unet model weights from .pth file.')
        model.unet.load_state_dict(torch.load(known.unet_weights))

    # lock the base model?
    if lightning_config.lock_base:
        for name, param in model.named_parameters():
            if ('zero_conv' not in name) and ('control' not in name):
                param.requires_grad = False

    if known.resume:
        if known.unet_weights is not None:
            raise Exception("Cannot resume training from checkpoint and load unet weights")
        trainer.fit(model, dataloader, ckpt_path=known.resume_from_checkpoint)
    else:
        trainer.fit(model, dataloader)

if __name__ == "__main__":
    main()

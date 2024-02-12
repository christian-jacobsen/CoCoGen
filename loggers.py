'''
Logger callback for pytorch lightning
Author: Christian Jacobsen, University of Michigan 2023
'''

import torch
import time
import os
import os.path as osp
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only, rank_zero_info
import torchvision
from PIL import Image
from omegaconf import OmegaConf
from utils import convert_to_rgb

class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, model_config, lightning_config, trainer_config, data_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.model_config = model_config
        self.lightning_config = lightning_config
        self.trainer_config = trainer_config
        self.data_config = data_config

    def on_keyboard_interrupt(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            if "callbacks" in self.lightning_config:
                if 'metrics_over_trainsteps_checkpoint' in self.lightning_config['callbacks']:
                    os.makedirs(os.path.join(self.ckptdir, 'trainstep_checkpoints'), exist_ok=True)
            print("Model config")
            print(OmegaConf.to_yaml(self.model_config))
            OmegaConf.save(self.model_config,
                           os.path.join(self.cfgdir, "{}-model.yaml".format(self.now)))

            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

            print("Trainer config")
            print(OmegaConf.to_yaml(self.trainer_config))
            OmegaConf.save(self.trainer_config,
                           os.path.join(self.cfgdir, "{}-trainer.yaml".format(self.now)))

            print("Data config")
            print(OmegaConf.to_yaml(self.data_config))
            OmegaConf.save(self.data_config,
                           os.path.join(self.cfgdir, "{}-data.yaml".format(self.now)))

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass


class ImageLogger(Callback):
    def __init__(self, max_images, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        #self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            pl.loggers.TensorBoardLogger: self._testtube,
        }
        #self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        #if not increase_log_steps:
        #    self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    @rank_zero_only
    def _testtube(self, pl_module, images, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=images[k].shape[0]//self.channels)
            #grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w

            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(
                tag, grid,
                global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=images[k].shape[0]//self.channels)
            '''
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            '''
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}.png".format(
                k,
                global_step,
                current_epoch
                )
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, split="train"):
        logger = type(pl_module.logger)

        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        with torch.no_grad():
            images = pl_module.log_images(batch, **self.log_images_kwargs)

        all_images = {}

        for k in images:
            self.channels = images[k].shape[1]
            N = min(images[k].shape[0], self.max_images)
            images[k] = images[k][:N]

            # extract each channel separately
            all_images[k] = []
            for channel in range(images[k].shape[1]):
                all_images[k].append(images[k][:, channel].unsqueeze(1))
            all_images[k] = torch.cat(all_images[k], dim=0)
            
            if isinstance(all_images[k], torch.Tensor):
                all_images[k] = all_images[k].detach().cpu()
                if self.clamp:
                    all_images[k] = torch.clamp(all_images[k], -1., 1.)

        # convert to rgb if grayscale (already on cpu)
        if all_images["inputs"].shape[1] == 1:
            all_images = convert_to_rgb(all_images)

        self.log_local(pl_module.logger.save_dir, split, all_images,
                        pl_module.global_step, pl_module.current_epoch)

        logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
        logger_log_images(pl_module, all_images, split)

        if is_train:
            pl_module.train()

    '''
    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (
                check_idx > 0 or self.log_first_step):
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                print(e)
                pass
            return True
        return False
    '''

    def on_train_start(self, trainer, pl_module):
        self.sampled_train_batches = self.sample_batches(trainer.train_dataloader)

    def on_validation_start(self, trainer, pl_module):
        self.sampled_val_batches = self.sample_batches(trainer.val_dataloaders)

    def sample_batches(self, dataloader):
        iterator = iter(dataloader)
        batch = next(iterator)
        return batch

    def on_train_epoch_end(self, trainer, pl_module): 
        self.log_img(pl_module, self.sampled_train_batches, split="train")

    def on_validation_epoch_end(self, trainer, pl_module):
        self.log_img(pl_module, self.sampled_val_batches, split="val")


class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.strategy.root_device.index)
        torch.cuda.synchronize(trainer.strategy.root_device.index)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        torch.cuda.synchronize(trainer.strategy.root_device.index)
        max_memory = torch.cuda.max_memory_allocated(trainer.strategy.root_device.index) / 2 ** 20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass

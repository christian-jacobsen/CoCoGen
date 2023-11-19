'''
Add controlnet layers to an existing unconditional model
Author: Christian Jacobsen, University of Michigan 2023
'''

import torch
from omegaconf import OmegaConf
import argparse
from utils import instantiate_from_config

def parse_args():
    parser = argparse.ArgumentParser(description='Add controlnet to unconditional model')
    parser.add_argument('--config', type=str, help='path to model configuration file *.yaml')
    parser.add_argument('--checkpoint', type=str, help='path to pretrained unconditional model checkpoint')
    parser.add_argument('--save', type=str, help='path to save the controlnet-augmented model weights to. Should be *.pth')
    return parser.parse_args()

def main():
    args = parse_args()
    config = OmegaConf.load(args.config)

    if config.model.params.unet_config.params.controlnet:
        print('Adding randomly initialized controlnet to pretrained model weights...')
        model_config = config.pop("model", OmegaConf.create())

        # initialize a random model with controlnet and get unet state dict
        model = instantiate_from_config(model_config)
        example_state_dict = model.unet.state_dict()

        # get old parameter values 
        ckpt = torch.load(args.checkpoint)
        old_pl_state_dict = ckpt['state_dict']
        old_state_dict = {}
        new_state_dict = {}

        '''
        # testing for locking the base model
        for name, param in model.named_parameters():
            if ('zero_conv' not in name) and ('control' not in name):
                param.requires_grad = False

        for name, param in model.named_parameters():
            print(name, "requires grad", param.requires_grad)
        '''

        # remove all keys that aren't in the unet
        for k in old_pl_state_dict.keys():
            if k.startswith('unet'):
                old_state_dict[k[5:]] = old_pl_state_dict[k]

        # copy old parameter values to new ones where we can
        for k in example_state_dict.keys():
            if k in old_state_dict.keys():
                new_state_dict[k] = old_state_dict[k]
            else:
                print("Adding controlnet weights ", k, "randomly.")
                new_state_dict[k] = example_state_dict[k]

        print('Saving ControlNet-augmented state dict to ', args.save)
        torch.save(new_state_dict, args.save)

    else:
        print("Model configuration specifies controlnet = False")
        print("No modification needed.")


if __name__ == "__main__":
    main()


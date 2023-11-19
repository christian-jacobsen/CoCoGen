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
    parser.add_argument('--checkpoint1', type=str, help='path to pretrained unconditional model checkpoint')
    parser.add_argument('--checkpoint2', type=str, help='path to trained model with controlnet checkpoint')
    return parser.parse_args()

def main():
    args = parse_args()

    # get old parameter values 
    ckpt1 = torch.load(args.checkpoint1)
    ckpt2 = torch.load(args.checkpoint2)
    state_dict1 = ckpt1['state_dict']
    state_dict2 = ckpt2['state_dict']

    total_norm = 0
    for k in state_dict1.keys():
        if k in state_dict2.keys():
            if ('num_batches' not in k) and ('running' not in k):
                norm = torch.mean((state_dict1[k].float() - state_dict2[k].float())**2)
                print(k, ", norm: ", norm)
                total_norm += norm

    for k in state_dict2.keys():
        if 'zero_conv' in k:
            print(k, state_dict2[k])

    print('Total norm (should be 0): ', total_norm)

if __name__ == "__main__":
    main()


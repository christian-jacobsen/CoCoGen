'''
Helpful commonly used functions
Author: Christian Jacobsen, University of Michigan 2023
'''
import importlib
import matplotlib.pyplot as plt
import numpy as np
import torch

def get_obj_from_str(string):
    module, cls = string.rsplit(".", 1)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        raise Exception("target not in config! ", config)
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def convert_to_rgb(images):
    # converts single channel pytorch data to rgb according to matplotlib colormaps
    cmap = plt.get_cmap('jet')
    converted_images = {}
    for k in images:
        converted_images[k] = []
        squeezed_images = images[k].squeeze().numpy()
        for image in squeezed_images:
            norm = plt.Normalize(vmin=np.min(image), vmax=np.max(image))
            converted_images[k].append(torch.from_numpy(cmap(norm(image))).permute(-1, 0, 1).unsqueeze(0))

        converted_images[k] = torch.cat(converted_images[k], dim=0)[:, :-1]

    return converted_images

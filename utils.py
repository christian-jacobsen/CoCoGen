'''
Helpful commonly used functions
Author: Christian Jacobsen, University of Michigan 2023
'''
import importlib
import matplotlib.pyplot as plt
import numpy as np
import torch

def zero_module(module):
    '''
    zero the parameters of a module and return it
    '''
    for p in module.parameters():
        p.detach().zero_()
    return module

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

def plot_inpainting(data, samples, sensors, path):
    batch_size = data.shape[0]

    plt.figure(1, figsize=(25, 38))
    plt.cla()
    plt.clf()
    cmap = plt.get_cmap('jet')
    cmap.set_bad('black')

    im_ratio = data.shape[3]/data.shape[2]


    # plot sensors (pressure)
    for i in range(batch_size):
        plt.subplot(7, batch_size, i+1)
        sensors[i][sensors[i]==0] = np.nan
        plt.imshow(sensors[i][0], cmap=cmap)
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.title('Sensor Locations (Pressure)')
        plt.colorbar(fraction=0.047*im_ratio)

    # plot data (pressure)
    for i in range(batch_size):
        plt.subplot(7, batch_size, batch_size+i+1)
        plt.imshow(data[i][0], cmap=cmap)
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.title('Data (Pressure)')
        plt.colorbar(fraction=0.047*im_ratio)

    # plot samples (pressure)
    for i in range(batch_size):
        plt.subplot(7, batch_size, 2*batch_size+i+1)
        plt.imshow(samples[i][0], cmap=cmap)
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.title('Samples (Pressure)')
        plt.colorbar(fraction=0.047*im_ratio)

    # plot error (pressure)
    for i in range(batch_size):
        plt.subplot(7, batch_size, 3*batch_size+i+1)
        v = np.max(np.abs(samples[i][0]-data[i][0]))
        plt.imshow(samples[i][0]-data[i][0], cmap='bwr', vmin=-v, vmax=v)
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.title('Error (Pressure)')
        plt.colorbar(fraction=0.047*im_ratio)

    # plot data (perm)
    for i in range(batch_size):
        plt.subplot(7, batch_size, 4*batch_size+i+1)
        plt.imshow(data[i][1], cmap=cmap)
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.title('Data (Permeability)')
        plt.colorbar(fraction=0.047*im_ratio)

    # plot samples (perm)
    for i in range(batch_size):
        plt.subplot(7, batch_size, 5*batch_size+i+1)
        plt.imshow(samples[i][1], cmap=cmap)
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.title('Samples (Permeability)')
        plt.colorbar(fraction=0.047*im_ratio)

    # plot error (perm)
    for i in range(batch_size):
        plt.subplot(7, batch_size, 6*batch_size+i+1)
        v = np.max(np.abs(samples[i][1]-data[i][1]))
        plt.imshow(samples[i][1]-data[i][1], cmap='bwr', vmin=-v, vmax=v)
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.title('Error (Permeability)')
        plt.colorbar(fraction=0.047*im_ratio)


    plt.savefig(path, bbox_inches='tight', dpi=300)

    # also plot means
    plt.figure(2, figsize=(19, 11))
    plt.subplot(2, 3, 1)
    plt.imshow(data[0][0], cmap=cmap)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.title('Data (Pressure)')
    plt.colorbar(fraction=0.047*im_ratio)

    plt.subplot(2, 3, 2)
    plt.imshow(np.mean(samples[:, 0, :, :], axis=0), cmap=cmap)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.title('Mean Reconstruction (Pressure)')
    plt.colorbar(fraction=0.047*im_ratio)

    plt.subplot(2, 3, 3)
    plt.imshow(np.std(samples[:, 0, :, :], axis=0), cmap=cmap)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.title('Standard Deviation Reconstruction (Pressure)')
    plt.colorbar(fraction=0.047*im_ratio)

    plt.subplot(2, 3, 4)
    plt.imshow(data[0][1], cmap=cmap)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.title('Data (Permeability)')
    plt.colorbar(fraction=0.047*im_ratio)

    plt.subplot(2, 3, 5)
    plt.imshow(np.mean(samples[:, 1, :, :], axis=0), cmap=cmap)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.title('Mean Reconstruction (Permeability)')
    plt.colorbar(fraction=0.047*im_ratio)

    plt.subplot(2, 3, 6)
    plt.imshow(np.std(samples[:, 1, :, :], axis=0), cmap=cmap)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.title('Standard Deviation Reconstruction (Permeability)')
    plt.colorbar(fraction=0.047*im_ratio)

    plt.savefig(path[:-4]+'_uncertainty.png', bbox_inches='tight', dpi=300)

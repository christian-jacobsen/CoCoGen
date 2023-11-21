import torch

def to_flattened_numpy(x):
  """Flatten a torch tensor `x` and convert it to numpy."""
  return x.detach().cpu().numpy().reshape((-1,))

def from_flattened_numpy(x, shape):
  """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
  return torch.from_numpy(x.reshape(shape))

def zero_module(module):
    '''
    zero the parameters of a module and return it
    '''
    for p in module.parameters():
        p.detach().zero_()
    return module
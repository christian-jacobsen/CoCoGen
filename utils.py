'''
Helpful commonly used functions
Author: Christian Jacobsen, University of Michigan 2023
'''
import importlib

def get_obj_from_str(string):
    module, cls = string.rsplit(".", 1)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        raise Exception("target not in config! ", config)
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

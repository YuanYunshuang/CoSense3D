import os
import json
import logging
import re
from functools import partial

import yaml
import torch
import numpy as np
from rich.logging import RichHandler


def setup_logger(exp_name, debug):
    from imp import reload

    reload(logging)
    # reload() reloads a previously imported module. This is useful if you have edited the module source file using an
    # external editor and want to try out the new version without leaving the Python interpreter.

    CUDA_TAG = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    EXP_TAG = exp_name

    logger_config = dict(
        level=logging.DEBUG if debug else logging.INFO,
        format=f"{CUDA_TAG}:[{EXP_TAG}] %(message)s",
        handlers=[RichHandler()],
        datefmt="[%X]",
    )
    logging.basicConfig(**logger_config)


def update_dict(dict_out, dict_add):
    """
    Merge config_add into config_out.
    Existing values in config_out will be overwritten by the config_add.

    Parameters
    ----------
    dict_out: dict
    dict_add: dict

    Returns
    -------
    config_out: dict
        Updated config_out
    """
    for add_key, add_content in dict_add.items():
        if add_key not in dict_out or not isinstance(add_content, dict):
            dict_out[add_key] = add_content
        else:
            update_dict(dict_out[add_key], add_content)

    return dict_out


def load_json(filename):
    with open(filename, 'r') as fh:
        data = json.load(fh)
    return data


def save_json(data, filename):
    with open(filename, 'w') as fh:
        json.dump(data, fh, indent=3)


def load_yaml(filename):
    """
    Load yaml file into dictionary.

    Parameters
    ----------
    filename : str
        Full path of yaml file.

    Returns
    -------
    params : dict
        A dictionary that contains defined parameters.
    """
    with open(filename, 'r') as stream:
        loader = yaml.Loader
        loader.add_implicit_resolver(
            u'tag:yaml.org,2002:float',
            re.compile(u'''^(?:
             [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$''', re.X),
            list(u'-+0123456789.'))
        params = yaml.load(stream, Loader=loader)
    return params


def save_yaml(data, filename):
    with open(filename, 'w') as fid:
        yaml.dump(data, fid, default_flow_style=False)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, mode=0o755)


def list_dirs(path):
    return sorted([x for x in os.listdir(path) if
                   os.path.isdir(os.path.join(path, x))])


# @gin.configurable
# def logged_hparams(keys):
#     C = dict()
#     for k in keys:
#         C[k] = gin.query_parameter(f"{k}")
#     return C


def load_from_pl_state_dict(model, pl_state_dict):
    state_dict = {}
    for k, v in pl_state_dict.items():
        state_dict[k[6:]] = v
    model.load_state_dict(state_dict)
    return model


def pad_list_to_array_np(data):
    """
    Pad list of numpy data to one single numpy array
    :param data: list of np.ndarray
    :return: np.ndarray
    """
    B = len(data)
    cnt = [len(d) for d in data]
    max_cnt = max(cnt)
    out = np.zeros(B, max_cnt, *data[0].shape[1:])
    for b in range(B):
        out[b, :cnt[b]] = data[b]
    return out


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))



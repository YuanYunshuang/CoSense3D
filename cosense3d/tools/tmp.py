import numpy as np
import torch
import glob
import os
import tqdm
from cosense3d.utils.misc import load_json, save_json

import matplotlib.pyplot as plt

files = glob.glob("/koko/cosense3d/dairv2x/*.json")
for file in files:
    data = load_json(file)
    for f, fdict in data.items():
        fdict['meta']['ego_id'] = str(fdict['meta']['ego_id'])

    save_json(data, file.replace('dairv2x', 'dairv2x-correction'))


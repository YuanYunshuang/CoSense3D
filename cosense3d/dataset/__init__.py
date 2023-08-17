import logging
import torch
import importlib


def get_dataloader(cfgs, mode='train'):
    name = cfgs['dataset']
    module = importlib.import_module(f'cosense3d.dataset.{name.lower()}_dataset')
    dataset_full_name = ''.join([n[:1].upper() + n[1:] for n in name.split('_')]) + 'Dataset'
    assert hasattr(module, dataset_full_name), "Invalid dataset."
    module_class = getattr(module, dataset_full_name)
    dataset = module_class(cfgs, mode)
    shuffle = cfgs.get('shuffle', True) if mode=='train' else False
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=cfgs[f'batch_size_{mode}'],
                                             sampler=None, num_workers=cfgs['n_workers'],
                                             shuffle=shuffle,
                                             collate_fn=dataset.collate_batch,
                                             drop_last=True)
    return dataloader





import torch
from cosense3d.model.pre_process import PreProcessorBase


class DimensionlessCoordinates(PreProcessorBase):
    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        kwargs:
            voxel_size: list
        """
        super().__init__(**kwargs)

        assert len(self.voxel_size) == 3
        if isinstance(self.voxel_size, list):
            self.voxel_size = torch.Tensor(self.voxel_size)

    def __call__(self, data_dict):
        if data_dict['pcds'] is None:
            return
        data_dict['coords'] = data_dict['pcds'][:, :4].clone()
        data_dict['coords'][:, 1:] = data_dict['coords'][:, 1:] / \
                              self.voxel_size.to(data_dict['pcds'].device).reshape(1, 3)
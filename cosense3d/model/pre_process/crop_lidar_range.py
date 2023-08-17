import torch
from cosense3d.model.pre_process import PreProcessorBase


class CropLidarRange(PreProcessorBase):
    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        kwargs:
            r: list, optional
            x: list, optional
            y: list, optional
            z: list
        """
        super().__init__(**kwargs)

    def __call__(self, data_dict):
        if data_dict['pcds'] is None:
            return
        ##########PCD###########
        mask_pcd = torch.ones_like(data_dict['pcds'][:, 0]).bool()
        # crop along axis
        for i, axis in enumerate('xyz'):
            if hasattr(self, axis):
                # shrink the range with 1e-4 to ensure ME coords rounds to the correct idx
                m_pcd = torch.logical_and(
                    data_dict['pcds'][:, i + 1] > getattr(self, axis)[0] + 1e-4,
                    data_dict['pcds'][:, i + 1] < getattr(self, axis)[1] - 1e-4
                )
                mask_pcd = torch.logical_and(mask_pcd, m_pcd)
        data_dict['pcds'] = data_dict['pcds'][mask_pcd]
        # crop according distance range
        if hasattr(self, 'r'):
            dist = torch.norm(data_dict['pcds'][:, 1:3], dim=1)
            data_dict['pcds'] = data_dict['pcds'][dist < self.r]
        ##########OBJECTS###########
        if data_dict['objects'] is not None:
            mask_obj = torch.ones_like(data_dict['objects'][:, 0]).bool()
            for i, axis in enumerate('xyz'):
                if hasattr(self, axis):
                    m_obj = torch.logical_and(
                        data_dict['objects'][:, i + 3] > getattr(self, axis)[0] + 1e-4,
                        data_dict['objects'][:, i + 3] < getattr(self, axis)[1] - 1e-4
                    )
                    mask_obj = torch.logical_and(mask_obj, m_obj)

            data_dict['objects'] = data_dict['objects'][mask_obj]

            # crop according distance range
            if hasattr(self, 'r'):
                dist = torch.norm(data_dict['objects'][:, 3:5], dim=1)
                data_dict['objects'] = data_dict['objects'][dist < self.r]
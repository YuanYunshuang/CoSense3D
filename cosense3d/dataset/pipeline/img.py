from cosense3d.dataset.pipeline.utils import ProcessNode


class LoadMultiViewImages(ProcessNode):
    def __init__(self, root_dir, **kwargs):
        super().__init__(**kwargs)
        self.root_dir = root_dir

    def __call__(self, batch_dict):
        pass
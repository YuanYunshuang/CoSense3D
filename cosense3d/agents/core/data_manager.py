from cosense3d.model.pre_process import PreProcess
from cosense3d.model.post_process import PostProcess


class DataManager:
    def __init__(self, pre_process, post_process):
        if pre_process is not None:
            self.preP = PreProcess(pre_process)
        if post_process is not None:
            self.postP = PostProcess(post_process)

    def pre_process(self, batch_dict):
        if not hasattr(self, 'preP'):
            return batch_dict
        else:
            return self.preP(batch_dict)

    def post_process(self, batch_dict):
        if not hasattr(self, 'postP'):
            return batch_dict
        else:
            return self.postP(batch_dict)


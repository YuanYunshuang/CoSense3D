from torch import nn


class ForwardRunner(nn.Module):
    def __init__(self, shared_models):
        self.shared_modules = shared_models

    def forward(self, tasks):
        """

        :param tasks: list[dict(module, input)]
        :return:
        """
        out_list = []
        for task in tasks:
            out = {'module': task['module']}
            out['output'] = self.shared_modules[task['module']](task['input'])
            out_list.append(out_list)
        return out_list


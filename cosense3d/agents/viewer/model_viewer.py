import matplotlib
import numpy as np

matplotlib.use('Qt5Agg')

from PyQt5 import QtCore, QtWidgets

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from cosense3d.utils.vislib import draw_points_boxes_plt


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, width=5, height=4, dpi=100, title='model', nrows=1, ncols=1):
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.suptitle(title, fontsize=16)
        self.axes = fig.subplots(nrows, ncols)
        super(MplCanvas, self).__init__(fig)


class WeightsStatistics(MplCanvas):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.colors = {'conv': [0, 0, 1], #blue
                       'mlp': [0, 1, 0], # green
                       'norm': [0, 1, 1], # cyan
                       'head': [1, 0.5, 0], # orange
                       'attn': [1, 0, 0]}  # red

    def refresh(self, model, **kwargs):
        modules = model.shared_modules
        xticks = []
        xlabels = []
        points_abs_max = []
        points_std = []
        markers = []
        for i, (name, module) in enumerate(modules.items()):
            xticks.append(i)
            xlabels.append(name)
            param_dict = {'conv.weight': [],'conv.bias': [],
                          'mlp.weight': [], 'mlp.bias':[],
                          'norm.weight': [], 'norm.bias':[],
                          'head.weight': [], 'head.bias':[],
                          'attn.weight': [], 'attn.bias':[],
                          }
            for n, param in module.named_parameters():
                if param.requires_grad:
                    if 'bn' in n or 'norm' in n:
                        self.append_params(param_dict, n, 'norm', param)
                    elif 'conv' in n or 'kernel' in n:
                        self.append_params(param_dict, n, 'conv', param)
                    elif 'mlp' in n or 'ffn' in n or 'emb' in n:
                        self.append_params(param_dict, n, 'mlp', param)
                    elif 'head' in n:
                        self.append_params(param_dict, n, 'head', param)
                    elif 'attention' in n or 'attn' in n:
                        self.append_params(param_dict, n, 'attn', param)
                    else:
                        pass

            for k, vs in param_dict.items():
                if len(vs) == 0:
                    continue
                vs = np.array(vs).T
                mean = vs[0].mean()
                std = vs[1].std()
                abs_max = vs[2].__abs__().max()

                if mean == 0 or mean == 1:
                    points_abs_max.append([i, mean, 500.0] + self.colors[k.split('.')[0]] + [0.5,])
                    points_std.append([i, mean, 500.0] + self.colors[k.split('.')[0]] + [1,])
                else:
                    points_abs_max.append([i, mean, abs_max * 1000] + self.colors[k.split('.')[0]] + [0.5,])
                    points_std.append([i, mean, std * 1000] + self.colors[k.split('.')[0]] + [1,])
                marker = 1 if 'weight' in k else 0
                markers.append(marker)
            # if len(points_std) == 0:
            #     continue
        points_std = np.array(points_std)
        points_abs_max = np.array(points_abs_max)
        for i, marker in enumerate(['*', 'o']):
            mask = np.array(markers) == i
            scatter = self.axes.scatter(points_abs_max[mask,0], points_abs_max[mask,1], s=points_abs_max[mask,2],
                                        c=points_abs_max[mask,3:], marker=marker)
            self.axes.scatter(points_std[mask,0], points_std[mask,1], s=points_std[mask,2], c=points_std[mask,3:], marker=marker)
        self.axes.set_xticks(xticks, xlabels)
        legend_elements = [Line2D([0], [0], marker='o', color=self.colors[k.split('.')[0]],
                                  label=k.split('.')[0], markersize=15) for k in param_dict.keys() if 'weight' in k]
        # legend_elements = legend_elements + [Line2D([0], [0], marker='*', color=self.colors[k.split('.')[0]],
        #                                             label=k, markersize=15) for k in param_dict.keys() if 'bias' in k]
        # legend1 = self.axes.legend(*scatter.legend_elements(**kw), loc='upper right', title='module types')
        # legend1 = self.axes.legend(loc='upper right', title='module types')
        self.axes.legend(loc='upper right', title='module types', handles=legend_elements)
        kw = dict(prop='sizes', num=5, color='gray', func=lambda s: s / 1000)
        legend2 = self.axes.legend(*scatter.legend_elements(**kw), loc='lower right', title='magnitudes')
        self.draw()

    def append_params(self, param_dict, name, key, param):
        mean = param.mean().item()
        std = param.std().item()
        abs_max = param.abs().max().item()
        if 'weight' in name or 'kernel' in name:
            param_dict[f'{key}.weight'].append([mean, std, abs_max])
        elif 'bias' in name:
            param_dict[f'{key}.bias'].append([mean, std, abs_max])


class ModelViewer(QtWidgets.QWidget):
    def __init__(self, plots, cols=4, parent=None):
        super(ModelViewer, self).__init__(parent)
        self.cols = cols
        layout = QtWidgets.QGridLayout(self)
        self.plots = []
        for p in plots:
            plot = globals()[p['title']](**p)
            layout.addWidget(plot)
            self.plots.append(plot)

    def refresh(self, model, **kwargs):
        for plot in self.plots:
            plot.refresh(model, **kwargs)



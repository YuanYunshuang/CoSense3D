import matplotlib.pyplot as plt

Memory = {
    'EviBEV all': [7508, 84.1, 19],
    'EviBEV ego': [6092, 79.5, 18],
    'AttnFusion all': [16741, 87.6, 37],
    'AttnFusion ego': [7315, 87.1, 22],
    'FPVRCNN all': [11824, 84.0, 30],
    'FPVRCNN ego': [6411, 84.9, 20],
    'F-Cooper all': [6858, 82.2, 15],
    'F-Cooper 2cav': [3333, 68.7, 15],
}

models = ['EviBEV', 'AttnFusion', 'FPVRCNN', 'F-Cooper']
symbols = ['*', '^', 'o', 's']

plt.rcParams.update({'font.size': 10})
fig = plt.figure(figsize=(8, 4))
axs = fig.subplots(1, 2)

for model, symbol in zip(models, symbols):
    for k in ['all', 'ego']:
        if model == 'F-Cooper' and k == 'ego':
            label = f'{model} 2cav'
        else:
            label = f'{model} {k}'
        mem, ap, time = Memory[label]

        color = 'orange' if k == 'all' else 'green'
        axs[0].plot(mem / 1024., ap, marker=symbol, color=color, linestyle='',
                    label=label, markersize=10, markerfacecolor='w', markeredgewidth=2)
        axs[1].plot(time, ap, marker=symbol, color=color, linestyle='',
                    label=label, markersize=10, markerfacecolor='w', markeredgewidth=2)
axs[0].set_xlabel('Memory Usage (G)')
axs[1].set_xlabel('Training Time (hours)')
axs[0].set_ylabel('AP@0.7')
axs[1].set_ylabel('AP@0.7')
axs[0].legend(loc='lower right', fontsize=10)
axs[1].legend(loc='lower right', fontsize=10)
fig.tight_layout()
plt.savefig("memory_time_usage.pdf", format="pdf", bbox_inches="tight")
plt.show()
plt.close()
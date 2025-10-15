import sys
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from src.ClevelandMcGill.weber import Weber

sys.path.append('../')
all_data = []
images = []

titles = ['10', '100', '1000']

classifiers = ['CvT', 'Swin', 'vViT']

human_values = [(4.00, 0.52),(5.39, 0.25), (5.46, 0.35)]
human_values.append((np.mean([v[0] for v in human_values]), np.mean([v[1] for v in human_values])))

all_data = [[[9, 9, 9.37], [5.21, 7.48, 7.17], [8.14, 8.27, 8.25], [9.01, 9, 9.03]],
            [[9, 9, 9 ], [8.43, 4.8, 4.87], [7.32, 7.45, 7.30], [8.16, 8.18, 8.15]],
            [[6.19, 7.48, 7.72], [5.48, 5.79, 5.93], [4.79, 4.79, 4.79],[4.8, 4.8, 4.8]]]

## images
images = []
for a, c_base in enumerate(['10', '100', '1000']):
    image, label = eval('Weber.base' + c_base)()
    image = image.astype(np.float32)
    image += np.random.uniform(0, 0.05, (100, 100))

    images.append(image)


fig = plt.figure(figsize=(7, 5), facecolor='white')
gs = gridspec.GridSpec(len(titles), 2, width_ratios=[.1, .3], hspace=.3)

j = 0  # grid index (running)
classifiers3 = ['Image', 'Human'] + classifiers + ['Dummy']
for z, experiment in enumerate(titles):

    for i, c in enumerate(classifiers3):
        if i == 0:
            fig = plt.subplot(gs[j])
            j += 1

            plt.title(titles[z].upper(), loc='left', fontsize=10)

            ax = plt.gca()
            from matplotlib.ticker import NullFormatter

            ax.xaxis.set_major_formatter(NullFormatter())
            ax.xaxis.set_ticks_position('none')
            ax.yaxis.set_major_formatter(NullFormatter())
            ax.yaxis.set_ticks_position('none')
            # plt.tight_layout()

            ax.set_xticklabels('')
            ax.set_yticklabels('')

            plt.imshow(images[z], cmap='Greys', interpolation='none')

            continue

        if i == 1:
            fig = plt.subplot(gs[j])
            j += 1
            means = human_values[z][0]
            confidence = human_values[z][1]

            errorbars = plt.errorbar(means, 6 - i, xerr=confidence, fmt='o', color='black', label='Human')
            continue
        if c == 'Dummy':
            continue

        data = all_data[z][i - 2]

        means = [np.mean(data)]
        confidence = [1.96 * np.std(data)]

        y_pos = range(len(means))

        factors = [3, 2, 1, -1, -2, -3]
        c_factor = factors[i - 1]

        y_pos = [v + c_factor for v in y_pos]

        plt.xlim(3, 10.1)

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        ax.get_yaxis().set_ticks([])

        ax.get_xaxis().set_ticks(np.concatenate((np.arange(0, 11, 10), np.arange(5, 5.1))))
        ax.tick_params(axis=u'both', which=u'both', length=0)

        if z !=2:
            from matplotlib.ticker import NullFormatter

            ax.xaxis.set_major_formatter(NullFormatter())
            ax.xaxis.set_ticks_position('none')

        plt.grid(True, color='gray', which='major', axis='x', alpha=1)
        plt.grid(True, color='gray', which='minor', axis='x', alpha=0.2)
        c_color = 'C' + str(i - 2)
        errorbars = plt.errorbar(means, y_pos, xerr=confidence, fmt='o', color=c_color, label=c)

    handles, labels = ax.get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc='lower center',
    bbox_to_anchor=(0.3, -0.6),
    ncol=len(classifiers3) - 2,
    fontsize=8
)
plt.savefig('weber_analysis.pdf', bbox_inches='tight', pad_inches=0.2)

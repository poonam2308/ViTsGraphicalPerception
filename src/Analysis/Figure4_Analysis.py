import sys
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

from src.ClevelandMcGill.figure4 import Figure4

sys.path.append('../')

classifiers = ['CvT', 'Swin', 'vViT']
human_values = [(1.4, 0.14), (1.72, 0.2), (1.84, 0.16), (2.35, 0.175), (2.72, 0.155)]
human_values.append((np.mean([v[0] for v in human_values]), np.mean([v[1] for v in human_values])))

all_data = [[[4.74, 4.79, 4.83], [4.72, 4.71, 4.69], [4.81, 4.74, 4.75], [4.72, 4.70, 4.81]],
            [[4.77, 5.10, 4.99], [4.74, 4.73, 4.81], [5.36, 5.32, 5.25], [4.86, 4.96, 4.75]],
            [[4.81, 5.9, 4.77], [4.72, 4.72, 4.73], [4.76, 4.8, 4.89], [4.71, 4.73, 4.73]],
            [[5.58, 5.58, 5.40], [4.71, 4.71, 4.73], [5.14, 5.04, 5.11], [5.18, 5.00, 5.12]],
            # new
            [[5.37, 5.37, 5.38], [4.74, 4.74, 4.75], [5.23, 5.06, 5.11], [5.19, 5.01, 5.10]]]

## images
images = []
for type_ in range(1, 6):
    data, labels = Figure4.generate_datapoint()

    image = eval('Figure4.data_to_type' + str(type_))(data)
    image = image.astype(np.float32)
    image += np.random.uniform(0, 0.05, (100, 100))
    images.append(image)
fig = plt.figure(figsize=(5, 9), facecolor='white')
gs = gridspec.GridSpec(5, 2, width_ratios=[.1, .3], hspace=.3)  # , wspace=.5)
classifiers3 = ['Image', 'Human'] + classifiers + ['Dummy']
j = 0  # grid index (running)
rows = 5
for row in range(rows):

    for i, c in enumerate(classifiers3):

        if i == 0:
            fig = plt.subplot(gs[j])
            j += 1

            c_label = 'TYPE ' + str(row + 1)
            if row == 5:
                c_label = 'MULTI'
            plt.title(c_label, loc='left', fontsize=8)

            ax = plt.gca()
            from matplotlib.ticker import NullFormatter, ScalarFormatter

            ax.xaxis.set_major_formatter(NullFormatter())
            ax.xaxis.set_ticks_position('none')
            ax.yaxis.set_major_formatter(NullFormatter())
            ax.yaxis.set_ticks_position('none')
            plt.tight_layout()

            ax.set_xticklabels('')
            ax.set_yticklabels('')

            plt.imshow(images[row], cmap='Greys', interpolation='none')

            continue

        if i == 1:
            fig = plt.subplot(gs[j])
            j += 1
            # this is human
            means = human_values[row][0]
            confidence = human_values[row][1]

            errorbars = plt.errorbar(means, 6 - i, xerr=confidence, fmt='o', color='black', label='Human')
            continue

        if c == 'Dummy':
            continue

        data = all_data[row][i - 2]

        means = [np.mean(data)]
        confidence = [1.96 * np.std(data)]

        y_pos = range(len(means))
        factors = [3, 2, 1, -1, -2, -3]
        c_factor = factors[i - 1]

        y_pos = [v + c_factor for v in y_pos]

        plt.xlim(0, 6.1)

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        #         if row != 0:
        ax.get_yaxis().set_ticks([])

        ax.get_xaxis().set_ticks(np.concatenate((np.arange(0, 7, 6), np.arange(3, 3.1))))
        ax.tick_params(axis=u'both', which=u'both', length=0)

        # remove tick marks
        if row != 4:
            from matplotlib.ticker import NullFormatter

            ax.xaxis.set_major_formatter(NullFormatter())
            ax.xaxis.set_ticks_position('none')

        # grid lines for X
        plt.grid(True, color='gray', which='major', axis='x', alpha=1)
        c_color = 'C' + str(i - 2)
        errorbars = plt.errorbar(means, y_pos, xerr=confidence, fmt='o', color=c_color, label=c)

    handles, labels = ax.get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc='lower center',
    bbox_to_anchor=(0.3, -0.5),
    ncol=len(classifiers3) - 2,
    fontsize=8
)

plt.savefig('pl_analysis.pdf', bbox_inches='tight', pad_inches=0.2)

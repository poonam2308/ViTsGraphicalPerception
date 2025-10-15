import sys
import matplotlib.pyplot as plt
from src.ClevelandMcGill.figure3 import Figure3
import numpy as np
from matplotlib import gridspec

sys.path.append('../')
all_data = []
images = []
experiments = ['Figure3.data_to_barchart',
               'Figure3.data_to_piechart',
               'Figure3.data_to_piechart_aa']

titles = ['Bar Chart', 'Pie Chart', 'Pie Chart 2']

## data
bar_data = [None] * 3
pie_data = [None] * 3

classifiers = ['CvT', 'Swin', 'vViT']

human_values = [(2.05, 0.115),(2.05, 0.115)]
human_values.append((np.mean([v[0] for v in human_values]), np.mean([v[1] for v in human_values])))

all_data = [[[4.79, 4.86, 4.51], [4.22, 4.21, 4.21],[5.48, 5.42, 5.46], [4.66, 4.75, 4.87]], #bar
            [[4.46, 4.49, 4.57], [4.21, 4.22, 4.22],[5.49, 5.51, 5.46], [5.10, 4.39, 4.52]], #pie
            [[4.67, 4.53, 4.83], [4.21, 4.21, 4.22],[5.39, 5.46, 5.48], [4.87, 5.08, 4.68]]] #pie_aa

## images
data, labels = Figure3.generate_datapoint()
bar_image = Figure3.data_to_barchart(data)
pie_image = Figure3.data_to_piechart(data)
pie_image_aa = Figure3.data_to_piechart_aa(data)

bar_image = bar_image.astype(np.float32)
bar_image += np.random.uniform(0, 0.05, (100, 100))

pie_image = pie_image.astype(np.float32)
pie_image += np.random.uniform(0, 0.05, (100, 100))

pie_image = pie_image.astype(np.float32)
pie_image += np.random.uniform(0, 0.05, (100, 100))
images = [bar_image, pie_image, pie_image_aa]

fig = plt.figure(figsize=(7, 5), facecolor='white')
gs = gridspec.GridSpec(len(experiments), 2, width_ratios=[.1, .3], hspace=.3)

j = 0
classifiers3 = ['Image', 'Human'] + classifiers + ['Dummy']
for z, experiment in enumerate(experiments):

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
            ax.set_xticks(np.arange(-.5, 100, 10), minor=False);
            ax.set_yticks(np.arange(-.5, 100, 10), minor=False);
            ax.set_axisbelow(True)

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

        plt.xlim(0, 6.1)

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        ax.get_yaxis().set_ticks([])

        ax.get_xaxis().set_ticks(np.concatenate((np.arange(0, 7, 6), np.arange(3, 3.1))))
        ax.tick_params(axis=u'both', which=u'both', length=0)
        if z != 2:
            from matplotlib.ticker import NullFormatter

            ax.xaxis.set_major_formatter(NullFormatter())
            ax.xaxis.set_ticks_position('none')


        plt.grid(True, color='gray', which='major', axis='x', alpha=1)
        plt.grid(True, color='gray', which='minor', axis='x', alpha=0.2)
        c_color = 'C' + str(i - 2)
        errorbars = plt.errorbar(means, y_pos, xerr=confidence,  fmt='o', color=c_color, label=c)

    handles, labels = ax.get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc='lower center',
    bbox_to_anchor=(0.3, -0.6),
    ncol=len(classifiers3) - 2,
    fontsize=11
)
plt.savefig('pa_analysis.pdf', bbox_inches='tight', pad_inches=0.2)
import sys
import matplotlib.pyplot as plt
from src.ClevelandMcGill.figure12 import Figure12
import numpy as np
from matplotlib import gridspec

sys.path.append('../')
all_data = []
images = []
experiments = ['Figure12.data_to_bars',
               'Figure12.data_to_framed_rectangles']

titles = ['Framed Rectangles', 'Bars']

bar_data = [None] * 3
pie_data = [None] * 3
human_values = [(3.33, 0.83), (3.93, 0.52),]
human_values.append((np.mean([v[0] for v in human_values]), np.mean([v[1] for v in human_values])))


# this analysis is only for three vision transformers: cvt, swin and vit (csv)

classifiers = ['CvT', 'Swin', 'vViT']
all_data = [[[4.77, 4.73, 4.79], [4.74, 4.77, 4.75], [5.49, 5.4, 5.49], [5.49, 4.94, 4.78]], # rect  [[cvt],[swin],[vit]]
            [[4.79, 4.71, 4.71], [4.76, 4.74, 4.76], [5.34, 5.23, 5.21], [5.22, 5.50, 5.25]]] # bar
## images
data, labels, parameters = Figure12.generate_datapoint()
bar_image = Figure12.data_to_bars(data)
rect_image = Figure12.data_to_framed_rectangles(data)

bar_image = bar_image.astype(np.float32)
bar_image += np.random.uniform(0, 0.05, (100, 100))

rect_image = rect_image.astype(np.float32)
rect_image += np.random.uniform(0, 0.05, (100, 100))
images = [rect_image, bar_image]

fig = plt.figure(figsize=(7, 5), facecolor='white')
gs = gridspec.GridSpec(len(experiments), 2, width_ratios=[.1, .3], hspace=.3)

j = 0  # grid index (running)
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
            plt.tight_layout()

            ax.set_xticklabels('')
            ax.set_yticklabels('')

            plt.imshow(images[z], cmap='Greys', interpolation='none')

            continue

        if i == 1:
            fig = plt.subplot(gs[j])
            j += 1
            means = human_values[z][0]
            confidence = human_values[z][1]

            errorbars = plt.errorbar(means, 6-i, xerr=confidence, fmt='o',  color='black', label='Human')
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

        # remove tick marks
        if z !=1:
            from matplotlib.ticker import NullFormatter

            ax.xaxis.set_major_formatter(NullFormatter())
            ax.xaxis.set_ticks_position('none')

        # grid lines for X
        plt.grid(True, color='gray', which='major', axis='x', alpha=1)
        plt.grid(True, color='gray', which='minor', axis='x', alpha=0.2)
        c_color = 'C' + str(i - 2)
        errorbars = plt.errorbar(means, y_pos, xerr=confidence, fmt='o', color=c_color, label=c)

    handles, labels = ax.get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc='lower center',
    bbox_to_anchor=(0.3, -0.4),
    ncol=len(classifiers3) - 2,
    fontsize=11
)
plt.savefig('bf_analysis.pdf', bbox_inches='tight', pad_inches=0.0)

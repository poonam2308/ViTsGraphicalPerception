import numpy as np
import scipy as sp
import scipy.stats
from matplotlib import gridspec
import matplotlib.pyplot as plt
from src.ClevelandMcGill.figure1 import Figure1

experiments = ['Figure1.position_common_scale',
               'Figure1.position_non_aligned_scale',
               'Figure1.length',
               'Figure1.direction',
               'Figure1.angle',
               'Figure1.area',
               'Figure1.volume',
               'Figure1.curvature',
               'Figure1.shading']


# this analysis is only for three vision transformers: cvt, swin and vit (csv)
classifiers = ['CvT', 'Swin', 'vViT',]
human_values=[(3.3, 1.08), (3.14, 1.48), (3.49, 1.08), (3.75, 0.9), (3.28, 1.0), (3.63, 0.79), (5.18, 0.9), (4.13, 0.3), (4.16, 0.68)]
human_values.append((np.mean([v[0] for v in human_values]), np.mean([v[1] for v in human_values])))

# the results are written this way : [[[all cvt], [all swin], [all vit],]
all_results = {'Figure1.position_common_scale': [[[4.66, 4.76, 5.2, 5.06]],
                                                 [[3.05, 1.85, 1.93, 1.44]],
                                                 [[2.91, 3.25, 3.98, 3.91]]],
               'Figure1.position_non_aligned_scale': [[[4.73, 5.4, 4.69, 5.46]],
                                                      [[4.2, -0.56, 1.34, 1.56]],
                                                      [[2.62, 3.03, 3.28, 4.34]]],
               'Figure1.length': [[[3.81,3.28, 3.77, 4]],
                                  [[2.3, 0.57, 2.2, -0.17]],
                                  [[0.98, 1.26, 0.83, 0.83]]],
               'Figure1.direction': [[[4.35, 3.62, 4.53, 4.14]],
                                     [[0.84, 0.57, 0.47, 0.93]],
                                     [[2.23, 2.07, 2.02, 2.16]]],
               'Figure1.angle': [[[4.29, 4.15, 4.54, 4.03 ]],
                                 [[1.74, 0.83, 0.62, 1.14]],
                                 [[3.31, 3.48, 3.57, 3.78]]],
               'Figure1.area': [[[5.15, 4.65, 4.02, 4.38]],
                                [[1.56, 2.94, 1.07, 0.46]],
                                [[3.75, 4.06, 4.26, 3.50]]],
               'Figure1.volume': [[[4.24, 4.6, 4.63, 4.04]],
                                  [[1.72, 2.54, 3.21, 3.19]],
                                  [[3.06, 2.81,3.85, 3.24]]],
               'Figure1.curvature': [[[4.28, 4.19, 3.58, 3.25]],
                                     [[1.31, 0.82, 1.16, 0.44]],
                                     [[1.47, 2.51, 1.45, 2.17]]],
               'Figure1.shading': [[[4.47, 5.19, 4.85, 4.93]],
                                   [[0.5, 0.95, -0.51, 0.16]],
                                   [[2.81, 2.46, 2.9, 2.79]]]
               }
all_labels = {'Figure1.position_common_scale': ['Position Y', '+ Position X', '+ Spotsize'],
              'Figure1.position_non_aligned_scale': ['Position Y', '+ Position X', '+ Spotsize'],
              'Figure1.length': ['Length', '+ Position Y', '+ Position X', '+ Width'],
              'Figure1.direction': ['Direction', '+ Position Y', '+ Position X'],
              'Figure1.angle': ['Angle', '+ Position Y', '+ Position X'],
              'Figure1.area': ['Area', '+ Position Y', '+ Position X'],
              'Figure1.volume': ['Volume', '+ Position Y', '+ Position X'],
              'Figure1.curvature': ['Curvature', '+ Position Y', '+ Position X'],
              'Figure1.shading': ['Shading', '+ Position Y', '+ Position X']
              }
presets = {
    'Figure1.position_common_scale': 40,
    'Figure1.position_non_aligned_scale': 10,
    'Figure1.length': 35,
    'Figure1.direction': 215,
    'Figure1.angle': 60,
    'Figure1.area': 20,
    'Figure1.volume': 18,
    'Figure1.curvature': 50,
    'Figure1.shading': 80
}
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1 + confidence) / 2., n - 1)
    return m, (m - h), m + h, h


fig = plt.figure(figsize=(6, 15), facecolor='white')
gs = gridspec.GridSpec(len(experiments), 2, width_ratios=[.3, 1], hspace=.3)

classifiers3 = ['Image', 'Human'] + classifiers + ['Dummy']
all_results_fresh = dict(all_results)
j = 0
for z, experiment in enumerate(experiments):

    for i, c in enumerate(classifiers3):

        if i == 0:
            fig = plt.subplot(gs[j])
            j += 1

            #plt.title(experiment.split('.')[-1].replace('_', ' ').upper(), loc='left')
            plt.title(experiment.split('.')[-1].replace('_', ' ').upper(), loc='left', fontsize=8)

            ax = plt.gca()
            from matplotlib.ticker import NullFormatter

            ax.xaxis.set_major_formatter(NullFormatter())
            ax.xaxis.set_ticks_position('none')
            ax.yaxis.set_major_formatter(NullFormatter())
            ax.yaxis.set_ticks_position('none')
            # plt.tight_layout()

            image = eval(experiment)(preset=presets[experiment])[1]
            image = image.astype(np.float32)
            image += np.random.uniform(0, 0.05, (100, 100))

            ax.set_xticklabels('')
            ax.set_yticklabels('')
            ax.set_xticks(np.arange(-.5, 100, 10), minor=False);
            ax.set_yticks(np.arange(-.5, 100, 10), minor=False);
            #     ax.grid(which='major', color='gray', linestyle=':', linewidth='0.5')
            ax.set_axisbelow(True)

            plt.imshow(image, cmap='Greys', interpolation='none')

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

        data = all_results_fresh[experiment][i - 2]
        means = [np.mean(r) for r in data]
        confidence = [1.96 * np.std(r) for r in data]
        y_pos = range(len(means))

        factors = [5, 3, 1, -1, -3, -5]
        c_factor = factors[i - 1]

        y_pos = [v + c_factor for v in y_pos]

        plt.xlim(-1, 6.1)

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.get_yaxis().set_ticks([])

        ax.get_xaxis().set_ticks(np.concatenate((np.arange(-1, 8, 8), np.arange(3, 3.1))))
        ax.tick_params(axis=u'both', which=u'both', length=0)

        # remove tick marks
        if z != 8:  # j <= 2*(len(experiments)-1):
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
    bbox_to_anchor=(0.3, -0.6),
    ncol=len(classifiers3) - 2,
    fontsize=8
)
plt.savefig('figure1.pdf', bbox_inches='tight', pad_inches=0.2)
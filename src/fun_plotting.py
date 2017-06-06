import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


mean_ranking_viper_15 = np.array([ 0.545,  0.795,  0.905,  0.955,  0.975,  0.98 ,  0.99 ,  0.99 ,
        0.99 ,  0.99 ,  1.   ,  1.   ,  1.   ,  1.   ,  1.   ,  1.   ,
        1.   ,  1.   ,  1.   ,  1.   ])


mean_ranking_viper_40 = np.array([ 0.705,  0.915,  0.955,  0.97 ,  0.985,  0.995,  0.995,  0.995,
        1.   ,  1.   ,  1.   ,  1.   ,  1.   ,  1.   ,  1.   ,  1.   ,
        1.   ,  1.   ,  1.   ,  1.   ])

mean_ranking_cuhk_15 = np.array([ 0.42 ,  0.65 ,  0.785,  0.9  ,  0.915,  0.965,  0.975,  0.995,
        0.995,  1.   ,  1.   ,  1.   ,  1.   ,  1.   ,  1.   ,  1.   ,
        1.   ,  1.   ,  1.   ,  1.   ])

mean_ranking_cuhk_40 = np.array([ 0.605,  0.775,  0.88 ,  0.94 ,  0.97 ,  0.99 ,  0.995,  0.995,
        1.   ,  1.   ,  1.   ,  1.   ,  1.   ,  1.   ,  1.   ,  1.   ,
        1.   ,  1.   ,  1.   ,  1.   ])


def plot_CMC():
    plt.figure(1)
    plt.subplot(211)
    plt.plot(range(1, 21), mean_ranking_viper_15 * 100, label='15 epochs')
    plt.plot(range(1, 21), mean_ranking_viper_40 * 100, label='40 epochs')
    plt.xticks(range(1, 21))

    plt.xlabel('Rank')
    plt.ylabel('Matching Rate')
    plt.legend()
    plt.title('VIPeR dataset')

    plt.subplot(212)
    plt.plot(range(1,21), mean_ranking_cuhk_15*100, label='15 epochs')
    plt.plot(range(1,21), mean_ranking_cuhk_40*100, label='40 epochs')

    plt.xticks(range(1, 21))

    plt.xlabel('Rank')
    plt.ylabel('Matching Rate')
    plt.legend()
    plt.title('CUHK01 dataset')
    plt.tight_layout()
    # sns.palplot(sns.color_palette("hls", 2))

    # plt.savefig('CMC_curve_viper_cihk01.png', format='png', dpi=400)
    plt.show()

# plot_CMC()

def plot_reli():
    y = [1, 0.95, 0.55, 0.45, 0.9, 1]
    x = [0.08, 0.45, 0.37, 0.01, 0.03, 0.05]
    labels = ['viper', 'cuhk02', 'market', 'caviar', 'grid', 'prid']
    plt.plot(x, y, 'o')
    plt.xlabel('percentage')
    plt.ylabel('rank-1')

    for label, x, y in zip(labels, x, y):
        plt.annotate(
            label,
            xy=(x, y), xytext=(40, 20),
            textcoords='offset points', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.show()


plot_reli()
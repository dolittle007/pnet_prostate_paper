import pandas as pd
from matplotlib import pyplot as plt, ticker
from os.path import join
import numpy as np
from sklearn import metrics
from sklearn.metrics import average_precision_score
import seaborn as sns
import matplotlib as mpl
import itertools
from sklearn.metrics import confusion_matrix
from mpl_toolkits.axes_grid1 import make_axes_locatable
from os.path import join, dirname
import os
current_dir = dirname(os.path.realpath(__file__))

# set default params
from config_path import PROSTATE_LOG_PATH

custom_rcParams = {
    'figure.figsize': (8, 3),
    'font.family': 'Arial',
    'font.size': 12,
    'font.weight': 'regular',
    'axes.labelsize': 12,
    'axes.formatter.useoffset': False,
    'axes.formatter.limits': (-4, 4),
    'axes.titlesize': 12,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'pdf.fonttype': 42
}

mpl.rcParams.update(custom_rcParams)
sns.set_context('paper', rc=custom_rcParams)
sns.set_style("white", {"grid.linestyle": '--', "axes.grid": False, "grid.color":"0.9"})


def plot_confusion_matrix(ax, cm, classes, labels=None,
                          normalize=False,
                          # title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = 100. * cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    # ax.set_title(title)
    # ax.colorbar()
    fig = plt.gcf()

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='10%', pad=0.1)

    # cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
    cb = fig.colorbar(im, cax=cax, orientation='vertical')
    cb.outline.set_visible(False)
    tick_marks = np.arange(len(classes))
    # ax.set_xticks(tick_marks, classes, rotation=45)
    # ax.set_xticks(tick_marks, classes)
    # ax.set_yticks(tick_marks, classes)


    if labels is None:
        fmt = '{:.2f}%' if normalize else '{:d}'
    else:
        fmt = '{}: {:.2f}%' if normalize else '{}: {:d}'

    thresh = cm.max() / 2.
    for i, j in itertools.product(list(range(cm.shape[0])), list(range(cm.shape[1]))):
        #         text= format(labels[i,j], cm[i, j], fmt)
        text = fmt.format(labels[i, j], cm[i, j])
        ax.text(j, i, text,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=12)

    # ax.tight_layout()
    fontproperties = {'family': 'Arial', 'weight': 'bold', 'size': 14}

    ax.set_ylabel('True label',  fontproperties )
    # ax.set_ylabel('True label', fontsize=12, fontweight = 'bold', fontproperties )
    ax.set_xlabel('Predicted label', fontproperties)
    # plt.gcf().subplots_adjust(bottom=0.25, left=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    print(tick_marks)
    print(classes)

    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    #
    # ax.set_yticks([t-0.25 for t in tick_marks])
    ax.set_yticks([t for t in tick_marks])
    ax.set_yticklabels(classes, rotation=0)

    plt.gcf().subplots_adjust(bottom=0.25)
    plt.gcf().subplots_adjust(left=0.25)

    # ax.set_xticks(tick_marks, classes)
    # ax.set_yticks(tick_marks, classes)

    # ax.set_xticks([])
    # ax.set_yticks([])
    # for minor ticks
    # ax.set_xticks([], minor=True)
    # ax.set_yticks([], minor=True)

#                     cmap = plt.cm.Greys)

def plot_confusion_matrix_all(ax):
    base_dir = join(PROSTATE_LOG_PATH, 'pnet')
    # models_base_dir = join(base_dir, 'onsplit_average_reg_10_tanh_large_testing_Apr-11_11-22')
    models_base_dir = join(base_dir, 'onsplit_average_reg_10_tanh_large_testing')
    filename = join(models_base_dir, 'P-net_ALL_testing.csv')
    df = pd.read_csv(filename, index_col=0)
    df.pred = df.pred_scores > 0.5
    df.head()

    y_t = df.y
    y_pred_test = df.pred
    cnf_matrix = confusion_matrix(y_t, y_pred_test)
    print(cnf_matrix)

    cm = np.array(cnf_matrix)
    classes = ['Primary', 'Metastatic']
    labels = np.array([['TN', 'FP'], ['FN ', 'TP']])
    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))

    plot_confusion_matrix(ax, cm, classes,
                          labels,
                          normalize=True,
                          # title='Confusion matrix',
                          # cmap=plt.cm.Blues)
                          cmap=plt.cm.Reds)
if __name__=='__main__':
    fig, ax = plt.subplots(nrows=1, ncols=1, sharey=True, figsize=(5,4), dpi=400)
    plot_confusion_matrix_all(ax)
    plt.savefig(join(current_dir,'./output/confusion matrix.png'), dpi=400)

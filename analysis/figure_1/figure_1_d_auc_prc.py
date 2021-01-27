import collections

import pandas as pd
from matplotlib import pyplot as plt, ticker
from os.path import join
import numpy as np
from sklearn import metrics
from sklearn.metrics import average_precision_score
import seaborn as sns
import matplotlib as mpl
from os.path import join, dirname
import os
current_dir = dirname(os.path.realpath(__file__))

# set default params
from config_path import PROSTATE_LOG_PATH

custom_rcParams = {
    'figure.figsize': (8, 3),
    'font.family': 'Arial',
    'font.size': 10,
    'font.weight': 'regular',
    'axes.labelsize': 10,
    'axes.formatter.useoffset': False,
    'axes.formatter.limits': (-4, 4),
    'axes.titlesize': 10,
    'legend.fontsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'pdf.fonttype': 42
}

mpl.rcParams.update(custom_rcParams)
sns.set_context('paper', rc=custom_rcParams)
sns.set_style("white", {"grid.linestyle": '--', "axes.grid": True, "grid.color":"0.9"})

mapping_dict = {'accuracy':'Accuracy',       'auc':'Area Under Curve (AUC)'  ,
                'aupr':'AUPRC', 'f1': 'F1', 'percision' :'Precision'  , 'recall':'Recall' }

def plot_roc(ax, y_test, y_pred_score, save_dir,color, label=''):
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_score, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    # plt.figure(fig.number)
#     plt.plot(fpr, tpr, label=label + ' (area = %0.2f)' % roc_auc, linewidth=2, color=color)
#     plt.plot(fpr, tpr, label=label + ' (area = %0.2f)' % roc_auc, linewidth=2)
    ax.plot(fpr, tpr, label=label + ' (area = %0.2f)' % roc_auc, linewidth=2, color=color)
    # plt.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    ax.set_xlabel('False Positive Rate', fontproperties)
    ax.set_ylabel('True Positive Rate', fontproperties)
    # ax.set_title('Receiver operating characteristic (ROC)', fontsize=18)


def plot_prc(ax, y_test, y_pred_score, save_dir, color, label=''):
    # plt.figure(fig.number)
    precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_pred_score)
    roc_auc = average_precision_score(y_test, y_pred_score)
    #     plt.plot(recall, precision, label=label + '(area= %0.2f)' % roc_auc, linewidth=2, color=color)
    ax.plot(recall, precision, label=label + '(area= %0.2f)' % roc_auc, linewidth=2, color=color)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall',fontproperties )
    ax.set_ylabel('Precision', fontproperties)
    # plt.show()



all_models_dict = {}
# base_dir = './../../run/logs/p1000/'
# base_dir = join(PROSTATE_LOG_PATH, 'pnet')
base_dir = PROSTATE_LOG_PATH
# models_base_dir = join(base_dir , 'compare/onsplit_ML_test_Apr-11_11-34')
models_base_dir = join(base_dir , 'compare/onsplit_ML_test')
models = ['Linear Support Vector Machine ', 'RBF Support Vector Machine ', 'L2 Logistic Regression', 'Random Forest',
          'Adaptive Boosting', 'Decision Tree']

for i, m in enumerate(models):
    df = pd.read_csv(join(models_base_dir, m + '_data_0_testing.csv'), sep=',', index_col=0, header=[0, 1])
    all_models_dict[m] = df

# pnet_base_dir = join(base_dir , 'pnet/onsplit_average_reg_10_tanh_large_testing_Apr-11_11-22')
pnet_base_dir = join(base_dir , 'pnet/onsplit_average_reg_10_tanh_large_testing')
df_pnet = pd.read_csv(join(pnet_base_dir, 'P-net_ALL_testing.csv'), sep=',', index_col=0, header=[0, 1])
all_models_dict['P-net'] = df_pnet
n = len(models)+1

def plot_prc_all(ax):
    # colors= sns.hls_palette(n, l=.4, s=.8)
    colors= sns.color_palette(None,n)
    import collections

    #sort based on area under prc
    sorted_dict={}
    for i, k in enumerate(all_models_dict.keys()):

        df = all_models_dict[k]
        y_test = df['y']
        y_pred_score = df['pred_scores']
        average_prc = average_precision_score(y_test, y_pred_score)
        sorted_dict[k] = average_prc

    sorted_dict = sorted(list(sorted_dict.items()), key=lambda kv: kv[1])
    sorted_dict = collections.OrderedDict(sorted_dict)
    print(('sorted_dict', sorted_dict))

    for i, k in enumerate(sorted_dict.keys()):
        df = all_models_dict[k]
        y_test = df['y']
        y_pred_score = df['pred_scores']
        print((i,k))
        plot_prc(ax, y_test, y_pred_score, None, label=k, color=colors[i])

    f_scores = np.linspace(0.2, 0.8, num=4)
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = ax.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2, linewidth=2)
        # ax.annotate('F1={0:0.1f}'.format(f_score), fontsize=8, xy=(0.9, y[45] + 0.02))
        ax.annotate('F1={0:0.1f}'.format(f_score), fontsize=8, xy=( y[45] - 0.03, 1.02))
    # plt.set_cmap('copper')
    ax.legend(loc="lower left", fontsize=8, framealpha=0.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)

    # ax.set_title('Precision-Recall Curve (PRC)', fontsize=14)


def plot_auc_all(ax):
    # sort based on area under prc
    colors = sns.color_palette(None, n)
    sorted_dict = {}
    for i, k in enumerate(all_models_dict.keys()):
        df = all_models_dict[k]
        y_test = df['y']
        y_pred_score = df['pred_scores']
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_score, pos_label=1)
        average_auc = metrics.auc(fpr, tpr)
        average_auc = average_precision_score(y_test, y_pred_score)
        sorted_dict[k] = average_auc

    sorted_dict = sorted(list(sorted_dict.items()), key=lambda kv: kv[1])
    sorted_dict = collections.OrderedDict(sorted_dict)

    # colors = sns.hls_palette(n, l=.4, s=.7)
    # for i, k in enumerate(all_models_dict.keys()):
    for i, k in enumerate(sorted_dict.keys()):
        df = all_models_dict[k]
        y_test = df['y']
        y_pred_score = df['pred_scores']
        plot_roc(ax, y_test, y_pred_score, None, color=colors[i], label=k)


fontproperties = {'family': 'Arial', 'weight': 'bold', 'size': 14}

if __name__=="__main__":

    fig, ax = plt.subplots(nrows=1, ncols=1, sharey=True, figsize=(5,4), dpi=400)
    plot_prc_all(ax)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig( join(current_dir,'./output/_prc'), dpi=400)

    # plt.savefig( '_prc', dpi=600)

if __name__=="__main__":
    fig, ax = plt.subplots(nrows=1, ncols=1, sharey=True, figsize=(5,4), dpi=400)
    plot_auc_all(ax)
    # plt.legend(loc="lower right", prop={'size':10}, framealpha=0.0)
    plt.legend(loc="lower right", fontsize=8, framealpha=0.0)
    plt.savefig(join(current_dir,'./output/_auc'), dpi=400)

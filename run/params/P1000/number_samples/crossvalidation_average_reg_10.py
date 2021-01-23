import os
from copy import deepcopy
from os.path import dirname
import numpy as np
from model.builders.prostate_models import build_pnet2

base_dirname = dirname(dirname(__file__))
print (base_dirname)
filename = os.path.basename(__file__)

task = 'classification_binary'
selected_genes = 'tcga_prostate_expressed_genes_and_cancer_genes.csv'
data_base = {'id': 'ALL', 'type': 'prostate_paper',
             'params': {
                 'data_type': ['mut_important', 'cnv_del', 'cnv_amp'],
                 'drop_AR': False,
                 'cnv_levels': 3,
                 'mut_binary': True,
                 'balanced_data': False,
                 'combine_type': 'union',  # intersection
                 'use_coding_genes_only': True,
                 'selected_genes': selected_genes,
                 'training_split': 10,
             }
             }
data = []
splits = np.arange(0, 20, 3)
for n in splits:
    d = deepcopy(data_base)
    d['id'] = 'data_{}'.format(n)
    d['params']['training_split'] = str(n)
    data.append(d)

n_hidden_layers = 5
# base_reg = 0.01
base_dropout = 0.5
# wregs= [float(base_reg)/float(i) for i in range(1,n_hidden_layers+1)]
wregs = [0.001] * 7
# loss_weights = range(3,9)
loss_weights = [2, 7, 20, 54, 148, 400]
# loss_weights = [2, 7, 20, 54, 100, 100]
# wreg_outcomes = [0.01]*6
wreg_outcomes = [0.01] * 6
pre = {'type': None}

nn_pathway = {
    'type': 'nn',
    'id': 'P-net',
    'params':
        {
            'build_fn': build_pnet2,
            'model_params': {
                'use_bias': True,
                'w_reg': wregs,
                'w_reg_outcomes': wreg_outcomes,
                'dropout': [base_dropout] + [0.1] * (n_hidden_layers + 1),
                'loss_weights': loss_weights,
                'optimizer': 'Adam',
                'activation': 'selu',
                'data_params': data_base,
                'add_unk_genes': False,
                'shuffle_genes': False,
                'kernel_initializer': 'lecun_uniform',
                'n_hidden_layers': n_hidden_layers,
                'attention': False,
                'dropout_testing': False  # keep dropout in testing phase, useful for bayesian inference

            }, 'fitting_params': dict(samples_per_epoch=10,
                                      select_best_model=False,
                                      monitor='val_o6_f1',
                                      verbose=2,
                                      epoch=300,
                                      shuffle=True,
                                      batch_size=50,
                                      save_name='pnet',
                                      debug=False,
                                      save_gradient=False,
                                      class_weight='auto',
                                      n_outputs=n_hidden_layers + 1,
                                      # prediction_output='final',
                                      prediction_output='average',
                                      early_stop=False,
                                      reduce_lr=False,
                                      reduce_lr_after_nepochs=dict(drop=0.25, epochs_drop=50),
                                      lr=0.001
                                      ),
            'feature_importance': 'deepexplain_grad*input'
        },
}

models = []

class_weight = {0: 0.75, 1: 1.5}
logistic = {'type': 'sgd', 'id': 'Logistic Regression',
            'params': {'loss': 'log', 'penalty': 'l2', 'alpha': 0.01, 'class_weight': class_weight}}
models.append(logistic)

# models.append(nn_pathway_average)
models.append(nn_pathway)
features = {}

# pipeline = {'type':  'FS', 'params': { 'save_train' : True}}
# pipeline = {'type':  'one_split', 'params': { 'save_train' : True, 'eval_dataset': 'test'}}
# pipeline = {'type':  'one_split', 'params': { 'save_train' : True, 'eval_dataset': 'validation'}}
pipeline = {'type': 'crossvalidation', 'params': {'n_splits': 5, 'save_train': True}}

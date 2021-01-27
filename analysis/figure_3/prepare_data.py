import pandas as pd
import os
from os.path import join
import sys
from  config_path import *

from analysis.figure_3.data_extraction_utils import get_node_importance, get_link_weights, get_link_weights_df, \
    get_data, get_degrees, adjust_coef_with_graph_degree, get_high_nodes, get_connections, filter_connections
from utils.loading_utils import DataModelLoader
os.environ['KMP_DUPLICATE_LIB_OK']='True'
current_dir = dirname(os.path.realpath(__file__))
saving_dir = join(current_dir, 'extracted')

def save_gradient_importance(node_weights_, node_weights_samples_dfs, info):
    for i, n in enumerate(node_weights_):
        n.to_csv(join(saving_dir,'gradient_importance_{}.csv'.format(i)))

    for i, n in enumerate(node_weights_samples_dfs):
        if i > 0:
            n['ind']= info
            n = n.set_index('ind')
            n.to_csv(join(saving_dir, 'gradient_importance_detailed_{}.csv'.format(i)))


def save_link_weights(link_weights_df):
    for i, link in enumerate(link_weights_df):
        link.to_csv(join(saving_dir,'link_weights_{}.csv'.format(i)))
    # df = pd.DataFrame(link_weights[-1], index= feature_names[-1], columns= ['root'])
    # df.to_csv('./extracted/link_weights_{}.csv'.format(i+1))

def save_activation(layer_outs_dict, feature_names):
    for l_name, l_outut in sorted(layer_outs_dict.items()):
        if l_name.startswith('h'):
            # print l_name, l_outut.shape
            l = int(l_name[1:])
            features = feature_names[l + 1]
            layer_output_df = pd.DataFrame(l_outut, index=info, columns=features)
            layer_output_df = layer_output_df.round(decimals=3)
            layer_output_df.to_csv(join(saving_dir,'activation_{}.csv'.format(l + 1)))

def save_graph_stats(degrees,fan_outs, fan_ins ):
    i = 1
    df = pd.concat([degrees[0], fan_outs[0]], axis=1)
    df.columns = ['degree', 'fan_out']
    df['fan_in'] = 0
    df.to_csv(join(saving_dir,'graph_stats_{}.csv'.format(i)))

    for i, (d, fin, fout) in enumerate(zip(degrees[1:], fan_ins, fan_outs[1:])):
        df = pd.concat([d, fin, fout], axis=1)
        df.columns = ['degree', 'fan_in', 'fan_out']
        # print df.head()
        df.to_csv(join(saving_dir, 'graph_stats_{}.csv'.format(i + 2)))



base_dir = join(PROSTATE_LOG_PATH, 'pnet')
model_name = 'onsplit_average_reg_10_tanh_large_testing'

# importance_type = ['deepexplain_deeplift']
importance_type = ['deepexplain_deeplift']
target = 'o6'
use_data = 'All' #{'Train', 'Test'}
dropAR = False

# load model and data --------------------
model_dir = join(base_dir, model_name)
model_file= 'P-net_ALL'
params_file = join(model_dir, model_file + '_params.yml')
print('loading ', params_file)
loader = DataModelLoader(params_file)
nn_model = loader.get_model(model_file)
feature_names= nn_model.feature_names
X, Y, info = get_data(loader, use_data, dropAR)
response = pd.DataFrame(Y, index=info, columns=['response'])
# print response.head()
response.to_csv(join(saving_dir, 'response.csv'))
#
print ('saving gradeint importance')
# #gradeint importance --------------------
node_weights_, node_weights_samples_dfs = get_node_importance(nn_model, X, Y, importance_type[0], target )
save_gradient_importance(node_weights_, node_weights_samples_dfs, info)
#
print ('saving link weights')
# # link weights --------------------
link_weights = get_link_weights(nn_model.model)
link_weights_df = get_link_weights_df(link_weights, feature_names)
save_link_weights(link_weights_df)
#
print ('saving activation')
# # activation --------------------
layer_outs_dict = nn_model.get_layer_outputs(X)
save_activation(layer_outs_dict, feature_names)
#
print ('saving graph stats')
# # graph stats --------------------
degrees, fan_ins, fan_outs = get_degrees(link_weights_df[1:])
save_graph_stats(degrees,fan_outs, fan_ins)
#
print('adjust weights with graph stats')
# # graph stats --------------------
#
node_importance = adjust_coef_with_graph_degree(node_weights_[1:], degrees, saving_dir)
node_importance.to_csv(join(saving_dir,'node_importance_graph_adjusted.csv'))


# # filter connection --------------------


high_nodes2 = get_high_nodes(node_importance, sigma=3)
connections_df = get_connections(link_weights_df[1:])

important_node_connections_df = filter_connections(connections_df, high_nodes2, add_unk=False)
important_node_connections_df.to_csv(join(saving_dir,'links_filtered.csv'))

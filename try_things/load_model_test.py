
from os.path import join
from config_path import PROSTATE_LOG_PATH
from utils.loading_utils import DataModelLoader
import numpy as np
from utils.evaluate import evalualte
from model.layers_custom import Diagonal
base_dir = join(PROSTATE_LOG_PATH, 'pnet')
model_name = 'onsplit_average_reg_10_tanh_large_testing'

model_dir = join(base_dir, model_name)
model_file= 'P-net_ALL'
params_file = join(model_dir, model_file + '_params.yml')
print('loading ', params_file)
loader = DataModelLoader(params_file)
nn_model = loader.get_model(model_file)
print(nn_model.model.summary())
# feature_names= nn_model.feature_names

x_train, x_test, y_train, y_test, info_train, info_test, columns = loader.get_data()

info = list(info_train) + list(info_test)
pred_scores = nn_model.predict_proba(x_test)[:, 1]
pred = nn_model.predict(x_test)
metrics = evalualte(pred, y_test, pred_scores)
print (metrics)

#training
pred_scores = nn_model.predict_proba(x_train)[:, 1]
pred = nn_model.predict(x_train)
print (pred)
print(sum(pred))
metrics = evalualte(pred, y_train, pred_scores)
print (metrics)




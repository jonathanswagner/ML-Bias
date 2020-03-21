
import pandas as pd
import numpy as np
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
from aif360.datasets import BinaryLabelDataset

def get_disparity_index(di):
    return 1 - np.minimum(di, 1 / di)

def calc_disparity_index(data, target_variable, protected_variable, privileged_input, unprivileged_input):
    df_aif = BinaryLabelDataset(df=data, label_names=[target_variable],
                                protected_attribute_names=[protected_variable])
    privileged_group = [{protected_variable: privileged_input}] #male=1
    unprivileged_group = [{protected_variable: unprivileged_input}] #female=0
    metric_orig = BinaryLabelDatasetMetric(df_aif, unprivileged_group, privileged_group)
    print('1-min(DI, 1/DI):', get_disparity_index(metric_orig.disparate_impact()).round(3))
    if get_disparity_index(metric_orig.disparate_impact()).round(3) < 0.2:
        print('The algorithm can be considered to be not biased')
    else:
        print('There is a potential bias')

def calc_stat_parity(data, target_variable, protected_variable, privileged_input, unprivileged_input):
    df_aif = BinaryLabelDataset(df=data, label_names=[target_variable],
                                protected_attribute_names=[protected_variable])
    privileged_group = [{protected_variable: privileged_input}] #male=1
    unprivileged_group = [{protected_variable: unprivileged_input}] #female=0
    metric_orig = BinaryLabelDatasetMetric(df_aif, unprivileged_group, privileged_group)
    print(metric_orig.statistical_parity_difference().round(3))
    if metric_orig.statistical_parity_difference().round(3) > -0.1:
        print('The algorithm can be considered to be not biased')
    else:
        print('There is a potential bias')

def data_generator(data):
    X_new = pd.DataFrame(columns=[])

    for c in data.columns:
        if is_binary(data[c]):
            X_new[c] = np.random.binomial(1, .5, 1000)
        else:
            X_new[c] = np.random.normal(data[c].describe()[1], data[c].describe()[2], 1000)
    return X_new

def create_eval(pre_trained_model, data):
    pred_y_n = pre_trained_model.predict(data)
    df_n = pd.DataFrame(pred_y_n, columns=['Pred'])
    pred_n = pd.concat([data.reset_index(drop='True'),df_n.reset_index(drop='True')],axis=1)
    return pred_n

def is_binary(series):
    return sorted(series.unique()) == [0,1]
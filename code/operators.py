
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
    if abs(metric_orig.statistical_parity_difference().round(3)) > 0.1:
        print('The algorithm can be considered to be not biased')
    else:
        print('There is a potential bias')

def calc_mean_diff(data, target_variable, protected_variable, privileged_input, unprivileged_input):
    df_aif = BinaryLabelDataset(df=data, label_names=[target_variable],
                                protected_attribute_names=[protected_variable])
    privileged_group = [{protected_variable: privileged_input}] #male=1
    unprivileged_group = [{protected_variable: unprivileged_input}] #female=0
    metric_orig = BinaryLabelDatasetMetric(df_aif, unprivileged_group, privileged_group)
    print(metric_orig.mean_difference().round(3))
    if abs(metric_orig.mean_difference().round(3)) < 0.2:
        print('The algorithm can be considered to be not biased')
    else:
        print('There is a potential bias')

def odds_diff(random_data, predicted_data, target_variable, protected_variable, privileged_input, unprivileged_input):
    random_data['Pred'] = np.random.binomial(1, .5, 1000)
    dataset = BinaryLabelDataset(df=random_data, label_names=[target_variable], protected_attribute_names=[protected_variable])
    classified_dataset = BinaryLabelDataset(df=predicted_data, label_names=[target_variable], protected_attribute_names=[protected_variable])
    privileged_group = [{protected_variable: privileged_input}] #male=1
    unprivileged_group = [{protected_variable: unprivileged_input}] #female=0
    metric = ClassificationMetric(dataset, classified_dataset, unprivileged_group, privileged_group)
    print(metric.average_abs_odds_difference())
    if abs(metric.average_abs_odds_difference().round(3)) < 0.2:
        print('The algorithm can be considered to be not biased')
    else:
        print('There is a potential bias')

def entropy_index(random_data, predicted_data, target_variable, protected_variable, privileged_input, unprivileged_input):
    random_data['Pred'] = np.random.binomial(1, .5, 1000)
    dataset = BinaryLabelDataset(df=random_data, label_names=[target_variable], protected_attribute_names=[protected_variable])
    classified_dataset = BinaryLabelDataset(df=predicted_data, label_names=[target_variable], protected_attribute_names=[protected_variable])
    privileged_group = [{protected_variable: privileged_input}] #male=1
    unprivileged_group = [{protected_variable: unprivileged_input}] #female=0
    metric = ClassificationMetric(dataset, classified_dataset, unprivileged_group, privileged_group)
    print(metric.between_all_groups_generalized_entropy_index(alpha=2))
    if abs(metric.between_all_groups_generalized_entropy_index(alpha=2).round(3)) < 0.2:
        print('The algorithm can be considered to be not biased')
    else:
        print('There is a potential bias')

def coeff_variation(random_data, predicted_data, target_variable, protected_variable, privileged_input, unprivileged_input):
    random_data['Pred'] = np.random.binomial(1, .5, 1000)
    dataset = BinaryLabelDataset(df=random_data, label_names=[target_variable], protected_attribute_names=[protected_variable])
    classified_dataset = BinaryLabelDataset(df=predicted_data, label_names=[target_variable], protected_attribute_names=[protected_variable])
    privileged_group = [{protected_variable: privileged_input}] #male=1
    unprivileged_group = [{protected_variable: unprivileged_input}] #female=0
    metric = ClassificationMetric(dataset, classified_dataset, unprivileged_group, privileged_group)
    print(metric.between_group_coefficient_of_variation())
    if abs(metric.between_group_coefficient_of_variation().round(3)) < 0.2:
        print('The algorithm can be considered to be not biased')
    else:
        print('There is a potential bias')

def equal_opportunity(random_data, predicted_data, target_variable, protected_variable, privileged_input, unprivileged_input):
    random_data['Pred'] = np.random.binomial(1, .5, 1000)
    dataset = BinaryLabelDataset(df=random_data, label_names=[target_variable], protected_attribute_names=[protected_variable])
    classified_dataset = BinaryLabelDataset(df=predicted_data, label_names=[target_variable], protected_attribute_names=[protected_variable])
    privileged_group = [{protected_variable: privileged_input}] #male=1
    unprivileged_group = [{protected_variable: unprivileged_input}] #female=0
    metric = ClassificationMetric(dataset, classified_dataset, unprivileged_group, privileged_group)
    print(metric.equal_opportunity_difference())
    if abs(metric.equal_opportunity_difference().round(3)) < 0.2:
        print('The algorithm can be considered to be not biased')
    else:
        print('There is a potential bias')

def data_generator_fd(data):
    X_new = pd.DataFrame(columns=[])

    for c in data.columns:
        if is_binary(data[c]):
            X_new[c] = np.random.binomial(1, .5, 1000)
        else:
            X_new[c] = np.random.normal(data[c].describe()[1], data[c].describe()[2], 1000)
    return X_new

def data_generator_fi(c_name1, c_name2, c_dt_1, c_dt_2, c1_mean=None, c2_mean=None, c1_std=None, c2_std=None, c3_mean=None,
                      c4_mean=None, c5_mean=None, c6_mean=None, c3_std=None, c4_std=None, c5_std=None,
                      c6_std=None, c3_name=None, c4_name=None, c5_name=None, c6_name=None, c3_dt=None,
                      c4_dt=None, c5_dt=None, c6_dt=None):
    print('start')
    X_new = pd.DataFrame(columns=[])
    if c_dt_1.lower() == 'binary':
        X_new[c_name1] = np.random.binomial(1, .5, 1000)
    if c_dt_1.lower() == 'numerical':
        X_new[c_name1] = np.random.normal(c1_mean, c1_std, 1000)
    if c_dt_2.lower() == 'binary':
        X_new[c_name2] = np.random.binomial(1, .5, 1000)
    if c_dt_2.lower() == 'numerical':
        X_new[c_name2] = np.random.normal(c2_mean, c2_std, 1000)
    return X_new


def create_eval(pre_trained_model, data):
    pred_y_n = pre_trained_model.predict(data)
    df_n = pd.DataFrame(pred_y_n, columns=['Pred'])
    pred_n = pd.concat([data.reset_index(drop='True'),df_n.reset_index(drop='True')],axis=1)
    return pred_n

def is_binary(series):
    return sorted(series.unique()) == [0,1]

import pandas as pd
import numpy as np
from typing import List, Union, Dict
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.datasets import BinaryLabelDataset
from aif360.explainers import MetricTextExplainer
import seaborn as sns
import matplotlib.pyplot as plt
import statistics

def feature_importance(model, data):
    imp = np.abs(model.coef_.squeeze())
    var = np.zeros(shape=imp.shape)
    return pd.DataFrame({'feature': data.columns.to_list(), 'importance': imp}).sort_values('importance', ascending=False)


def plot_feature_importance(**kwargs) -> None:
    #print('The protected column is: ', str(data[data['protected'].values]['col_id'].values))
    ax = sns.barplot(**kwargs)
    for l in ax.get_xticklabels():
        l.set_rotation(90)

def get_disparity_index(di):
    return 1 - np.minimum(di, 1 / di)


def calc_disparity_index(data, target_variable, protected_variable, unprivileged_input):
    df_aif = BinaryLabelDataset(df=data, label_names=[target_variable],
                                protected_attribute_names=[protected_variable])
    privileged_group = []
    for v in data[protected_variable].unique()[data[protected_variable].unique() != unprivileged_input]:
        privileged_group.append({protected_variable: v})
    unprivileged_group = [{protected_variable: unprivileged_input}] #female=0
    metric_orig = BinaryLabelDatasetMetric(df_aif, unprivileged_group, privileged_group)
    print('1-min(DI, 1/DI):', get_disparity_index(metric_orig.disparate_impact()).round(3))
    if get_disparity_index(metric_orig.disparate_impact()).round(3) < 0.2:
        print('The algorithm can be considered to be not biased')
    else:
        print('There is a potential bias')

def disparity_values(data, target_variable, protected_variable, unprivileged_input):
    df_aif = BinaryLabelDataset(df=data, label_names=[target_variable],
                                protected_attribute_names=[protected_variable])
    privileged_group = []
    for v in data[protected_variable].unique()[data[protected_variable].unique() != unprivileged_input]:
        privileged_group.append({protected_variable: v})
    unprivileged_group = [{protected_variable: unprivileged_input}] #female=0
    metric_orig = BinaryLabelDatasetMetric(df_aif, unprivileged_group, privileged_group)
    return abs(get_disparity_index(metric_orig.disparate_impact()).round(3))


def calc_stat_parity(data, target_variable, protected_variable, unprivileged_input):
    df_aif = BinaryLabelDataset(df=data, label_names=[target_variable],
                                protected_attribute_names=[protected_variable])
    privileged_group = []
    for v in data[protected_variable].unique()[data[protected_variable].unique() != unprivileged_input]:
        privileged_group.append({protected_variable: v})
    unprivileged_group = [{protected_variable: unprivileged_input}] #female=0
    metric_orig = BinaryLabelDatasetMetric(df_aif, unprivileged_group, privileged_group)
    print(metric_orig.statistical_parity_difference().round(3))
    if abs(metric_orig.statistical_parity_difference().round(3)) < 0.1:
        print('The algorithm can be considered to be not biased')
    else:
        print('There is a potential bias')

def calc_mean_diff(data, target_variable, protected_variable, unprivileged_input):
    df_aif = BinaryLabelDataset(df=data, label_names=[target_variable],
                                protected_attribute_names=[protected_variable])
    privileged_group = []
    for v in data[protected_variable].unique()[data[protected_variable].unique() != unprivileged_input]:
        privileged_group.append({protected_variable: v})
    unprivileged_group = [{protected_variable: unprivileged_input}] #female=0
    metric_orig = BinaryLabelDatasetMetric(df_aif, unprivileged_group, privileged_group)
    print(metric_orig.mean_difference().round(3))
    if abs(metric_orig.mean_difference().round(3)) < 0.2:
        print('The algorithm can be considered to be not biased')
    else:
        print('There is a potential bias')

def mean_diff_values(data, target_variable, protected_variable, unprivileged_input):
    df_aif = BinaryLabelDataset(df=data, label_names=[target_variable],
                                protected_attribute_names=[protected_variable])
    privileged_group = []
    for v in data[protected_variable].unique()[data[protected_variable].unique() != unprivileged_input]:
        privileged_group.append({protected_variable: v})
    unprivileged_group = [{protected_variable: unprivileged_input}] #female=0
    metric_orig = BinaryLabelDatasetMetric(df_aif, unprivileged_group, privileged_group)
    return abs(metric_orig.mean_difference().round(3))

def odds_diff(random_data, predicted_data, target_variable, protected_variable, unprivileged_input):
    random_data['Pred'] = np.random.binomial(1, .5, 1000)
    dataset = BinaryLabelDataset(df=random_data, label_names=[target_variable], protected_attribute_names=[protected_variable])
    classified_dataset = BinaryLabelDataset(df=predicted_data, label_names=[target_variable], protected_attribute_names=[protected_variable])
    privileged_group = []
    for v in predicted_data[protected_variable].unique()[predicted_data[protected_variable].unique() != unprivileged_input]:
        privileged_group.append({protected_variable: v})
    unprivileged_group = [{protected_variable: unprivileged_input}] #female=0
    metric = ClassificationMetric(dataset, classified_dataset, unprivileged_group, privileged_group)
    print(metric.average_abs_odds_difference())
    if abs(metric.average_abs_odds_difference().round(3)) < 0.2:
        print('The algorithm can be considered to be not biased')
    else:
        print('There is a potential bias')

def entropy_index(random_data, predicted_data, target_variable, protected_variable, unprivileged_input):
    random_data['Pred'] = np.random.binomial(1, .5, 1000)
    dataset = BinaryLabelDataset(df=random_data, label_names=[target_variable], protected_attribute_names=[protected_variable])
    classified_dataset = BinaryLabelDataset(df=predicted_data, label_names=[target_variable], protected_attribute_names=[protected_variable])
    privileged_group = []
    for v in predicted_data[protected_variable].unique()[predicted_data[protected_variable].unique() != unprivileged_input]:
        privileged_group.append({protected_variable: v})
    unprivileged_group = [{protected_variable: unprivileged_input}] #female=0
    metric = ClassificationMetric(dataset, classified_dataset, unprivileged_group, privileged_group)
    print(metric.between_all_groups_generalized_entropy_index(alpha=2))
    if abs(metric.between_all_groups_generalized_entropy_index(alpha=2).round(3)) < 0.2:
        print('The algorithm can be considered to be not biased')
    else:
        print('There is a potential bias')

def coeff_variation(random_data, predicted_data, target_variable, protected_variable, unprivileged_input):
    random_data['Pred'] = np.random.binomial(1, .5, 1000)
    dataset = BinaryLabelDataset(df=random_data, label_names=[target_variable], protected_attribute_names=[protected_variable])
    classified_dataset = BinaryLabelDataset(df=predicted_data, label_names=[target_variable], protected_attribute_names=[protected_variable])
    privileged_group = []
    for v in predicted_data[protected_variable].unique()[predicted_data[protected_variable].unique() != unprivileged_input]:
        privileged_group.append({protected_variable: v})
    unprivileged_group = [{protected_variable: unprivileged_input}] #female=0
    metric = ClassificationMetric(dataset, classified_dataset, unprivileged_group, privileged_group)
    print(metric.between_group_coefficient_of_variation())
    if abs(metric.between_group_coefficient_of_variation().round(3)) < 0.2:
        print('The algorithm can be considered to be not biased')
    else:
        print('There is a potential bias')

def equal_opportunity(random_data, predicted_data, target_variable, protected_variable, unprivileged_input):
    random_data['Pred'] = np.random.binomial(1, .5, 1000)
    dataset = BinaryLabelDataset(df=random_data, label_names=[target_variable], protected_attribute_names=[protected_variable])
    classified_dataset = BinaryLabelDataset(df=predicted_data, label_names=[target_variable], protected_attribute_names=[protected_variable])
    privileged_group = []
    for v in predicted_data[protected_variable].unique()[predicted_data[protected_variable].unique() != unprivileged_input]:
        privileged_group.append({protected_variable: v})
    unprivileged_group = [{protected_variable: unprivileged_input}] #female=0
    metric = ClassificationMetric(dataset, classified_dataset, unprivileged_group, privileged_group)
    print(metric.equal_opportunity_difference())
    if abs(metric.equal_opportunity_difference().round(3)) < 0.2:
        print('The algorithm can be considered to be not biased')
    else:
        print('There is a potential bias')

def equal_ops_values(random_data, predicted_data, target_variable, protected_variable, unprivileged_input):
    random_data['Pred'] = np.random.binomial(1, .5, 1000)
    dataset = BinaryLabelDataset(df=random_data, label_names=[target_variable], protected_attribute_names=[protected_variable])
    classified_dataset = BinaryLabelDataset(df=predicted_data, label_names=[target_variable], protected_attribute_names=[protected_variable])
    privileged_group = []
    for v in predicted_data[protected_variable].unique()[predicted_data[protected_variable].unique() != unprivileged_input]:
        privileged_group.append({protected_variable: v})
    unprivileged_group = [{protected_variable: unprivileged_input}] #female=0
    metric = ClassificationMetric(dataset, classified_dataset, unprivileged_group, privileged_group)
    return abs(metric.equal_opportunity_difference())

def data_input(mock):
    X_new = pd.DataFrame(columns=[])
    for ind in mock.index:
        if mock['data_type'][ind] == 'binary':
            X_new[mock['col_id'][ind]] = np.random.binomial(1, .5, 1000)
        if mock['data_type'][ind] == 'categorical':
            X_new[mock['col_id'][ind]] = np.random.randint(mock['max'][ind]+1, size=1000)
        if mock['data_type'][ind] == 'numerical':
            X_new[mock['col_id'][ind]] = np.random.normal(mock['mean'][ind], mock['std'][ind], 1000)
    return X_new

def data_generator_fd(data):
    X_new = pd.DataFrame(columns=[])

    for c in data.columns:
        if is_binary(data[c]):
            X_new[c] = np.random.binomial(1, .5, 1000)
        if data[c].dtype == 'int64':
            X_new[c] = np.random.randint(data[c].describe()[7]+1, size=1000)
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

def create_binary(data, target_variable, protected_variable, unprivileged_input):
    df_aif = BinaryLabelDataset(df=data, label_names=[target_variable],
                                protected_attribute_names=[protected_variable])
    privileged_group = []
    for v in data[protected_variable].unique()[data[protected_variable].unique() != unprivileged_input]:
        privileged_group.append({protected_variable: v})
    unprivileged_group = [{protected_variable: unprivileged_input}] #female=0
    return BinaryLabelDatasetMetric(df_aif, unprivileged_groups=unprivileged_group, privileged_groups=privileged_group)

def create_eval(pre_trained_model, data):
    pred_y_n = pre_trained_model.predict(data)
    df_n = pd.DataFrame(pred_y_n, columns=['Pred'])
    pred_n = pd.concat([data.reset_index(drop='True'),df_n.reset_index(drop='True')],axis=1)
    return pred_n

def is_binary(series):
    return sorted(series.unique()) == [0,1]

class MetricAdditions:
    def explain(self,
                disp: bool=True) -> Union[None, str]:
        """Explain everything available for the given metric."""

        # Find intersecting methods/attributes between MetricTextExplainer and provided metric.
        inter = set(dir(self)).intersection(set(dir(self.metric)))

        # Ignore private and dunder methods
        metric_methods = [getattr(self, c) for c in inter if c.startswith('_') < 1]

        # Call methods, join to new lines
        s = "\n".join([f() for f in metric_methods if callable(f)])

        if disp:
            print(s)
        else:
            return s
        
        
class MetricTextExplainer_(MetricTextExplainer, MetricAdditions):
    """Combine explainer and .explain."""
    pass

def create_boxplots(input, model, disadvantaged):
    for c in input[input['protected']]['col_id']: c
    disparity = []
    for i in range(100):
        X_new = data_input(input)
        pred_n = create_eval(model, X_new)
        disparity.append(disparity_values(pred_n, 'Pred', c, disadvantaged))
    sns.set(style="whitegrid")
    print('With an iteration of 100, the values')
    print('of the disparity index are shown below:')
    ax = sns.boxplot(x=disparity)
    plt.show()
    
    mean_diff = []
    for i in range(100):
        X_new = data_input(input)
        pred_n = create_eval(model, X_new)
        mean_diff.append(mean_diff_values(pred_n, 'Pred', c, disadvantaged))
    sns.set(style="whitegrid")
    print('With an iteration of 100, the values')
    print('of the mean difference are shown below:')
    ax = sns.boxplot(x=mean_diff)
    plt.show()
    
    equal_ops = []
    for i in range(100):
        X_new = data_input(input)
        pred_n = create_eval(model, X_new)
        equal_ops.append(equal_ops_values(X_new,pred_n, 'Pred', c, disadvantaged))    
    print('With an iteration of 100, the values')
    print('of the equal opportunity are shown below:')
    ax = sns.boxplot(x=equal_ops)  
    plt.show()

def create_output(input, model, disadvantaged, request_id):
    for c in input[input['protected']]['col_id']: c
    output = pd.DataFrame(columns=['request_id'])
    output = output.set_index('request_id')
    disparity = []
    for i in range(100):
        X_new = data_input(input)
        pred_n = create_eval(model, X_new)
        disparity.append(disparity_values(pred_n, 'Pred', c, disadvantaged))
    output.loc[request_id, 'disparity_index'] = statistics.mean(disparity)

    mean_diff = []
    for i in range(100):
        X_new = data_input(input)
        pred_n = create_eval(model, X_new)
        mean_diff.append(mean_diff_values(pred_n, 'Pred', c, disadvantaged))
    output.loc[request_id, 'mean_difference'] = statistics.mean(mean_diff)
    
    equal_ops = []
    for i in range(100):
        X_new = data_input(input)
        pred_n = create_eval(model, X_new)
        equal_ops.append(equal_ops_values(X_new,pred_n, 'Pred', c, disadvantaged))
    output.loc[request_id, 'equal_opportunity'] = statistics.mean(equal_ops)

    positive = []
    for i in range(100):
        X_new = data_input(input)
        pred_n = create_eval(model, X_new)
        positive.append(pred_n['Pred'].sum())
    output.loc[request_id, 'positive_outcomes'] = statistics.mean(positive)/1000
    return output
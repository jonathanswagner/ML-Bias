import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.axes

from sklearn.svm import SVC,LinearSVC, SVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.externals import joblib

from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, classification_report 
from sklearn.metrics import precision_recall_curve, confusion_matrix, roc_curve, auc

from scipy.stats import skew, kurtosis

from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split, cross_val_score

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

#plt.style.use('dark_background')
current_palette = sns.color_palette('colorblind')
sns.palplot(current_palette)


def preprocess(data):
    data.LoanAmount = data.LoanAmount*1000
    data.Loan_Status.value_counts(normalize = True).reset_index()
    data.dropna(inplace=True)

def prop_check(data):
    f, axes = plt.subplots(6,2,figsize= (12,20))
    plt.suptitle('Train data, count vs proportion of each object feature vs Loan_Status', size =16, y = 0.9)
    col = data.columns[1:data.shape[1]-1]
    r = 0
    for i in col:
        if (data.dtypes == 'object')[i]:        
            data_prop = (data['Loan_Status']
                          .groupby(data[i])
                          .value_counts(normalize = True)
                          .rename('prop')
                          .reset_index())
            sns.countplot(data = data, 
                          x ='Loan_Status', 
                          hue = i, 
                          ax = axes[r,0], 
                          hue_order=data_prop[i].unique(), 
                          palette=current_palette)
            sns.barplot(data = data_prop, 
                        x = 'Loan_Status', 
                        y = 'prop',
                        hue = i,
                        ax = axes[r,1],
                        palette=current_palette)
            r = r+1

def make_index(data):
    data.set_index('Loan_ID', inplace=True)
    return data

def categorize(data):
    data.Gender.replace({'Male': 1, 'Female': 0}, inplace = True)
    data.Married.replace({'Yes': 1, 'No': 0}, inplace = True)
    data.Education.replace({'Graduate': 1, 'Not Graduate': 0}, inplace = True)
    data.Self_Employed.replace({'Yes': 1, 'No': 0}, inplace = True)
    data = data.join(pd.get_dummies(data.Dependents, prefix='Dependents'))
    data.drop(columns= ['Dependents', 'Dependents_3+'], inplace=True)
    data = data.join(pd.get_dummies(data.Property_Area, prefix='Property_Area'))
    data.drop(columns= ['Property_Area', 'Property_Area_Rural'], inplace=True)
    return data

def add_feat(data):
    ln_monthly_return = np.log(data.LoanAmount/data.Loan_Amount_Term)
    data['ln_monthly_return'] = (ln_monthly_return - np.mean(ln_monthly_return))/(np.std(ln_monthly_return)/np.sqrt(len(ln_monthly_return)))
    
    ln_total_monthly_income = np.log(data.ApplicantIncome + data.CoapplicantIncome)
    data['ln_total_income'] = (ln_total_monthly_income - np.mean(ln_total_monthly_income))/(np.std(ln_total_monthly_income)/np.sqrt(len(ln_total_monthly_income)))
    
    ln_LoanAmount = np.log(1000*data.LoanAmount)
    data['ln_LoanAmount'] = (ln_LoanAmount - np.mean(ln_LoanAmount))/(np.std(ln_LoanAmount)/np.sqrt(len(ln_LoanAmount)))
    
    
    return data

def norm_plt(df):
    f, axes = plt.subplots(3,2,figsize= (12,15),squeeze=False)

    ######total income########
    sns.distplot(df.ln_total_income
                 ,ax=axes[0,0]).set_title('ln(total_income) norm distribution')
    #axes[0,0].set_xlim(-100,100)
    axes[0,0].text(0.03, 0.85,
                   'skew: {0:0.2}\nkurtosis: {1:0.2f}'
                   .format(skew(df.ln_total_income),
                                          kurtosis(df.ln_total_income)),
                   horizontalalignment='left',
                   verticalalignment='bottom',
                   transform=axes[0,0].transAxes,
                   bbox={'facecolor': 'white'})
    sns.distplot((df.ApplicantIncome+df.CoapplicantIncome),
                 ax=axes[0,1]).set_title('total_income distribution')
    axes[0,1].text(0.7, 0.85,
                   'skew: {0:0.2f}\nkurtosis: {1:0.2f}'
                   .format(skew(df.ApplicantIncome+df.CoapplicantIncome),
                           kurtosis(df.ApplicantIncome+df.CoapplicantIncome)),
                   horizontalalignment='left',
                   verticalalignment='bottom',
                   transform=axes[0,1].transAxes,
                   bbox={'facecolor': 'white'})

    #######monthly return###########
    sns.distplot(df.ln_monthly_return,
                 ax=axes[1,0]).set_title('ln(monthly_return) norm distribution')
    #axes[1,0].set_xlim(-100,100)
    axes[1,0].text(0.03, 0.85,
                   'skew: {0:0.2}\nkurtosis: {1:0.2f}'
                   .format(skew(df.ln_monthly_return),
                           kurtosis(df.ln_monthly_return)),
                   horizontalalignment='left',
                   verticalalignment='bottom',
                   transform=axes[1,0].transAxes,
                   bbox={'facecolor': 'white'})

    sns.distplot((1000*df.LoanAmount/df.Loan_Amount_Term),
                 ax=axes[1,1]).set_title('monthly_return distribution')
    axes[1,1].text(0.7, 0.85,
                   'skew: {0:0.2f}\nkurtosis: {1:0.2f}'
                   .format(skew(df.LoanAmount/df.Loan_Amount_Term),
                           kurtosis(df.LoanAmount/df.Loan_Amount_Term)),
                   horizontalalignment='left',
                   verticalalignment='bottom',
                   transform=axes[1,1].transAxes,
                   bbox={'facecolor': 'white'})

    ######norm ln_LoanAmount########
    sns.distplot(df.ln_LoanAmount
                 ,ax=axes[2,0]).set_title('ln(LoanAmount) norm distribution')
    #axes[2,0].set_xlim(-100,100)
    axes[2,0].text(0.03, 0.85,
                   'skew: {0:0.2}\nkurtosis: {1:0.2f}'
                   .format(skew(df.ln_LoanAmount),
                                          kurtosis(df.ln_LoanAmount)),
                   horizontalalignment='left',
                   verticalalignment='bottom',
                   transform=axes[2,0].transAxes,
                   bbox={'facecolor': 'white'})
    sns.distplot((df.LoanAmount),
                 ax=axes[2,1]).set_title('LoanAmount distribution')
    axes[2,1].text(0.7, 0.85,
                   'skew: {0:0.2f}\nkurtosis: {1:0.2f}'
                   .format(skew(df.LoanAmount),
                           kurtosis(df.LoanAmount)),
                   horizontalalignment='left',
                   verticalalignment='bottom',
                   transform=axes[2,1].transAxes,
                   bbox={'facecolor': 'white'})
    
    
    ####### adding grid to the graph#########
    for i in range(3):
        for j in range(2):
            axes[i,j].grid(b=True, which='both', axis='both', color='grey', linestyle = '--', linewidth = '0.3')
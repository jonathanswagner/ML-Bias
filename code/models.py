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
#sns.palplot(current_palette)

def cv_check(X,y, CV):
    models = [
        RandomForestClassifier(criterion='gini',
                               n_estimators=50,
                               max_depth=11,
                               max_features=6,
                               random_state=42,
                               class_weight='balanced_subsample',
                               n_jobs=4),
        SVC(C=1, kernel='rbf', gamma='auto',random_state=42,class_weight='balanced'),
        LogisticRegression(solver='lbfgs',
                           multi_class='ovr',
                           max_iter=500,
                           C=1,
                           random_state=42,
                           class_weight='balanced'),
        GaussianNB(),
        #LinearSVC(C=1, 
        #         max_iter=500,
        #          random_state=0),
        DummyClassifier(strategy='most_frequent',random_state=42)
    ]

    entries = []
    
    for model in models:
        model_name = model.__class__.__name__
        print ("Currently fitting: {}".format(model_name))
        accuracies = cross_val_score(model,
                                     X,
                                     y, 
                                     scoring='roc_auc', cv=CV, n_jobs=4)
        for fold_idx, accuracy in enumerate(accuracies):
            entries.append((model_name, fold_idx, accuracy))
        cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'roc_auc'])
        
    return cv_df

def cv_bp(cv_df, title, axes):
    axes.grid(b=True, 
              which='both', 
              axis='both', 
              color='grey', 
              linestyle = '--', 
              linewidth = '0.3')    
    sns.boxplot(x='model_name', 
                y='roc_auc', 
                data=cv_df, 
                width = 0.5, 
                ax=axes,
                palette=current_palette).set_title(title)
    sns.stripplot(x='model_name', 
                  y='roc_auc',
                  data=cv_df, 
                  size=5, jitter=True, 
                  edgecolor="grey", 
                  linewidth=1, 
                  ax=axes)
    plt.ylim(0.2,1)
    plt.savefig('{}.png'.format(title), format='png')

def model_score(train, model, grid_values, scorers_list):
    X_train = train.drop(columns=['Loan_Status'])
    y_train = train['Loan_Status']
    
    clf_dict = {}
    
    for i, scorer in enumerate(scorers_list):
        clf_eval = GridSearchCV(model, param_grid=grid_values, scoring=scorer, cv=5, iid=False)
        clf_eval.fit(X_train,y_train)
        print('Grid best parameters for {0}: {1} scoring: {2}'
              .format(scorer, clf_eval.best_params_, round(clf_eval.best_score_,3)))
        clf_dict[scorer] = clf_eval
    return clf_dict

def mod_eval(df,predictions, predprob, y_test, title):
    # prints confusion matrix heatmap    
    cm = confusion_matrix(df.Loan_Status[y_test.index], predictions)
    sns.heatmap(cm, annot=True, fmt='.3g', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes']).set_title(title)
    plt.xlabel('Real')
    plt.ylabel('Predict')
    
    print(classification_report(df.Loan_Status[y_test.index], predictions))
    
    f, axes = plt.subplots(1,2,figsize= (20,6),squeeze=False)

    fpr, tpr, _ = roc_curve(df.Loan_Status[y_test.index], predprob[:,1])
    roc_auc = auc(fpr,tpr)
    axes[0,0].plot(fpr, tpr, lw=3)
    axes[0,0].set_title('{} ROC curve (area = {:0.2f})'.format(title, roc_auc))
    axes[0,0].set(xlabel='False Positive Rate',ylabel='True Positive Rate')
    axes[0,0].grid(b=True, which='both', axis='both', color='grey', linestyle = '--', linewidth = '0.3')

    precision, recall, thresholds = precision_recall_curve(y_test, predprob[:,1])
    best_index = np.argmin(np.abs(precision-recall)) # set the best index to be the minimum delta between precision and recall
    axes[0,1].plot(precision,recall)
    axes[0,1].set_title('{} Precision-Recall Curve'.format(title))
    axes[0,1].set(xlabel='Precision', ylabel='Recall', xlim=(0.4,1.05))
    axes[0,1].plot(precision[best_index],recall[best_index],'o',color='r')
    axes[0,1].grid(b=True, which='both', axis='both', color='grey', linestyle = '--', linewidth = '0.3')

def model_training(classifier,df, ts):
    clf = classifier
    t=df.drop(columns=['Loan_Status'])
    X_train, X_test, y_train, y_tests = train_test_split(t,
                                                         df['Loan_Status'],
                                                         test_size=ts,
                                                         stratify=df['Loan_Status'])
    clf.fit(X_train, y_train)
    return clf
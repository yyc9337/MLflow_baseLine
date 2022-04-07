#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from imblearn.over_sampling import BorderlineSMOTE

def stratified_kfold(model_dict, nsplit, data, features, label, threshold=0.5):

    kfold_accs = []
    kfold_precisions = []
    kfold_recalls = []
    kfold_f1s = []
    kfold_specificities = []
    kfold_aucs = []
    kfold_metric_lists = [kfold_accs, kfold_precisions, kfold_recalls, kfold_f1s, kfold_specificities, kfold_aucs]

    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'Specificity', 'AUC']
    
    fprs_tprs = []

    stkfold = StratifiedKFold(n_splits=nsplit)

    for c_iter, (train_idx, valid_idx) in enumerate(stkfold.split(data[features], data[label])):

        X_train, X_valid = data[features].iloc[train_idx], data[features].iloc[valid_idx]
        y_train, y_valid = data[label].iloc[train_idx], data[label].iloc[valid_idx]
            

        print(f"{c_iter+1}번째 교차검증")
        print('-'*30)
        print(f"학습 데이터 개수: {len(y_train)}")
        print(f"검증 데이터 개수: {len(y_valid)}")
        print('-'*30)

        models = model_dict

        accs = []
        precisions = []
        recalls = []
        f1s = []
        specificities = []
        aucs = []
        metric_lists = [accs, precisions, recalls, f1s, specificities, aucs]
        
        fpr_tpr = []
        

        for model_name, model in models.items():
            model.fit(X_train, y_train)

            prediction = (model.predict_proba(X_valid)[:, 1] > threshold).astype(np.int)
            prediction_proba = model.predict_proba(X_valid)[:, 1]

            acc = accuracy_score(y_valid, prediction)
            precision = precision_score(y_valid, prediction)
            recall = recall_score(y_valid, prediction)
            f1 = f1_score(y_valid, prediction)
            cm = confusion_matrix(y_valid, prediction)
            specificity = cm[0, 0]/(cm[0, 0] + cm[0, 1])
            auc = roc_auc_score(y_valid, prediction_proba)
            metrics = [acc, precision, recall, f1, specificity, auc]

            for metric_list, metric in zip(metric_lists, metrics):
                metric_list.append(metric)
            
            fpr, tpr, _ = roc_curve(y_valid, prediction_proba)
            fpr_tpr.append([fpr, tpr])

            print(model_name)
            for metric_name, metric in zip(metric_names, metrics):
                print(f"{metric_name}: {np.round(metric, 2)}")
            print('-'*30)
        print('='*50)
        for kfold_metric_list, metric_list in zip(kfold_metric_lists, metric_lists):
            kfold_metric_list.append(metric_list)
        fprs_tprs.append(fpr_tpr)
    fprs_tprs = np.array(fprs_tprs)
    
    fprs_tprs_dict = dict()
    for model_idx, model_name in enumerate(models.keys()):
        fprs_tprs_dict[model_name] = fprs_tprs[:, model_idx, :]

    print('\nK-Fold Average')
    print('-'*30)
    for model_idx, model_name in enumerate(models.keys()):
        print(model_name)
        for metric_name, kfold_metric_list in zip(metric_names, kfold_metric_lists):
            kfold_metric_array = np.vstack(kfold_metric_list)[:, model_idx]
            metric_mean = kfold_metric_array.mean()
            print(f"{metric_name}:{np.round(metric_mean, 2)}")
        print('-'*30)
    
    average_kfold_metric = {
    'Accuracy':np.array(kfold_accs).mean(axis=0),
    'Precision':np.array(kfold_precisions).mean(axis=0),
    'Recall':np.array(kfold_recalls).mean(axis=0),
    'F1-score':np.array(kfold_f1s).mean(axis=0),
    'Specificity':np.array(kfold_specificities).mean(axis=0),
    'AUC':np.array(kfold_aucs).mean(axis=0)
    }
    
    return average_kfold_metric, fprs_tprs_dict
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix
import seaborn as sns
from matplotlib.collections import EllipseCollection
from matplotlib.colors import Normalize

# 한글 폰트 설정
plt.rc("font", family='Malgun Gothic')

# - 깨짐 설정
plt.rcParams['axes.unicode_minus'] = False


def roc_graph(ax, y_true, y_prediction_proba, **kwargs):
    fpr, tpr, _ = roc_curve(y_true, y_prediction_proba)
    ax.plot(fpr, tpr, **kwargs)
    ax.set_xlabel('False Positive Rate', fontsize=10)
    ax.set_ylabel('True Positive Rate', fontsize=10)
    ax.set_title('ROC Curve', fontsize=15)
    ax.fill_between(fpr, tpr, alpha=0.3, **kwargs)

def confusion_matrix_heatmap(y_data, prediction):
    cm = confusion_matrix(y_data, prediction)

    fig, ax = plt.subplots(figsize=(10, 10))
    
    ax = sns.heatmap(cm, 
                        linewidths = 0.1,
                       square=True, cmap=plt.cm.PuBu,
                       linecolor='white', annot=True, annot_kws={'size':20}, fmt='d')
    
    ax.set_ylabel('Ground truth', fontsize=20)
    ax.set_xlabel('Prediction', fontsize=20)
    
    plt.show()
    
def category_data_distribution(data, columns, col_wrap=1, hue=None, color='#1f77b4', alpha=0.5, grid=False, **kwargs):
    row = len(columns)//col_wrap
    rest = len(columns)%col_wrap
    axes = []
    if rest == 0:
        fig = plt.figure(figsize=(col_wrap*5, row*5))
        for ax_idx in range(row*col_wrap):
            axes.append(fig.add_subplot(row, col_wrap, ax_idx+1))
    else:
        fig = plt.figure(figsize=(col_wrap*5, (row+1)*5))
        for ax_idx in range(len(columns)):
            axes.append(fig.add_subplot(row+1, col_wrap, ax_idx+1))
    
    if hue == None:
        for ax_idx, ax in enumerate(axes):
            col_unique, col_counts = data[columns[ax_idx]].value_counts().index, data[columns[ax_idx]].value_counts().values
            xticks = range(len(col_unique))
            ax.set_xticks(xticks)
            ax.set_xticklabels(col_unique)
            ax.bar(xticks, col_counts, color=color, alpha=alpha, **kwargs)
            ax.set_xlabel(columns[ax_idx], fontsize=15)
            ax.set_ylabel('Count', fontsize=15)
            ax.tick_params(labelsize=14, width=2.5, length=4)
            if grid == True:
                ax.grid(axis='y', color='gray', linestyle='--', alpha=0.7)
    else:
        uniques = data[hue].value_counts().sort_values(ascending=False).index.tolist()
        
        unique_datas = []
        for unique in uniques:
            unique_datas.append(data[data[hue]==unique])
        colors = ['#1f77b4', '#d62728' ,'#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']*100
        for ax_idx, ax in enumerate(axes):
            xticks = range(len(data[columns[ax_idx]].unique()))
            xticklabels = data[columns[ax_idx]].value_counts().index
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels)
            
            xtick_dict = dict()
            for xtick, xticklabel in enumerate(xticklabels):
                xtick_dict[xticklabel] = xtick
            
            for unique_idx, unique_data in enumerate(unique_datas):
                unique_col_unique, unique_col_counts = unique_data[columns[ax_idx]].value_counts().index, unique_data[columns[ax_idx]].value_counts().values
                unique_xticks = []
                for unique_col_unique_one in unique_col_unique:
                    unique_xticks.append(xtick_dict[unique_col_unique_one])
                ax.bar(unique_xticks, unique_col_counts, color=colors[unique_idx], alpha=alpha, label=uniques[unique_idx], **kwargs)
                
            ax.set_xlabel(columns[ax_idx], fontsize=15)
            ax.set_ylabel('Count', fontsize=15)
            ax.tick_params(labelsize=14, width=2.5, length=4)
            ax.legend(fontsize=13)
            if grid == True:
                ax.grid(axis='y', color='gray', linestyle='--', alpha=0.7)
        
    plt.tight_layout()
    return fig, axes

    
def histogram(data, columns, col_wrap=1, hue=None, color='#1f77b4', alpha=0.5, grid=False, **kwargs):
    row = len(columns)//col_wrap
    rest = len(columns)%col_wrap
    axes = []
    if rest == 0:
        fig = plt.figure(figsize=(col_wrap*5, row*5))
        for ax_idx in range(row*col_wrap):
            axes.append(fig.add_subplot(row, col_wrap, ax_idx+1))
    else:
        fig = plt.figure(figsize=(col_wrap*5, (row+1)*5))
        for ax_idx in range(len(columns)):
            axes.append(fig.add_subplot(row+1, col_wrap, ax_idx+1))
    
    if hue == None:
        for ax_idx, ax in enumerate(axes):
            sns.histplot(data=data, x=columns[ax_idx], ax=ax, color=color, alpha=alpha, **kwargs)
            ax.set_xlabel(columns[ax_idx], fontsize=15)
            ax.set_ylabel('Count', fontsize=15)
            ax.tick_params(labelsize=14, width=2.5, length=4)
            if grid == True:
                ax.grid(color='gray', linestyle='--', alpha=0.7)
    else:
        uniques = data[hue].value_counts().sort_values(ascending=False).index.tolist()
        
        unique_datas = []
        for unique in uniques:
            unique_datas.append(data[data[hue]==unique])
        colors = ['#1f77b4', '#d62728' ,'#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']*100
        for ax_idx, ax in enumerate(axes):
            for unique_idx, unique_data in enumerate(unique_datas):
                sns.histplot(data=unique_data, x=columns[ax_idx], ax=ax, color=colors[unique_idx], alpha=alpha, label=uniques[unique_idx], **kwargs)
            ax.set_xlabel(columns[ax_idx], fontsize=15)
            ax.set_ylabel('Count', fontsize=15)
            ax.tick_params(labelsize=14, width=2.5, length=4)
            ax.legend(fontsize=13)
            if grid == True:
                ax.grid(color='gray', linestyle='--', alpha=0.7)
            
    plt.tight_layout()
    return fig, axes


def plot_corr_ellipses(data, figsize=None, **kwargs):
    M = np.array(data)
    if not M.ndim == 2:
        raise ValueError('data must be a 2D array')
    fig, ax = plt.subplots(1, 1, figsize=figsize, subplot_kw={'aspect':'equal'})
    ax.set_xlim(-0.5, M.shape[1] - 0.5)
    ax.set_ylim(-0.5, M.shape[0] - 0.5)
    ax.invert_yaxis()

    # xy locations of each ellipse center
    xy = np.indices(M.shape)[::-1].reshape(2, -1).T

    # set the relative sizes of the major/minor axes according to the strength of
    # the positive/negative correlation
    w = np.ones_like(M).ravel() + 0.01
    h = 1 - np.abs(M).ravel() - 0.01
    a = 45 * np.sign(M).ravel()

    ec = EllipseCollection(widths=w, heights=h, angles=a, units='x', offsets=xy,
                           norm=Normalize(vmin=-1, vmax=1),
                           transOffset=ax.transData, array=M.ravel(), **kwargs)
    ax.add_collection(ec)

    # if data is a DataFrame, use the row/column names as tick labels
    if isinstance(data, pd.DataFrame):
        ax.set_xticks(np.arange(M.shape[1]))
        ax.set_xticklabels(data.columns, rotation=90)
        ax.set_yticks(np.arange(M.shape[0]))
        ax.set_yticklabels(data.index)

    return fig, ax, ec


def correlation(data, style='heatmap', figsize=None):
    corr = data.corr(method='pearson')
    if style == 'heatmap':
        fig, ax = plt.subplots(figsize=(figsize[0]*1.2, figsize[0]))
        ax = sns.heatmap(corr, vmin=-1, vmax=1,
                        cmap=sns.diverging_palette(20, 220, as_cmap=True), annot=True)
    
    elif style == 'ellipse':
        fig, ax, ec = plot_corr_ellipses(corr, figsize=figsize, cmap='bwr_r')
        cb = fig.colorbar(ec)
        cb.set_label('Correlation coefficient')
    
    plt.tight_layout()
    return fig, ax


def boxplot(data, columns, col_wrap=1, hue=None, color='#1f77b4', alpha=0.5, palette='RdBu', grid=False, **kwargs):
    row = len(columns)//col_wrap
    rest = len(columns)%col_wrap
    axes = []
    if rest == 0:
        fig = plt.figure(figsize=(col_wrap*5, row*5))
        for ax_idx in range(row*col_wrap):
            axes.append(fig.add_subplot(row, col_wrap, ax_idx+1))
    else:
        fig = plt.figure(figsize=(col_wrap*5, (row+1)*5))
        for ax_idx in range(len(columns)):
            axes.append(fig.add_subplot(row+1, col_wrap, ax_idx+1))
    
    if hue == None:
        for ax_idx, ax in enumerate(axes):
            sns.boxplot(data=data[columns[ax_idx]], color=color, ax=ax,  boxprops={'alpha':alpha}, **kwargs)
            ax.get_xaxis().set_visible(False)
            ax.set_ylabel(columns[ax_idx], fontsize=15)
            ax.tick_params(axis='y', labelsize=14, width=2.5, length=4)
            if grid == True:
                ax.grid(axis='y', color='gray', linestyle='--', alpha=0.7)
    else:
        for ax_idx, ax in enumerate(axes):
            sns.boxplot(data=data, x=hue, y=columns[ax_idx], palette=palette, ax=ax, **kwargs)
            ax.set_xlabel(hue, fontsize=15)
            ax.set_ylabel(columns[ax_idx], fontsize=15)
            ax.tick_params(axis='y', labelsize=14, width=2.5, length=4)
            if grid == True:
                ax.grid(axis='y', color='gray', linestyle='--', alpha=0.7)
            
    plt.tight_layout()
    return fig, axes


def violinplot(data, columns, col_wrap=1, hue=None, color='#1f77b4', alpha=0.5, palette='RdBu', grid=False, **kwargs):
    row = len(columns)//col_wrap
    rest = len(columns)%col_wrap
    axes = []
    if rest == 0:
        fig = plt.figure(figsize=(col_wrap*5, row*5))
        for ax_idx in range(row*col_wrap):
            axes.append(fig.add_subplot(row, col_wrap, ax_idx+1))
    else:
        fig = plt.figure(figsize=(col_wrap*5, (row+1)*5))
        for ax_idx in range(len(columns)):
            axes.append(fig.add_subplot(row+1, col_wrap, ax_idx+1))
    
    if hue == None:
        for ax_idx, ax in enumerate(axes):
            sns.violinplot(data=data[columns[ax_idx]], color=color, ax=ax , **kwargs)
            ax.collections[0].set_alpha(alpha)
            ax.get_xaxis().set_visible(False)
            ax.set_ylabel(columns[ax_idx], fontsize=15)
            ax.tick_params(axis='y', labelsize=14, width=2.5, length=4)
            if grid == True:
                ax.grid(axis='y', color='gray', linestyle='--', alpha=0.7)
    else:
        for ax_idx, ax in enumerate(axes):
            sns.violinplot(data=data, x=hue, y=columns[ax_idx], palette=palette, ax=ax, **kwargs)
            ax.set_xlabel(hue, fontsize=15)
            ax.set_ylabel(columns[ax_idx], fontsize=15)
            ax.tick_params(axis='y', labelsize=14, width=2.5, length=4)
            if grid == True:
                ax.grid(axis='y', color='gray', linestyle='--', alpha=0.7)
            
    plt.tight_layout()
    return fig, axes

def feature_importance(tree_model, ax, features, color='firebrick'):
    feature_names = np.array(features)
    feature_importances = np.array(tree_model.feature_importances_)
    
    sort_idx = np.argsort(feature_importances)
    sort_feature_names = feature_names[sort_idx]
    sort_feature_importances = feature_importances[sort_idx]
    
    yticks = range(len(feature_names))
    ytick_labels = sort_feature_names
    
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels)
    
    ax.barh(yticks, sort_feature_importances, color=color)
    
    ax.set_title('Feature Importance', fontsize=15)
    ax.set_ylim([-1, len(feature_names)])
    
    ax.grid(axis='x', linewidth=0.8, linestyle=':', alpha=0.8)
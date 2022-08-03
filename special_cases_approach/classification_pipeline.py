# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 17:12:28 2021

@author: GauthierFrecon
"""

import os 
import numpy as np
from pandas import DataFrame 
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut, cross_validate, cross_val_predict, ParameterGrid
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, recall_score, f1_score, accuracy_score, roc_auc_score, balanced_accuracy_score, roc_curve, RocCurveDisplay, precision_recall_curve
from collections import Counter
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

from radiomics_computation import radiomics_extraction

from scipy.stats import spearmanr, pearsonr, mannwhitneyu

from metrics_function import compute_metrics_sklearn, compute_metrics_option


# import rpy2.robjects as ro
# import rpy2.robjects.numpy2ri
# rpy2.robjects.numpy2ri.activate()



def LOO_cross_validation(X, y, remove_mask, valid_exclude_patients, mask_exclude_patients, model, list_patients_int, specificity_threshold=0.7, additional_comparison=False, bool_plot=True):
    
    """
    Evaluate a model by LOO cross validation, y_pred is predict with LOO prediction and then metrics are computed (accuracy, f1 score positive and negative, sensitivity, specifity, balanced accuracy, auc)

    Parameters
    ----------
    X : array shape (n_patients, n_features)
        Radiomics features array
    y : array int
        0 negative EI, 1 positive EI   
    remove_mask: bool
        remove patients relative to the mask, for instance remove nosuspicion of the study or not
    valid_exclude_patients: bool
        exclude of validation patients relative to mask_exclude patients, for instance exclude nosuspicion patients of validation, or exclude train/validation dataset of validation
    mask_nosuspicion: array of bool
        True: patients not from SVD database and others nosuspicion patients
    model : Pipeline sklearn or imblearn
        ML model, example: Pipeline([('sampling', ADASYN()), ('clf', LogisticRegression(C=0.01))])
    list_patients_int : list of int, shape (n_patients)
        int correspondant to the patient name
    specifity_threshold : float
        specificity threshold ex: 0.7 or 0.9 in case of optim_metric_name=balanced_accuracy
    additional_comparison : bool, optional
        confusion matrix and tpr fpr for roc curves. The default is False.
    bool_plot : bool, optional
        confusion matrix and tpr fpr for roc curves. The default is False.

    Returns
    -------
    results_cv: dict
        dict of the LOO evaluation metrics

    """
    
    if remove_mask:
        X = X[mask_exclude_patients, :]
        y = y[mask_exclude_patients]
        list_patients_int = np.array(list_patients_int)[mask_exclude_patients]
        mask_exclude_patients = mask_exclude_patients[mask_exclude_patients]

    model_name = str(model['clf']) + "+" + str(model['sampling'])

    loo = LeaveOneOut()
    if valid_exclude_patients:
        # loo but without nosuspicion patients : n_fold= n_patients - n_nosuspicion
        X_exclude_patients = X[~mask_exclude_patients, :]
        y_exclude_patients = y[~mask_exclude_patients]
        X = X[mask_exclude_patients, :]
        y = y[mask_exclude_patients]
        list_patients_int = np.array(list_patients_int)[mask_exclude_patients]
        n_fold = len(y)
        
        y_proba = np.zeros((n_fold,2))
        

        for fold, (train_index, valid_index) in enumerate(loo.split(X, y)):
            
            
            ### Dividing data into folds
            train_index = np.array(train_index, dtype="int")
            
            x_train_fold = X[train_index]
            y_train_fold = y[train_index]
            
            # add nosuspicion patients to train if nosuspicion patients excluded of valid:
            x_train_fold = np.concatenate( (x_train_fold, X_exclude_patients), 0)
            y_train_fold = np.concatenate( (y_train_fold, y_exclude_patients), 0)
            
            model.fit(x_train_fold, y_train_fold) ### RESET TO DO ? ###
            x_valid_fold = X[valid_index]
            y_valid_fold = y[valid_index]  ### unuse
            y_proba[fold] = model.predict_proba(x_valid_fold)[0]
            
            
    else: 
        y = np.reshape(y, (X.shape[0],))
        y_proba = cross_val_predict(model, X, y, method ="predict_proba", cv=loo, n_jobs=-1, verbose=0)    #proba to predict 0 (negative) or  1 (positive to EI)
    
    
    ## option3 : smooth roc curve in R, fixed a specificity that fixed a threshold probability and compute other metrics
    y_proba = y_proba[:, 1]
    y_pred, accuracy, specificity, sensitivity, f1_score_positive, f1_score_negative, balanced_accuracy, auc = compute_metrics_option(y, y_proba, specificity_threshold, bool_plot)

    
    #metrics results put in a dictionnary
    results_cv = {"accuracy": accuracy, "f1_score_positive": f1_score_positive, "f1_score_negative": f1_score_negative, "sensitivity": sensitivity, "specificity": specificity, 'balanced_accuracy': balanced_accuracy, 'auc':auc }
    patients_misclassified = np.array(list_patients_int)[y_pred!=y]
    results_cv["patients_misclassified"] = patients_misclassified

       
    return(results_cv) 
    


def gridsearch_loo(X, y, remove_nosuspicion, valid_exclude_nosuspicion, mask_nosuspicion, pipeline, dict_mod, list_patients_int, loop_evaluation=3, optim_metric_name="auc", specificity_threshold= 0.7, additional_comparison=False):
    """
    compute a parameters grid (sklearn class) and evaluate  models associated to each set of parameters of te grid (iterations:  loop_evaluation times) with LOO_cross_validation.

    Parameters
    ----------
    X : array shape (n_patients, n_features)
        Radiomics features array
    y : array int
        0 negative EI, 1 positive EI  
    remove_nosuspicion: bool, params of function LOO_cross_validation
        remove nosuspicion patients of the study or not
    valid_exclude_nosuspicion: bool, params of function LOO_cross_validation
        exclude nosuspicion patients of validation
    mask_nosuspicion: array of bool, params of function LOO_cross_validation
        True: patients not from SVD database and nosuspicion others patients
    pipeline : Pipeline sklearn or imblearn
        ML model, example: Pipeline([('sampling', ADASYN()), ('clf', LogisticRegression(C=0.01))])
    dict_mod : dict
        Dictionnary as input of Parametergrid to set parameters of pipelines. example: LR_lasso_dict = {'sampling': sampling_opt, 'clf': [LogisticRegression()], 'clf__C': [1, 0.1, 0.01]]}  
    list_patients_int : list of int, shape (n_patients)
        int correspondant to the patient name
    loop_evaluation : int, optional
        number of iterations for the evaluation of one model, after the metrics means are taken (stud evaluate for auc). The default is 3.
    optim_metric_name: str
        metric to select hyperparameters: "auc" or "balanced_accuracy"
    specifity_threshold : float
        specificity threshold ex: 0.7 or 0.9
    additional_comparison : bool, optional
        confusion matrix and tpr fpr for roc curves, only for one model evalluation not all loop_evaluation iterations. The default is False.

    Returns
    -------
    metrics: array, shape (n_grid, n_cols) with n_cols = n_infos + n_metrics + 1
        3 first columns: model name, sampling, parameters. then 6 columns: metrics means, last columns: std depends of index_optim_metric
    dict_best: metrics with respect to the parameter with the highest auc score
        contain index of the best set of parameters, set of parameters associated, fpr, tpr.
    """
    
    # hard-coded constant !
    n_metrics = 7 # accuracy, f1 score*2, recall*2, balanced_accuracy, auc
    n_infos = 3 # name, sampling, params
    n_cols = n_infos + n_metrics + 1 #name, metrics + std best metrics (auc)
    
    print("### optim metric: ", optim_metric_name)
    if optim_metric_name=="auc":
        index_optim_metric=6
    elif optim_metric_name=="balanced_accuracy":
        index_optim_metric=5
    else:
        raise Warning("Not the good optim metric")
        
    #metric to choos which model (and his set of parameters) is the best, here auc score 
      
    optim_metric = 0 
    index_best = 0
    
    #initialization
    name_family = str(dict_mod['clf'][0])   
    param_grid = ParameterGrid(dict_mod)
    n_grid = len(param_grid)
    metrics = np.zeros((n_grid, n_cols), dtype='object')

    #loop for each parameters set of the grid
    for i in range(n_grid):
        params = param_grid[i]
        print("processing (model, n_grid):", (name_family,i) )
        sampling = str(params['sampling']) 
        params_clf = [key[5:] + ": " + str(param_value) for (key, param_value) in params.items() if key.startswith('clf__')] #to print parameters in the df
        params_clf = ', '.join(params_clf)
        metrics_loop = np.zeros((loop_evaluation, n_metrics))
        
        # loop evaluation of the same model associated with parameters to have metrics mean and std of auc
        for k in range(loop_evaluation):
            model = pipeline.set_params(**params) 
            cv_results = LOO_cross_validation(X, y, remove_nosuspicion, valid_exclude_nosuspicion, mask_nosuspicion, model, list_patients_int, specificity_threshold=specificity_threshold, additional_comparison=additional_comparison, bool_plot=False)
            metrics_loop[k,:] = cv_results["accuracy"], cv_results["f1_score_positive"], cv_results["f1_score_negative"], cv_results["sensitivity"], cv_results["specificity"], cv_results["balanced_accuracy"], cv_results["auc"]
        
        # models names    
        metrics[i, 0:3] = name_family, sampling, params_clf
        metrics_mean = np.mean(metrics_loop, axis=0)
        # models metrics means and auc std
        metrics[i, 3:-1] =  np.round(metrics_mean, 2)
        std = np.std(metrics_loop[:,index_optim_metric])
        metrics[i, -1] = np.round(std, 2)
        
        #update of best model
        if optim_metric < metrics_mean[index_optim_metric]:
            index_best = i
            optim_metric = metrics_mean[index_optim_metric]
            params_best = params
            patients_misclassified = cv_results["patients_misclassified"]  #becareful just the last one, not for all evaluation of  loops

                
    #definition of the dictionnary for the the best model            
    dict_best = {"index": index_best} 
    dict_best["patients_misclassified"] = patients_misclassified
    dict_best["params"] = params_best

        
        
    return(metrics, dict_best)        
    


def gridsearch_models(X, y, remove_nosuspicion, valid_exclude_nosuspicion, mask_nosuspicion, dict_models, list_patients_int, loop_evaluation, optim_metric_name="auc", specificity_threshold=0.7, additional_comparison=False, plot_models=True):
    """
    
    Compute gridsearch for each models family (LR, SVC, RF) and give as output dataframe with metrics associated with the best auc score of the gridsearch for each models family

    Parameters
    ----------
    X : array shape (n_patients, n_features)
        Radiomics features array
    y : array int
        0 negative EI, 1 positive EI    
    remove_nosuspicion: bool, params of function LOO_cross_validation
        remove nosuspicion patients of the study or not
    valid_exclude_nosuspicion: bool, params of function LOO_cross_validation
        exclude nosuspicion patients of validation
    mask_nosuspicion: array of bool, params of function LOO_cross_validation
        True: patients not from SVD database and nosuspicion no suspicion patients 
    dict_models : list of dict  
        List of dictionnary for each model family. example: [LR_dict, RF_dict]
        dictionnary ares input of Parametergrid to set parameters of pipelines. example: LR_dict = {'sampling': sampling_opt, 'clf': [LogisticRegression()], 'clf__C': [1, 0.1, 0.01]]}
    list_patients_int : list of int, shape (n_patients)
        int correspondant to the patient name
    loop_evaluation : int, optional
        number of iterations for the evaluation of one model, after the metrics means are taken (std evaluate for auc). The default is 3.
    optim_metric_name: str
         metric to select hyperparameters:  auc, or balanced_accuracy
    specifity_threshold : float
        specificity threshold ex: 0.7 or 0.9
    additional_comparison : TYPE, optional
        DESCRIPTION. The default is False.
    plot_models :  bool, optional
        confusion matrix and tpr fpr for roc curves, only for one model evalluation not all loop_evaluation iterations. The default is False.

    Returns
    -------
    df_metrics: data frame, shape: n family model (usually LR, SVC, RF), n_infos + n_metrics + 1 = 10
        3 first columns: model name, sampling, parameters. then 6 columns: metrics means, last columns: std depends of index_optim_metric
        columns=['model', 'sampling', 'params', 'accuracy', "f1_score_positive", "f1_score_negative", 'sensitivity', 'specificity', 'balanced_accuracy', 'auc', 'std']
        

    """

    # hard-coded constant
    pipeline = Pipeline([('sampling', 'passthrough'), ('clf', SVC()) ]) 
    n_metrics = 7 # accuracy, f1 score*2, recall*2, , balanced_accuracy, auc
    n_infos = 3 # name, sampling, params
    n_cols = n_infos + n_metrics + 1 #name, metrics + std best metrics (auc)
    
    #initialization
    n_models = len(dict_models)
    metrics = np.zeros((n_models, n_cols), dtype='object')
    
    
    #loop to do gridsearch for each models family
    for k in range(n_models):
        dict_mod = dict_models[k] #dictionnary associated to one family (LR, SVC or RF)
        print("processing grid:", dict_mod['clf'])
        # grid search for this family
        results, dict_best = gridsearch_loo(X, y, remove_nosuspicion, valid_exclude_nosuspicion, mask_nosuspicion, pipeline, dict_mod, list_patients_int, loop_evaluation, optim_metric_name, specificity_threshold, additional_comparison=additional_comparison)
        metrics[k,:] =  results[dict_best['index'],:] # metrics correspondant to the model and the parameters set with the highest auc score
        #additionnal comparison for roc curves

            
    df_metrics = DataFrame(metrics, columns=['model', 'sampling', 'params', 'accuracy', "f1_score_positive", "f1_score_negative", 'sensitivity', 'specificity', 'balanced_accuracy', 'auc',  'std'])    
          
    if plot_models:
        positions = range(n_models)
        
        # Accuracy
        fig = plt.figure()
        fig.suptitle('Models Comparison - Accuracy' )
        ax = fig.add_subplot(111)
        ax.plot(positions, metrics[:,3], '_', c="r", mew=3, ms=50)
        ax.set_xticks(positions)
        ax.set_xticklabels(metrics[:,0])
        plt.xticks(rotation=70)
        plt.subplots_adjust(top=0.960, bottom=0.225)
        plt.xlabel("Models")
        plt.ylabel("Accuracy")
        plt.ylim([0.6, 0.85])
        plt.show()
        
        #AUC
        fig = plt.figure()
        fig.suptitle('Models Comparison - AUC' )
        ax = fig.add_subplot(111)
        ax.plot(positions, metrics[:,9], '_', c="r", mew=3, ms=50)
        ax.set_xticks(positions)
        ax.set_xticklabels(metrics[:,0])
        plt.xticks(rotation=70)
        plt.subplots_adjust(top=0.960, bottom=0.225)
        plt.xlabel("Models")
        plt.ylabel("AUC")
        plt.ylim([0.65, 0.9])
        plt.show()
        
        # Comparison f1_score_positive, f1_score_negative
        fig = plt.figure()
        fig.suptitle('Models Comparison' )
        ax = fig.add_subplot(111)
        ax.plot(positions, metrics[:,4], '_', c="r", mew=3, ms=30, label="f1 score positive")
        ax.plot(positions, metrics[:,5], '_', c="b", mew=3, ms=30, label="f1 score negative")
        ax.set_xticks(positions)
        ax.set_xticklabels(metrics[:,0])
        plt.xticks(rotation=70)
        plt.subplots_adjust(top=0.960, bottom=0.2)
        plt.xlabel("Models")
        plt.ylabel("metric")
        plt.legend(loc="best")
        plt.ylim([0.6, 0.85])
        plt.show()
        
        # Comparison Specificity, Sensitivity
        fig = plt.figure()
        fig.suptitle('Models Comparison' )
        ax = fig.add_subplot(111)
        ax.plot(positions, metrics[:,6], '_', c="r", mew=3, ms=30, label="sensitivity")
        ax.plot(positions, metrics[:,7], '_', c="b", mew=3, ms=30, label="specificitty")
        ax.set_xticks(positions)
        plt.subplots_adjust(top=0.960, bottom=0.2)
        ax.set_xticklabels(metrics[:,0])
        plt.xticks(rotation=70)
        plt.xlabel("Models")
        plt.ylabel("metric")
        plt.legend(loc="best")
        plt.ylim([0.6, 0.85])
        plt.show()

        # balanced_accuracy
        fig = plt.figure()
        fig.suptitle('Models Comparison - Balanced Accuracy' )
        ax = fig.add_subplot(111)
        ax.plot(positions, metrics[:,8], '_', c="r", mew=3, ms=50)
        ax.set_xticks(positions)
        ax.set_xticklabels(metrics[:,0])
        plt.xticks(rotation=70)
        plt.subplots_adjust(top=0.960, bottom=0.225)
        plt.xlabel("Models")
        plt.ylabel("Balanced Accuracy")
        plt.ylim([0.6, 0.85])
        plt.show()        
        
        
       
    return(df_metrics)



def models_analysis(X, y, remove_nosuspicion, valid_exclude_nosuspicion, mask_nosuspicion, dict_models, list_patients_int, n_loop=5, specificity_threshold=0.7, plot_models=True):
    """
    Evaluate (n_loop iteration) each models associated to parameters in dict_models (usually only one parameter set for each models family (SC, RF...))
    Give as output dataframe with metrics means on iterations and std of the auc score. PLot also Metrics Box plot. 

    Parameters
    ----------
    X : array shape (n_patients, n_features)
        Radiomics features array
    y : array int
        0 negative EI, 1 positive EI    
    remove_nosuspicion: bool, params of function LOO_cross_validation
        remove nosuspicion patients of the study or not
    valid_exclude_nosuspicion: bool, params of function LOO_cross_validation
        exclude nosuspicion patients of validation
    mask_nosuspicion: array of bool, params of function LOO_cross_validation
        True: patients not from SVD database and others nosuspicion patients
    dict_models : list of dict  
        List of dictionnary for each model family with one set of parameters. example: [LR_dict, RF_dict]
        Only one set of parameter in each dictionnary because no Gridsearch is applied !!!
    list_patients_int: list of int, shape (n_patients)
        int correspondant to the patient name
    n_loop :  int, optional
        number of iterations for the evaluation of one model, after the metrics means are taken (std evaluate for auc). The default is 5.
    specifity_threshold : float
        specificity threshold ex: 0.7 or 0.9
    plot_models : bool, optional
        Boxplot of metrics for the n_loop iterations. The default is True.

    Returns
    -------
    df_metrics: data frame, shape: n family model (usually LR, SVC, RF), n_infos + n_metrics + 1 = 9
        3 first columns: model name, sampling, parameters. then 6 columns: metrics means, last columns: std depends of index_optim_metric
        columns=['model', 'sampling', 'params', 'accuracy', "f1_score_positive", "f1_score_negative", 'sensitivity', 'specificity', 'balanced_accuracy', 'auc', 'std']
    """
    
    #hard-coded constants
    pipeline = Pipeline([('sampling', 'passthrough'), ('clf', SVC()) ])  
    n_metrics = 7 # accuracy, f1 score*2, recall*2, balanced_accuracy, auc
    n_infos = 3 # name, sampling, params
    n_cols = n_infos + n_metrics + 1 #name, metrics + std best metrics (auc)
    
    
    #initialization
    n_models = len(dict_models)
    mean_metrics = np.zeros((n_models, n_cols), dtype='object')
    metrics = np.zeros((n_models, n_loop, n_metrics))
    patients_misclassified = []
    
    # loop to evaluate each models (initially differents family specific models)
    for k in range(n_models):
        dict_mod = dict_models[k]
        print("processing grid:", dict_mod['clf'])
        name_family = str(dict_mod['clf'][0])   
        param_grid = ParameterGrid(dict_mod) # normally grid 1*1, only one set of parameters!
        params = param_grid[0] #only the first parameters set (noramlly only one!)
        sampling = str(params['sampling'])
        params_clf = [key[5:] + ": " + str(param_value) for (key, param_value) in params.items() if key.startswith('clf__')] #to print parameters in the df
        params_clf = ', '.join(params_clf)
        model = pipeline.set_params(**params) 
        patients_misclassified = [[]]
        # evalutation to compute metrics mean and std of auc
        for l in range(n_loop):
            print(l)
            cv_results = LOO_cross_validation(X, y, remove_nosuspicion, valid_exclude_nosuspicion, mask_nosuspicion, model, list_patients_int, specificity_threshold=specificity_threshold, additional_comparison=False, bool_plot=False)
            metrics[k,l,:] = cv_results["accuracy"], cv_results["f1_score_positive"], cv_results["f1_score_negative"], cv_results["sensitivity"], cv_results["specificity"], cv_results["balanced_accuracy"], cv_results["auc"]
            patients_misclassified[-1].append(cv_results["patients_misclassified"])
            
        #means and std on iterations
        mean_metrics[k, 0:n_infos] =  name_family, sampling, params_clf
        mean_loop = np.mean(metrics[k,:,:], axis=0)
        std_main_auc = np.std(metrics[k, :, -1])
        mean_metrics[k, n_infos:-1] = np.round(mean_loop, 2)
        mean_metrics[k, -1] = np.round(std_main_auc, 2)
       
    df_metrics = DataFrame(mean_metrics, columns=['model', 'sampling', 'params', 'accuracy', "f1_score_positive", "f1_score_negative", 'sensitivity', 'specificity', 'balanced_accuracy', 'auc', 'std auc'])    
          
    if plot_models:
        positions = range(n_models)
        
        # Acc
        fig = plt.figure()
        fig.suptitle('Models Comparison - Accuracy' )
        ax = fig.add_subplot(111)
        bp0 = ax.boxplot(metrics[:,:,0].T, positions=positions, patch_artist=True, showfliers=False)
        plt.setp(bp0['boxes'], color='darkorange', facecolor='moccasin')
        plt.setp(bp0['medians'], color='darkorange', linewidth=2)
        plt.setp(bp0['whiskers'], linestyle=':')
        plt.setp(bp0['caps'], color='darkorange')
        ax.set_xticklabels(mean_metrics[:,0])
        plt.legend([bp0["boxes"][0]], ["Accuracy"])
        plt.xticks(rotation=70)
        plt.subplots_adjust(top=0.960, bottom=0.45)
        plt.xlabel("Models")
        plt.ylabel("Accuracy")
        plt.ylim([0.6, 0.85])
        plt.show()
            
        
        # AUC
        fig = plt.figure()
        fig.suptitle('Models Comparison - AUC' )
        ax = fig.add_subplot(111)
        bp0 = ax.boxplot(metrics[:,:,6].T, positions=positions, patch_artist=True, showfliers=False)
        plt.setp(bp0['boxes'], color='darkorange', facecolor='moccasin')
        plt.setp(bp0['medians'], color='darkorange', linewidth=2)
        plt.setp(bp0['whiskers'], linestyle=':')
        plt.setp(bp0['caps'], color='darkorange')
        ax.set_xticklabels(mean_metrics[:,0])
        plt.legend([bp0["boxes"][0]], ["AUC"])
        plt.xticks(rotation=70)
        plt.subplots_adjust(top=0.960, bottom=0.45)
        plt.xlabel("Models")
        plt.ylabel("AUC")
        plt.ylim([0.65, 0.9])
        plt.show()
            
        # Comparison f1_score_positive, f1_score_negative
        fig = plt.figure()
        fig.suptitle('Models Comparison - f1score' )
        ax = fig.add_subplot(111)   
        bp1 = ax.boxplot(metrics[:,:,1].T, positions=positions, patch_artist=True, showfliers=False)
        bp2 = ax.boxplot(metrics[:,:,2].T, positions=positions, patch_artist=True, showfliers=False)
        plt.setp(bp1['whiskers'], linestyle=':')
        plt.setp(bp1['boxes'], color='darkred', facecolor='lightcoral')
        plt.setp(bp1['medians'], color='darkred', linewidth=2)
        plt.setp(bp1['caps'], color='darkred')
        plt.setp(bp2['whiskers'], linestyle=':')
        plt.setp(bp2['boxes'], color='royalblue', facecolor='lightsteelblue')
        plt.setp(bp2['medians'], color='royalblue', linewidth=2)
        plt.setp(bp2['caps'], color='darkblue')
        ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['f1score positive', 'f1score negative'], loc='best')
        ax.set_xticks(positions)
        ax.set_xticklabels(mean_metrics[:,0])
        plt.xticks(rotation=70)
        plt.subplots_adjust(top=0.960, bottom=0.45)
        plt.xlabel("Models")
        plt.ylabel("metric")
        plt.ylim([0.6, 0.85])
        plt.show()
        
        # Comparison sensitivity specificity
        fig = plt.figure()
        fig.suptitle('Models Comparison- recall' )
        ax = fig.add_subplot(111)
        bp1 = ax.boxplot(metrics[:,:,3].T, positions=positions, patch_artist=True, showfliers=False)
        bp2 = ax.boxplot(metrics[:,:,4].T, positions=positions, patch_artist=True, showfliers=False)
        plt.setp(bp1['whiskers'], linestyle=':')
        plt.setp(bp1['boxes'], color='darkred', facecolor='lightcoral')
        plt.setp(bp1['medians'], color='darkred', linewidth=2)
        plt.setp(bp2['whiskers'], linestyle=':')
        plt.setp(bp2['boxes'], color='royalblue', facecolor='lightsteelblue')
        plt.setp(bp2['medians'], color='royalblue', linewidth=2)
        plt.setp(bp2['caps'], color='darkblue')
        ax.legend([bp1["boxes"][0], bp2["boxes"][0]],  ['sensitivity', 'specificity'], loc='best')
        ax.set_xticks(positions)
        ax.set_xticklabels(mean_metrics[:,0])
        plt.xticks(rotation=70)
        plt.subplots_adjust(top=0.960, bottom=0.45)
        plt.xlabel("Models")
        plt.ylabel("metric")
        plt.ylim([0.6, 0.85])
        plt.ylim([0.6, 0.85])
        plt.show()
        
        # Balanced_accuracy
        fig = plt.figure()
        fig.suptitle('Models Comparison - Balanced_accuracy' )
        ax = fig.add_subplot(111)
        bp0 = ax.boxplot(metrics[:,:,5].T, positions=positions, patch_artist=True, showfliers=False)
        plt.setp(bp0['boxes'], color='darkorange', facecolor='moccasin')
        plt.setp(bp0['medians'], color='darkorange', linewidth=2)
        plt.setp(bp0['whiskers'], linestyle=':')
        plt.setp(bp0['caps'], color='darkorange')
        ax.set_xticklabels(mean_metrics[:,0])
        plt.legend([bp0["boxes"][0]], ["AUC"])
        plt.xticks(rotation=70)
        plt.subplots_adjust(top=0.960, bottom=0.45)
        plt.xlabel("Models")
        plt.ylabel("Balanced Accuracy")
        plt.ylim([0.6, 0.85])
        plt.show()
        
       
    return(df_metrics, metrics)


        
def feature_importance(X, y, features_name, pipeline, n_loop=10):
    
    """
    
    Compute the mean importance for each features w.r.t to a model from LR family or RF family. LR: importance = weights coefficients, RF gini importance.
    Boxplot and barplot of features importance

    Parameters
    ----------
    X : array shape (n_patients, n_features)
        Radiomics features array
    y : array int
        0 negative EI, 1 positive EI    
    features_names list
        name of features
    pipeline : Pipeline sklearn or imblearn
        ML model of LR family or RF family, example: Pipeline([('sampling', ADASYN()), ('clf', LogisticRegression(C=0.01))])
    n_loop : int, optional
        compute importance for each n_loop iteration to get the mean. The default is 10.

    Returns
    -------
    importance_mean
    
    """
     
    n_patients, n_features = np.shape(X)
    pipeline = clone(pipeline) #Constructs a new unfitted estimator with the same parameters.
    importance = np.zeros((n_loop, n_features))#initialization
    
    for k in range(n_loop):
        #importance for LR family
        if str(pipeline['clf']).startswith('LogisticRegression'):
            pipeline_fit = pipeline
            pipeline_fit.fit(X, y)
            model = pipeline_fit['clf']
            importance[k, :] = model.coef_[0]           
        #importance for RF family
        if str(pipeline['clf']).startswith('RandomForestClassifier'):
            pipeline.fit(X, y)
            model = pipeline['clf']
            importance[k, :] = model.feature_importances_
    
    positions = range(n_features)
    importance_abs = np.abs(importance)
    importance_mean = np.mean(importance_abs, axis=0)
    
    #Boxplot importance
    fig = plt.figure()
    ax = fig.add_subplot(111)
    bp = ax.boxplot(importance, positions=positions, showfliers=False)
    plt.setp(bp['medians'], linewidth=2)
    ax.set_xticks(positions)
    ax.set_xticklabels(features_name)
    plt.axhline(0, color='black',linestyle=':', lw=0.5)
    plt.xticks(rotation=90)
    plt.subplots_adjust(top=0.960, bottom=0.4)
    plt.title("features importance")
    plt.show()
           
    #Bar mean importance     
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(positions, importance_mean)
    ax.set_xticks(positions)
    ax.set_xticklabels(features_name)
    plt.xticks(rotation=90)
    plt.subplots_adjust(top=0.960, bottom=0.4)
    plt.title("features importance")
    plt.show()

    return(importance_mean)    
        

def  outcome_analysis(X, y, features_name, alternative="greater"):
    """
    plot pvalue MWW u test postive versus negative population

    Parameters
    ----------
    X : array shape (n_patients, n_features)
        Radiomics features array
    y : array int
        0 negative EI, 1 positive EI    
    features_names list
        name of features
    alternative: see mannwhitenyu function and alternative parameter

    Returns
    -------
    None.

    """
    #initialization
    n_features = np.shape(X)[1]
    vect_stat = np.zeros(n_features)
    vect_pvalue = np.zeros(n_features)
    #loop on features
    for k in range(n_features):
        feature = X[:,k]
        (stat, pvalue) = mannwhitneyu(feature[y==0], feature[y==1], alternative=alternative)
        vect_stat[k] = stat
        vect_pvalue[k] = pvalue


    positions = np.array(range(n_features))
    
    #pvalue MWW u test postive versus negative population
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(positions, vect_pvalue, 'k--')
    ax.plot(positions[vect_pvalue>0.05], vect_pvalue[vect_pvalue>.05], 'None', marker='o', markerfacecolor='b', label="no significance 95%")
    ax.plot(positions[vect_pvalue<0.05], vect_pvalue[vect_pvalue<0.05], 'None', marker='o', markerfacecolor='r', label="significance 95%")
    ax.set_xticks(positions)
    ax.set_xticklabels(features_name)
    plt.xticks(rotation=90)
    plt.subplots_adjust(top=0.960, bottom=0.40)
    plt.ylabel("pvalue")
    plt.legend(loc='best')
    plt.title("pvalue MWW u test, positive population versus negative population")
    plt.yscale('log')
    plt.show()    

    return vect_stat, vect_pvalue


    
def  feature_correlation(X, index, treshold, features_name):
    '''
    plot spearman correlation with others features
    
    Parameters
    ----------
    X : arr features
    index : index feature to analyze, int
    treshold : treshold for the -- line, float
    features_name : list, features name
        
    
    Returns
    -------
    None.

    '''
    #initialization
    n_features = np.shape(X)[1]
    vect_stat = np.zeros(n_features)
    vect_pvalue = np.zeros(n_features)
    
    #spearman correlatin and p value
    (arr_stat, arr_pvalue) = spearmanr(X)
    stat = np.abs(arr_stat[index,:])
    pvalue = arr_pvalue[index,:]
    feature_name = features_name[index]

    #plot
    positions = np.array(range(n_features))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot([positions[0]-1, positions[-1]+1], [treshold, treshold], 'k--', label="treshold: "+str(treshold))
    ax.plot(positions[pvalue>0.05], stat[pvalue>.05], 'None', marker='o', markerfacecolor='b', label="no significance 95%")
    ax.plot(positions[pvalue<0.05], stat[pvalue<0.05], 'None', marker='o', markerfacecolor='r', label="significance 95%")
    ax.set_xticks(positions)
    ax.set_xticklabels(features_name)
    plt.xticks(rotation=90)
    plt.subplots_adjust(top=0.960, bottom=0.40)
    plt.ylabel("correlation")
    plt.legend(loc='best')
    plt.title("absolute spearman correlation")
    #plt.yscale('log')
    plt.show()    
    



def test_analysis_hold_out(X, y, X_test, y_test,  list_patients_test, remove_nosuspicion, mask_nosuspicion, dict_models, n_loop=5, specificity_threshold=0.7, plot_models=True):
    """
    Hold out test set, train models on the train set, then Evaluate on the test set. (n_loop iteration).
    Each models is associated to parameters in dict_models (usually only one parameter set for each models family (SC, RF...))
    Give as output dataframe with metrics means on iterations and std of the auc score. PLot also Metrics Box plot. 

    Parameters
    ----------
    X : array shape (n_patients, n_features)
        Radiomics features array
    y : array int
        0 negative EI, 1 positive EI    
    X_test : array shape (n_patients, n_features)
        Radiomics features array
    y_test : array int
        0 negative EI, 1 positive EI  
    list_patients_test: array
        if of patients
    remove_nosuspicion: bool, params of function LOO_cross_validation
        remove nosuspicion patients of the study or not

    mask_nosuspicion: array of bool, params of function LOO_cross_validation
        True: patients not from SVD database and others nosuspicion patients   
    dict_models : list of dict  
        List of dictionnary for each model family with one set of parameters. example: [LR_dict, RF_dict]
        Only one set of parameter in each dictionnary because no Gridsearch is applied !!!
    n_loop :  int, optional
        number of iterations for the evaluation of one model, after the metrics means are taken (std evaluate for auc). The default is 5.
    specifity_threshold : float
        specificity threshold ex: 0.7 or 0.9
    plot_models : bool, optional
        Boxplot of metrics for the n_loop iterations. The default is True.

    Returns
    -------
    df_metrics: data frame, shape: n family model (usually LR, SVC, RF), n_infos + n_metrics + 1 = 10
        3 first columns: model name, sampling, parameters. then 6 columns: metrics means, last columns: std depends of index_optim_metric
        columns=['model', 'sampling', 'params', 'accuracy', "f1_score_positive", "f1_score_negative", 'sensitivity', 'specificity', 'balanced_accuracy', 'auc', 'std']
    """
    
    #hard-coded constants
    pipeline = Pipeline([('sampling', 'passthrough'), ('clf', SVC()) ])  
    n_metrics = 7 # accuracy, f1 score*2, recall*2, balanced accuracy, auc
    n_infos = 3 # name, sampling, params
    n_cols = n_infos + n_metrics + 1 #name, metrics + std best metrics (auc)
    n_test = len(y_test)

    # remove no suspicion, redundant with extraction from excel ?
    if remove_nosuspicion:
        X = X[mask_nosuspicion, :]
        y = y[mask_nosuspicion]
        mask_nosuspicion = mask_nosuspicion[mask_nosuspicion]


    #initialization
    n_models = len(dict_models)
    mean_metrics = np.zeros((n_models, n_cols), dtype='object')
    metrics = np.zeros((n_models, n_loop, n_metrics))
    metrics_sklearn = np.zeros((n_models, n_metrics-2)) # output proba and diag for each patients in order to study reclassement, just one loop !
    y_pred_models = np.zeros((n_test, n_models))  # output proba and diag for each patients in order to study reclassement
 
    
    # loop to evaluate each models (initially differents family specific models)
    for k in range(n_models):
        dict_mod = dict_models[k]
        print("processing grid:", dict_mod['clf'])
        name_family = str(dict_mod['clf'][0])   
        param_grid = ParameterGrid(dict_mod) # normally grid 1*1, only one set of parameters!
        params = param_grid[0] #only the first parameters set (noramlly only one!)
        sampling = str(params['sampling'])
        params_clf = [key[5:] + ": " + str(param_value) for (key, param_value) in params.items() if key.startswith('clf__')] #to print parameters in the df
        params_clf = ', '.join(params_clf)
        model = pipeline.set_params(**params) 


        # evalutation to compute metrics mean and std of auc
        for l in range(n_loop):
            print(l)

            # evalutation of the test database
            model = pipeline.set_params(**params) 
            model.fit(X,y)
            y_proba = model.predict_proba(X_test) #probabilities prediction
            
            #metrics computaiton
            y_proba = y_proba[:, 1]
            y_pred, accuracy, specificity, sensitivity, f1_score_positive, f1_score_negative, balanced_accuracy, auc = compute_metrics_option(y_test, y_proba, specificity_threshold, True) 

            metrics[k,l,:] = accuracy, f1_score_positive, f1_score_negative, sensitivity, specificity, balanced_accuracy, auc
            
            y_pred, accuracy, specificity, sensitivity, f1_score_positive, f1_score_negative, balanced_accuracy, auc = compute_metrics_sklearn(y_proba, y_test)
            metrics_sklearn[k,:]  =  accuracy, sensitivity, specificity, balanced_accuracy, auc
                            
                            
        y_pred_models[:, k] = np.array(y_proba>0.5, dtype='int').T # output proba and diag for each patients in order to study reclassement

        #means and std on iterations
        mean_metrics[k, 0:n_infos] =  name_family, sampling, params_clf
        mean_loop = np.mean(metrics[k,:,:], axis=0)
        std_auc = np.std(metrics[k, :, -1])
        mean_metrics[k, n_infos:-1] = np.round(mean_loop, 2)
        mean_metrics[k, -1] = np.round(std_auc,2)

    df_metrics = DataFrame(mean_metrics, columns=['model', 'sampling', 'params', 'accuracy', "f1_score_positive", "f1_score_negative", 'sensitivity', 'specificity', 'balanced_accuracy', 'auc', 'std auc'])    
    
    y_pred_models = np.column_stack((list_patients_test.T[:,None], y_pred_models))
    df_proba = DataFrame(y_pred_models, columns=["patients",  "LR ridge"]) #, "LR lasso", "SVC", "RF"])
    metrics_sklearn = np.column_stack((mean_metrics[:, 0][:,None], metrics_sklearn))
    df_metrics_sklearn = DataFrame(metrics_sklearn, columns=["model", "accuracy", "sensitivity", "specificity", "balanced_accuracy", "auc"])
        
    if plot_models:
        positions = range(n_models)

        # Acc
        fig = plt.figure()
        fig.suptitle('Models Comparison - Accuracy' )
        ax = fig.add_subplot(111)
        bp0 = ax.boxplot(metrics[:,:,0].T, positions=positions, patch_artist=True, showfliers=False)
        plt.setp(bp0['boxes'], color='darkorange', facecolor='moccasin')
        plt.setp(bp0['medians'], color='darkorange', linewidth=2)
        plt.setp(bp0['whiskers'], linestyle=':')
        plt.setp(bp0['caps'], color='darkorange')
        ax.set_xticklabels(mean_metrics[:,0])
        plt.legend([bp0["boxes"][0]], ["Accuracy"])
        plt.xticks(rotation=70)
        plt.subplots_adjust(top=0.960, bottom=0.45)
        plt.xlabel("Models")
        plt.ylabel("Accuracy")
        plt.ylim([0.6, 0.85])
        plt.show()


        # AUC
        fig = plt.figure()
        fig.suptitle('Models Comparison - AUC' )
        ax = fig.add_subplot(111)
        bp0 = ax.boxplot(metrics[:,:,6].T, positions=positions, patch_artist=True, showfliers=False)
        plt.setp(bp0['boxes'], color='darkorange', facecolor='moccasin')
        plt.setp(bp0['medians'], color='darkorange', linewidth=2)
        plt.setp(bp0['whiskers'], linestyle=':')
        plt.setp(bp0['caps'], color='darkorange')
        ax.set_xticklabels(mean_metrics[:,0])
        plt.legend([bp0["boxes"][0]], ["AUC"])
        plt.xticks(rotation=70)
        plt.subplots_adjust(top=0.960, bottom=0.45)
        plt.xlabel("Models")
        plt.ylabel("AUC")
        plt.ylim([0.65, 0.9])
        plt.show()

        # Comparison f1_score_positive, f1_score_negative
        fig = plt.figure()
        fig.suptitle('Models Comparison - f1score' )
        ax = fig.add_subplot(111)   
        bp1 = ax.boxplot(metrics[:,:,1].T, positions=positions, patch_artist=True, showfliers=False)
        bp2 = ax.boxplot(metrics[:,:,2].T, positions=positions, patch_artist=True, showfliers=False)
        plt.setp(bp1['whiskers'], linestyle=':')
        plt.setp(bp1['boxes'], color='darkred', facecolor='lightcoral')
        plt.setp(bp1['medians'], color='darkred', linewidth=2)
        plt.setp(bp1['caps'], color='darkred')
        plt.setp(bp2['whiskers'], linestyle=':')
        plt.setp(bp2['boxes'], color='royalblue', facecolor='lightsteelblue')
        plt.setp(bp2['medians'], color='royalblue', linewidth=2)
        plt.setp(bp2['caps'], color='darkblue')
        ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['f1score positive', 'f1score negative'], loc='best')
        ax.set_xticks(positions)
        ax.set_xticklabels(mean_metrics[:,0])
        plt.xticks(rotation=70)
        plt.subplots_adjust(top=0.960, bottom=0.45)
        plt.xlabel("Models")
        plt.ylabel("metric")
        plt.ylim([0.6, 0.85])
        plt.show()

        # Comparison sensitivity specificity
        fig = plt.figure()
        fig.suptitle('Models Comparison - recall' )
        ax = fig.add_subplot(111)
        bp1 = ax.boxplot(metrics[:,:,3].T, positions=positions, patch_artist=True, showfliers=False)
        bp2 = ax.boxplot(metrics[:,:,4].T, positions=positions, patch_artist=True, showfliers=False)
        plt.setp(bp1['whiskers'], linestyle=':')
        plt.setp(bp1['boxes'], color='darkred', facecolor='lightcoral')
        plt.setp(bp1['medians'], color='darkred', linewidth=2)
        plt.setp(bp2['whiskers'], linestyle=':')
        plt.setp(bp2['boxes'], color='royalblue', facecolor='lightsteelblue')
        plt.setp(bp2['medians'], color='royalblue', linewidth=2)
        plt.setp(bp2['caps'], color='darkblue')
        ax.legend([bp1["boxes"][0], bp2["boxes"][0]],  ['sensitivity', 'specificity'], loc='best')
        ax.set_xticks(positions)
        ax.set_xticklabels(mean_metrics[:,0])
        #plt.xticks(rotation=70)
        plt.subplots_adjust(top=0.960, bottom=0.45)
        #plt.xlabel("Models")
        plt.ylabel("metric")
        plt.ylim([0.5,1]) #[0.6, 0.85])
        plt.show()

        # Balanced Accuracy
        fig = plt.figure()
        fig.suptitle('Models Comparison - Balanced Accuracy' )
        ax = fig.add_subplot(111)
        bp0 = ax.boxplot(metrics[:,:,5].T, positions=positions, patch_artist=True, showfliers=False)
        plt.setp(bp0['boxes'], color='darkorange', facecolor='moccasin')
        plt.setp(bp0['medians'], color='darkorange', linewidth=2)
        plt.setp(bp0['whiskers'], linestyle=':')
        plt.setp(bp0['caps'], color='darkorange')
        ax.set_xticklabels(mean_metrics[:,0])
        plt.legend([bp0["boxes"][0]], ["Balanced Accuracy"])
        plt.xticks(rotation=70)
        plt.subplots_adjust(top=0.960, bottom=0.45)
        plt.xlabel("Models")
        plt.ylabel("Balanced Accuracy")
        plt.ylim([0.6, 0.85])
        plt.show()
        
    return(df_metrics, metrics, df_proba, df_metrics_sklearn)





def test_analysis_LOO(X, y, X_test, y_test,  remove_nosuspicion, mask_nosuspicion, dict_models, n_loop=5, index_optim_metric=6, specificity_threshold=0.7, plot_models=True):
    """
    LOO test evaluation.
    Evaluate (n_loop iteration) each models associated to parameters in dict_models (usually only one parameter set for each models family (SC, RF...))
    Give as output dataframe with metrics means on iterations and std of the auc score. PLot also Metrics Box plot. 
    
    Train on train set + Test set except the patient "leaved out". Prediction for each patients of the test set "leaved out". Then concatenation of probabilities and metrics computation. (n_loop iterations).

    Parameters
    ----------
    X : array shape (n_patients, n_features)
        Radiomics features array
    y : array int
        0 negative EI, 1 positive EI    
    X_test : array shape (n_patients, n_features)
        Radiomics features array
    y_test : array int
        0 negative EI, 1 positive EI  
    remove_nosuspicion: bool, params of function LOO_cross_validation
        remove nosuspicion patients of the study or not

    mask_nosuspicion: array of bool, params of function LOO_cross_validation
        True: patients not from SVD database and others nosuspicion patients   
    dict_models : list of dict  
        List of dictionnary for each model family with one set of parameters. example: [LR_dict, RF_dict]
        Only one set of parameter in each dictionnary because no Gridsearch is applied !!!
    n_loop :  int, optional
        number of iterations for the evaluation of one model, after the metrics means are taken (std evaluate for auc). The default is 5.
    index_optim_metric: int
        index of the metric to select hyperparameters: 6 optim based on auc, 5 optim based on balanced_accuracy
    specifity_threshold : float
        specificity threshold ex: 0.7 or 0.9
    plot_models : bool, optional
        Boxplot of metrics for the n_loop iterations. The default is True.

    Returns
    -------
    df_metrics: data frame, shape: n family model (usually LR, SVC, RF), n_infos + n_metrics + 1 = 10
        3 first columns: model name, sampling, parameters. then 6 columns: metrics means, last columns: std depends of index_optim_metric
        columns=['model', 'sampling', 'params', 'accuracy', "f1_score_positive", "f1_score_negative", 'sensitivity', 'specificity', 'balanced_accuracy', 'auc', 'std']
    """
    
    #hard-coded constants
    pipeline = Pipeline([('sampling', 'passthrough'), ('clf', SVC()) ])  
    n_metrics = 7 # accuracy, f1 score*2, recall*2, balanced accuracy, auc
    n_infos = 3 # name, sampling, params
    n_cols = n_infos + n_metrics + 1 #name, metrics + std best metrics (auc)
    
    
    # remove no suspicion, redundant with extraction from excel ?
    if remove_nosuspicion:
        X = X[mask_nosuspicion, :]
        y = y[mask_nosuspicion]
        mask_nosuspicion = mask_nosuspicion[mask_nosuspicion]
    
    n_train, n_features = np.shape(X)
    n_test, n_features = np.shape(X_test)
    X = np.concatenate((X, X_test))
    y = np.concatenate((y, y_test))
    mask_test = np.ones(n_train+n_test, dtype='bool')
    mask_test[0:n_train] = False
    list_patients_int = np.arange(0, n_train+n_test, 1, 'int')
    
    #initialization
    n_models = len(dict_models)
    mean_metrics = np.zeros((n_models, n_cols), dtype='object')
    metrics = np.zeros((n_models, n_loop, n_metrics))

    
    # loop to evaluate each models (initially differents family specific models)
    for k in range(n_models):
        dict_mod = dict_models[k]
        print("processing grid:", dict_mod['clf'])
        name_family = str(dict_mod['clf'][0])   
        param_grid = ParameterGrid(dict_mod) # normally grid 1*1, only one set of parameters!
        params = param_grid[0] #only the first parameters set (noramlly only one!)
        sampling = str(params['sampling'])
        params_clf = [key[5:] + ": " + str(param_value) for (key, param_value) in params.items() if key.startswith('clf__')] #to print parameters in the df
        params_clf = ', '.join(params_clf)
        model = pipeline.set_params(**params) 

        
        # evalutation to compute metrics mean and std of auc
        for l in range(n_loop):
            print(l)
            
            
            cv_results = LOO_cross_validation(X, y, remove_mask=False, valid_exclude_patients=True, mask_exclude_patients=mask_test, model=model, list_patients_int=list_patients_int, specificity_threshold=specificity_threshold, additional_comparison=False, bool_plot=False)
            metrics[k,l,:] = cv_results["accuracy"], cv_results["f1_score_positive"], cv_results["f1_score_negative"], cv_results["sensitivity"], cv_results["specificity"], cv_results["balanced_accuracy"], cv_results["auc"]


        #means and std on iterations
        mean_metrics[k, 0:n_infos] =  name_family, sampling, params_clf
        mean_loop = np.mean(metrics[k,:,:], axis=0)
        std_main_metric = np.std(metrics[k, :, index_optim_metric])
        mean_metrics[k, n_infos:-1] = np.round(mean_loop, 2)
        mean_metrics[k, -1] = np.round(std_main_metric,2)
       
    df_metrics = DataFrame(mean_metrics, columns=['model', 'sampling', 'params', 'accuracy', "f1_score_positive", "f1_score_negative", 'sensitivity', 'specificity', 'balanced_accuracy', 'auc', 'std'])    
          
    if plot_models:
        positions = range(n_models)
        
        # Acc
        fig = plt.figure()
        fig.suptitle('Models Comparison - Accuracy' )
        ax = fig.add_subplot(111)
        bp0 = ax.boxplot(metrics[:,:,0].T, positions=positions, patch_artist=True, showfliers=False)
        plt.setp(bp0['boxes'], color='darkorange', facecolor='moccasin')
        plt.setp(bp0['medians'], color='darkorange', linewidth=2)
        plt.setp(bp0['whiskers'], linestyle=':')
        plt.setp(bp0['caps'], color='darkorange')
        ax.set_xticklabels(mean_metrics[:,0])
        plt.legend([bp0["boxes"][0]], ["Accuracy"])
        plt.xticks(rotation=70)
        plt.subplots_adjust(top=0.960, bottom=0.45)
        plt.xlabel("Models")
        plt.ylabel("Accuracy")
        plt.ylim([0.6, 0.85])
        plt.show()
            
        
        # AUC
        fig = plt.figure()
        fig.suptitle('Models Comparison - AUC' )
        ax = fig.add_subplot(111)
        bp0 = ax.boxplot(metrics[:,:,6].T, positions=positions, patch_artist=True, showfliers=False)
        plt.setp(bp0['boxes'], color='darkorange', facecolor='moccasin')
        plt.setp(bp0['medians'], color='darkorange', linewidth=2)
        plt.setp(bp0['whiskers'], linestyle=':')
        plt.setp(bp0['caps'], color='darkorange')
        ax.set_xticklabels(mean_metrics[:,0])
        plt.legend([bp0["boxes"][0]], ["AUC"])
        plt.xticks(rotation=70)
        plt.subplots_adjust(top=0.960, bottom=0.45)
        plt.xlabel("Models")
        plt.ylabel("AUC")
        plt.ylim([0.65, 0.9])
        plt.show()
            
        # Comparison f1_score_positive, f1_score_negative
        fig = plt.figure()
        fig.suptitle('Models Comparison - f1score' )
        ax = fig.add_subplot(111)   
        bp1 = ax.boxplot(metrics[:,:,1].T, positions=positions, patch_artist=True, showfliers=False)
        bp2 = ax.boxplot(metrics[:,:,2].T, positions=positions, patch_artist=True, showfliers=False)
        plt.setp(bp1['whiskers'], linestyle=':')
        plt.setp(bp1['boxes'], color='darkred', facecolor='lightcoral')
        plt.setp(bp1['medians'], color='darkred', linewidth=2)
        plt.setp(bp1['caps'], color='darkred')
        plt.setp(bp2['whiskers'], linestyle=':')
        plt.setp(bp2['boxes'], color='royalblue', facecolor='lightsteelblue')
        plt.setp(bp2['medians'], color='royalblue', linewidth=2)
        plt.setp(bp2['caps'], color='darkblue')
        ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['f1score positive', 'f1score negative'], loc='best')
        ax.set_xticks(positions)
        ax.set_xticklabels(mean_metrics[:,0])
        plt.xticks(rotation=70)
        plt.subplots_adjust(top=0.960, bottom=0.45)
        plt.xlabel("Models")
        plt.ylabel("metric")
        plt.ylim([0.6, 0.85])
        plt.show()
        
        # Comparison sensitivity specificity
        fig = plt.figure()
        fig.suptitle('Models Comparison- recall' )
        ax = fig.add_subplot(111)
        bp1 = ax.boxplot(metrics[:,:,3].T, positions=positions, patch_artist=True, showfliers=False)
        bp2 = ax.boxplot(metrics[:,:,4].T, positions=positions, patch_artist=True, showfliers=False)
        plt.setp(bp1['whiskers'], linestyle=':')
        plt.setp(bp1['boxes'], color='darkred', facecolor='lightcoral')
        plt.setp(bp1['medians'], color='darkred', linewidth=2)
        plt.setp(bp2['whiskers'], linestyle=':')
        plt.setp(bp2['boxes'], color='royalblue', facecolor='lightsteelblue')
        plt.setp(bp2['medians'], color='royalblue', linewidth=2)
        plt.setp(bp2['caps'], color='darkblue')
        ax.legend([bp1["boxes"][0], bp2["boxes"][0]],  ['sensitivity', 'specificity'], loc='best')
        ax.set_xticks(positions)
        ax.set_xticklabels(mean_metrics[:,0])
        plt.xticks(rotation=70)
        plt.subplots_adjust(top=0.960, bottom=0.45)
        plt.xlabel("Models")
        plt.ylabel("metric")
        plt.ylim([0.6, 0.85])
        plt.show()
        
        # Balanced Accuracy
        fig = plt.figure()
        fig.suptitle('Models Comparison - Balanced Accuracy' )
        ax = fig.add_subplot(111)
        bp0 = ax.boxplot(metrics[:,:,5].T, positions=positions, patch_artist=True, showfliers=False)
        plt.setp(bp0['boxes'], color='darkorange', facecolor='moccasin')
        plt.setp(bp0['medians'], color='darkorange', linewidth=2)
        plt.setp(bp0['whiskers'], linestyle=':')
        plt.setp(bp0['caps'], color='darkorange')
        ax.set_xticklabels(mean_metrics[:,0])
        plt.legend([bp0["boxes"][0]], ["Balanced Accuracy"])
        plt.xticks(rotation=70)
        plt.subplots_adjust(top=0.960, bottom=0.45)
        plt.xlabel("Models")
        plt.ylabel("Balanced Accuracy")
        plt.ylim([0.6, 0.85])
        plt.show()
        
       
    return(df_metrics, metrics)



if __name__ == "__main__":
    
    a=1
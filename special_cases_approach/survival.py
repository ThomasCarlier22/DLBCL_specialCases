# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 15:17:35 2022

@author: gafrecon
"""



import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines.utils.sklearn_adapter import sklearn_adapter
from lifelines.utils import k_fold_cross_validation
from sklearn.model_selection import GridSearchCV, train_test_split
from sksurv.compare import compare_survival
from sksurv.util import Surv 
import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN
from scipy.stats import combine_pvalues, hmean, wilcoxon
import os
#from features_processing import features_processing
from statsmodels.stats.multitest import multipletests
from pandas import DataFrame 


def km_plot(survival_data, mask_group="", name="", label1="", label2=""):
    """
    - plot kaplan meier curve of the survival data of the whole data
    - plot kaplan meier curve of high and low risk (optional)

    Parameters
    ----------
    survival_data : array (n_patients, 2)
        censorship and survival time
    mask_group : boolean arr, optional
        mask to split patients in two groups (high and low risk). The default is "".
    name : str, optional
        name to add in plot title . The default is "".
    label1 : str, optional
        plot label . The default is "".
    label2 : str, optional
        plot label. The default is "".

    Returns
    -------
    None.

    """
    
    
    status = np.array(survival_data[:, 0], dtype=bool)
    survival = survival_data[:, 1]

    
    # KM survivakl curve for two group
    if mask_group!="":
        results = logrank_test(survival[mask_group], survival[~mask_group] , status[mask_group], status[~mask_group], alpha=.95)
        pvalue_logrank = (results.summary["p"].values).round(4)[0]
        test_statistic = (results.summary["test_statistic"].values).round(3)
        print('test stat: ', test_statistic)
        print('pvalue: ', pvalue_logrank)
        
        plt.figure()
        kmf = KaplanMeierFitter(label=label1)
        kmf.fit(durations=survival[mask_group], event_observed=status[mask_group])
        kmf.plot_survival_function()        
        kmf = KaplanMeierFitter(label=label2)
        kmf.fit(durations=survival[~mask_group], event_observed=status[~mask_group])
        kmf.plot_survival_function()  
        plt.annotate(f"logrank test, pvalue: {pvalue_logrank}", xy=(0.05, 0.05), xycoords='axes fraction')
              
        
        plt.ylabel("est. probability of survival ")
        plt.xlabel("time (in months)")
        plt.title("Kaplan-Meier: {}".format(name))
        plt.show()
             

    # KM survival curve    
    plt.figure()
    kmf = KaplanMeierFitter()
    kmf.fit(durations = survival, event_observed = status)
    kmf.plot_survival_function(at_risk_counts = True)
    plt.ylabel("est. probability of survival ")
    plt.xlabel("time (in months)")
    plt.title("Kaplan-Meier: {}".format(name))
    plt.show()
  
    



def survival_pipeline(survival_data, X, features_name, dict_params, test_size = 0.2, n_loop=2, correction_method="bonferroni", multivariate_significant=False, plot_survival=False):
    """
    survival pipeline 
    
    loop to highlight train/test variability:
        - split train/test
        - Cox univariate selection: significant features slected (a correction on pvalues can be applied)
        - Cox multivariate: CV to select penalization, training to have acces to features coefficients and pvalues. (ption to select only significant features) 
        - Regression score bi*Xi based on multivariate features coefficient
        - quartiles of regression score on the train set
        - logrank and kaplan meier curves of groups divided by regression score quartiles applied to train and test set

    Parameters
    ----------
    survival_data : array (n_patients, 2)
        censorship and survival time
    X : array shape (n_patients, n_features)
        features array
    features_name : list
        name of features
    dict_params : dict
        parameters grid of the penalized Cox Multivariate model
    test_size : float, optional
        proportion of the test set. The default is 0.2.
    n_loop : int, optional
        number of loop to highlight split variability. The default is 2.
    correction_method : str, optional
        name of the pvalue correction method in Cox univariate (multiple test correction). The default is "bonferroni".
    multivariate_significant : bool, optional
        wheter or not select only significant features in multivariate analysis. The default is False.
    plot_survival : bool, optional
        plot Regression score histogram, train and test kaplan meier curves. The default is False.

    Returns
    -------
    logrank_pvalues, logrank_stats: float arr, shape: (n_loop)
        logrank test of groups based on quartiles of train regression score applied to test set

    """

    
    # prepare data
    status = np.array(survival_data[:, 0], dtype=bool)
    survival = np.array(survival_data[:, 1], dtype=int)
    stratification = np.zeros_like(survival)
    n_patients = np.shape(status)[0]
    n_test_size = int(n_patients * test_size)
    
    data_array = np.column_stack((survival_data, X))
    columns_name =   ["status", "survival"] + list(features_name)
    df_data = pd.DataFrame(data_array, columns=columns_name)
    
    # stratification for the train/test split: 8 class with respect to status=1 or 0 and quartile of the censorship  
    # patient with events
    q_25, q_50, q_75 = np.percentile(survival[status], [25, 50, 75])
    mask_1 = survival[status] < q_25
    mask_2 = (survival[status] >= q_25) & (survival[status] < q_50)
    mask_3 = (survival[status] >= q_50) & (survival[status] < q_75)
    mask_4 = survival[status] >= q_75
    stratification[status] = 1*mask_1 + 2*mask_2 + 3*mask_3 + 4*mask_4
    
    # patient without events
    q_25, q_50, q_75 = np.percentile(survival[~status], [25, 50, 75])
    mask_1 = survival[~status] < q_25
    mask_2 = (survival[~status] >= q_25) & (survival[~status] < q_50)
    mask_3 = (survival[~status] >= q_50) & (survival[~status] < q_75)
    mask_4 = survival[~status] >= q_75
    stratification[~status] = 5*mask_1 + 6*mask_2 + 7*mask_3 + 8*mask_4
    
    # output initalization
    logrank_pvalues = np.zeros(n_loop)
    logrank_stats = np.zeros(n_loop)
    
    # train/test loop
    for k in range(n_loop):     
        print("loop: ", k)
        try:
            # split train/test
            df_data_train, df_data_test = train_test_split(df_data, test_size=n_test_size, shuffle=True, stratify=stratification, random_state=k)       
            survival_data_train = df_data_train.to_numpy("float")[:, :2]
            survival_data_test = df_data_test.to_numpy("float")[:, :2]
            
            # univariate analysis
            mask_univariate, coef_univariate, pvalue_univariate = survival_univariate(df_data_train, threshold_pvalue=0.05, correction_method=correction_method, plot_summary=plot_survival)
            coef_univariate = coef_univariate[mask_univariate]
            mask_univariate = np.hstack((np.array([True, True]), mask_univariate)) # select survival datan, and significant features in Cox univariate
            df_data_train = df_data_train.loc[:, mask_univariate]
            df_data_test = df_data_test.loc[:, mask_univariate] 
            
            # multivariate analysis
            mask_multivariate, coef_multivariate, pvalue_multivariate = survival_multivariate(df_data_train, dict_params, plot=plot_survival)
            
            # select significant features or not
            if multivariate_significant:
                coef_multivariate = coef_multivariate[mask_multivariate]
                mask_multivariate = np.hstack((np.array([False, False]), mask_multivariate)) # select significant features in Cox multivariate, not select survival data
                mask_RS = mask_multivariate
                coef_RS = coef_multivariate      
            else:
                mask_RS = np.ones(len(coef_univariate)+2, dtype=bool)
                mask_RS[0] = False
                mask_RS[1] = False
                coef_RS = coef_multivariate
         
            # Regression score: coef bi*Xi 
            df_data_train = df_data_train.loc[:, mask_RS]
            df_data_test = df_data_test.loc[:, mask_RS]
            features_name = df_data_train.columns
            X_train = df_data_train.to_numpy("float")
            X_test = df_data_test.to_numpy("float")
          
            regression_score_train = np.sum(X_train*coef_RS, axis=1) # RS train
            regression_score_test = np.sum(X_test*coef_RS, axis=1) # test set
            q_01, q_25, q_50, q_75, q_99 = np.percentile(regression_score_train, [1, 25, 50, 75, 99]) # quartiles of the train regression score
            
            # kaplan meier curves and logranktest, groups build on quartiles train regression score applied to train and test set
            survival_regression_score(survival_data_train, regression_score_train, q_01, q_25, q_50, q_75, q_99, title="train", plot=plot_survival)
            logrank_pvalue, test_statistic = survival_regression_score(survival_data_test, regression_score_test, q_01, q_25, q_50, q_75, q_99, title='test', plot=plot_survival)
            
            logrank_pvalues[k] = logrank_pvalue
            logrank_stats[k] = test_statistic
        
        except KeyError:
            pass
        except IndexError:
            pass
        except ValueError:
            pass
            
    return(logrank_pvalues, logrank_stats)    



def survival_univariate(df_data, threshold_pvalue=0.05, correction_method="", plot_summary=False): 
    """
    
    apply a cox univariate model to each feature in order to select significant features (a multiple test correction can be applied )
    
    Parameters
    ----------
    df_data : dataframe
        dataframe with survival data and features
    threshold_pvalue : float, optional
        pvalue theshold that can be corrected by a multiple test correction. The default is 0.05.
    correction_method : str, optional
        name of the multiple test correction. The default is "".
    plot_summary : bool, optional
        plot summary of the cox univariate analysis. The default is False.

    Returns
    -------
    mask_univariate : bool arr, shape (n_features)
        features pvalue < threshold_pvalue to keep.
    coef_univariate : float arr, shape (n_features)
        coeff in the cox univariate
    pvalue_univariate : float arr, shape (n_features)
        p_values in the cox univariate

    """
    
    #initialization
    n_patients, n_columns = np.shape(df_data)
    n_features = n_columns-2 # not count survival data (status and time)
    mask_univariate = np.zeros(n_features, dtype=bool)  # mask for dataframe (n_columns)
    pvalue_univariate = np.zeros(n_features, dtype=float) # pvalue of features (n_features)
    coef_univariate = np.zeros(n_features, dtype=float) # coef of features (n_features)
    
    # walk through features
    for k in range(n_features):
        
        # Cox univariate regression 
        mask_selection = np.zeros(n_columns, dtype=bool)
        mask_selection[0] = True
        mask_selection[1] = True
        mask_selection[k+2] = True 
        df_data_univariate = df_data.loc[:, mask_selection]

        cph = CoxPHFitter()
        cph.fit(df_data_univariate, duration_col = 'survival', event_col = 'status')
        pvalue_univariate[k] = (cph.summary["p"].values).round(3)
        coef_univariate[k] = (cph.summary["coef"].values)
     
    # apply a mutliple test correciton or not    
    if correction_method=="":
        mask_univariate = pvalue_univariate<threshold_pvalue    
    if correction_method=="bonferroni":
        threshold_pvalue = threshold_pvalue/n_features # Bonferroni correction
        mask_univariate = pvalue_univariate<threshold_pvalue
    if correction_method=="sidak":
        threshold_pvalue =1-(1-threshold_pvalue)**(1/n_features) # Bonferroni correction
        mask_univariate = pvalue_univariate<threshold_pvalue
    if correction_method=="fdr_bh":
        mask_univariate, pvals_corrected, alphacSidak, alphacBonf = multipletests(pvalue_univariate, alpha=threshold_pvalue, method="fdr_bh")
        
    # plot summary or not    
    if plot_summary:
        print()
        print("### Univariate - all features ###")
        print(df_data.columns)
        print("coef univariate, :" , coef_univariate)
        print("pvalue univariate, :", pvalue_univariate)
        print()

    return mask_univariate, coef_univariate, pvalue_univariate



def survival_multivariate(df_data, dict_params, threshold_pvalue = 0.05, verbose=2, n_fold=4, plot=False):
    """
    tune cox multivariate model (penalization grid). train the model selected and return coefficient and features pvalues

    Parameters
    ----------
    df_data : dataframe
        dataframe with survival data and features
    dict_params : dict
        parameters grid of the penalized Cox Multivariate model
    threshold_pvalue : float, optional
        pvalue theshold. The default is 0.05.
    verbose : int, optional
        verbose . The default is 2.
    n_fold : int, optional
        number of fold in k fold CV. The default is 4.
    plot : bool, optional
        plot or not HR in cox multivariate. The default is False.

    Returns
    -------
    mask_significant : bool arr
        mask of significant features 
    coef_multivariate : float arr
        coefficient of the cox multivariate model
    pvalue_multivariate : float arr
        pvalues of the cox multivariate model

    """
    

    list_penalizer = dict_params["penalizer"]
    list_l1_ratio = dict_params["l1_ratio"]
    
    # grid search initializatrion
    max_concordance = 0
    penalizer_best= list_penalizer[0]
    l1_ratio_best= list_l1_ratio[0]
    for penalizer in list_penalizer: # walk through parameters
        for l1_ratio in list_l1_ratio:
            print(f"params gridsearch = penalizer: {penalizer}, l1_ratio: {l1_ratio}")
            cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
            scores = k_fold_cross_validation(cph, df_data, 'survival', event_col='status', k=n_fold, scoring_method="concordance_index", seed=0) # CV 
            mean_score = np.mean(scores)
            if mean_score>max_concordance: # update best hyperparams
                penalizer_best = penalizer
                l1_ratio_best = l1_ratio
                max_concordance = mean_score
    max_concordance = max_concordance.round(3)        
    print(f"best concordance: {max_concordance}, best penalizer: {penalizer_best}, best l1_ratio: {l1_ratio_best}")          
            
    
    # refit with best hyperparams
    print()
    print("### Multivariate - all features ###")
    cph = CoxPHFitter(penalizer=penalizer_best, l1_ratio=l1_ratio_best)
    cph.fit(df_data, duration_col = 'survival', event_col = 'status')
    cph.print_summary()
    if plot:
        plt.subplots(figsize = (10, 6))
        cph.plot()
    
    
    # p_value
    coef_multivariate = cph.summary["coef"].values
    pvalue_multivariate = (cph.summary["p"].values).round(3)
    mask_significant = (cph.summary["p"]<threshold_pvalue).values
    summary_significant = cph.summary[mask_significant]
    print()
    print("### Multivariate - significant features ###")
    print(summary_significant)

    return mask_significant, coef_multivariate, pvalue_multivariate   



    
def survival_regression_score(survival_data, regression_score, q_01, q_25, q_50, q_75, q_99, title, plot=False):
    """
    split train or test patients regresion score into 4 and two groups (quartiles or median) according to train regression score quartiles. Apply logrank test.

    Parameters
    ----------
    survival_data : array (n_patients, 2)
        censorship and survival time
    regression_score : float arr, shape (n_patients)
        regression score for each patients
    q_01, q_25, q_50, q_75, q_99 : float
        percentile of the train regression score to apply to train or test set 
    title : str
        title for the kaplan meier curves
    plot : bool, optional
        plot or not histogram and kaplan meier curves. The default is False.

    Returns
    -------
    pvalue_logrank : float
        logrank pvalue (train or test set)
    test_statistic : float
        logrank stat (train or test set)

    """

    
    # prepare data
    status = np.array(survival_data[:, 0], dtype=bool)
    survival = survival_data[:, 1]
    
    #plot the regression score histogram
    if plot:
        plt.figure()
        plt.hist(regression_score, range=[q_01, q_99], density=True) 
        plt.title("histogram regression score - " + title)
        plt.xlabel("regression score")
        plt.ylabel("proportion")
        plt.show()
    
    # prepare patients groups according to quartiles
    mask_1 = regression_score < q_25
    mask_2 = (regression_score >= q_25) & (regression_score < q_50)
    mask_3 = (regression_score >= q_50) & (regression_score < q_75)
    mask_4 = regression_score >= q_75
    
    # logrank test with 4 groups: quartiles split
    groups = 1*mask_1 + 2*mask_2 + 3*mask_3 + 4*mask_4
    y_surv = Surv.from_arrays(status, survival)
    chisq,  pvalue, df, cov = compare_survival(y_surv, groups, return_stats=True)
    pvalue_klogrank = pvalue.round(4)
    
    # plot kaplan meieir curvres
    if plot:    
     
        plt.figure()
        kmf = KaplanMeierFitter()
        kmf.fit(durations=survival[mask_1], event_observed=status[mask_1], label="1")
        kmf.plot_survival_function()
        if np.sum(mask_2)!=0:
            kmf.fit(durations=survival[mask_2], event_observed=status[mask_2], label="2")
            kmf.plot_survival_function()
            kmf.fit(durations=survival[mask_3], event_observed=status[mask_3], label="3")
            kmf.plot_survival_function()
            kmf.fit(durations=survival[mask_4], event_observed=status[mask_4], label="4")
            kmf.plot_survival_function()
            
            plt.annotate(f"logrank test, pvalue: {pvalue_klogrank}", xy=(0.05, 0.05), xycoords='axes fraction')
            plt.ylabel("est. probability of survival ")
            plt.xlabel("time (in months)")
            plt.title("Kaplan-Meier: quartiles regression score - {}".format(title))
            plt.show()
    
    # logrank test of two groups: median split
    mask_median = regression_score < q_50
    results = logrank_test(survival[mask_median], survival[~mask_median] , status[mask_median], status[~mask_median], alpha=.95)
    pvalue_logrank = (results.summary["p"].values)[0]
    test_statistic = results.summary["test_statistic"].values
    
    #plot kaplan meieir curve
    if plot:
        
        plt.figure()
        kmf = KaplanMeierFitter()
        kmf.fit(durations=survival[mask_median], event_observed=status[mask_median], label="RS<median")
        kmf.plot_survival_function()
        kmf.fit(durations=survival[~mask_median], event_observed=status[~mask_median], label="RS>median")
        kmf.plot_survival_function()
        
        plt.annotate(f"logrank test, pvalue: {pvalue_logrank.round(4)}", xy=(0.05, 0.05), xycoords='axes fraction')
        plt.ylabel("est. probability of survival")
        plt.xlabel("time (in months)")
        plt.title("Kaplan-Meier: median regression score - {}".format(title))
        plt.show()
    
    return pvalue_logrank, test_statistic



def survival_clinical_features_test(parameters, test_size = 0.2):
    """
    the aim of this function is to compare clinical features ability to split patients into high and low risk groups versus others groups of features. 
    So we have to apply the same train/test than in survival_pipeline function
    
    
    loop to split patients into train/test (according to censorship and time stratification):
        - specific threshold for each clinical feature that split patient into high and low risk
        - logrank and pvalue
    

    Parameters
    ----------
    parameters : class parameters
        parameters script
    test_size : float, optional
        proportion of the test set. The default is 0.2.

    Returns
    -------
    logrank_pvalues_groups: logrank of each clinical features split separation

    """
    
    # Just clin features
    parameters.aaIPI = True
    parameters.clinical_features = True
    parameters.classical_features = False
    parameters.radiomics_features = False

    
    parameters.outcome = "pfs"
    parameters.standardization = False #False for the threshold
    X, features_name, survival_data, patients_id = features_processing(parameters)

    # initialize pvalues, and combinaiton
    n_clin_features = 6
    logrank_pvalues_groups = np.ones((parameters.n_loop_survival, n_clin_features), dtype="float")
    
    # prepare data
    status = np.array(survival_data[:, 0], dtype=bool)
    survival = np.array(survival_data[:, 1], dtype=int)
    stratification = np.zeros_like(survival)
    n_patients = np.shape(status)[0]
    n_test_size = int(n_patients * test_size)
    
    data_array = np.column_stack((survival_data, X))
    columns_name =   ["status", "survival"] + list(features_name)
    df_data = pd.DataFrame(data_array, columns=columns_name)
    
    # stratification for the train/test split: 8 class with respect to status=1 or 0 and quartile of the censorship    
    # patient with events
    q_25, q_50, q_75 = np.percentile(survival[status], [25, 50, 75])
    mask_1 = survival[status] < q_25
    mask_2 = (survival[status] >= q_25) & (survival[status] < q_50)
    mask_3 = (survival[status] >= q_50) & (survival[status] < q_75)
    mask_4 = survival[status] >= q_75
    stratification[status] = 1*mask_1 + 2*mask_2 + 3*mask_3 + 4*mask_4
    
    # patient without events
    q_25, q_50, q_75 = np.percentile(survival[~status], [25, 50, 75])
    mask_1 = survival[~status] < q_25
    mask_2 = (survival[~status] >= q_25) & (survival[~status] < q_50)
    mask_3 = (survival[~status] >= q_50) & (survival[~status] < q_75)
    mask_4 = survival[~status] >= q_75
    stratification[~status] = 5*mask_1 + 6*mask_2 + 7*mask_3 + 8*mask_4

    # same train/split loop than in survival_pipeline function
    for k in range(parameters.n_loop_survival):     
        print("loop: ", k)
        df_data_train, df_data_test = train_test_split(df_data, test_size=n_test_size, shuffle=True, stratify=stratification, random_state=k)
        X_train = df_data_train.to_numpy("float")[:, 2:]
        X_test = df_data_test.to_numpy("float")[:, 2:]
        survival_data_test = df_data_test.to_numpy("float")[:, :2]
    
        status = np.array(survival_data_test[:, 0], dtype=bool)
        survival = np.array(survival_data_test[:, 1], dtype=int)
        
        # mask of clinical features, specific theshold relative to clincal feature
        mask_aaIPI = X_test[:, 0]>2
        mask_age = X_test[:, 1] > np.median(X_train[:, 1])
        mask_aas = X_test[:, 2]>2
        mask_ecog = X_test[:, 3]>1
        mask_extranodal_site = X_test[:, 4]>1
        mask_LDH = X_test[:, 5] > np.median(X_train[:, 5])
    
        # apply mask to split patients and compute logrank test
        mask_group = mask_aaIPI
        results = logrank_test(survival[mask_group], survival[~mask_group] , status[mask_group], status[~mask_group], alpha=.95)
        pvalue_logrank = (results.summary["p"].values)[0]
        logrank_pvalues_groups[k, 0] = pvalue_logrank
        
        mask_group = mask_age
        results = logrank_test(survival[mask_group], survival[~mask_group] , status[mask_group], status[~mask_group], alpha=.95)
        pvalue_logrank = (results.summary["p"].values)[0]
        logrank_pvalues_groups[k, 1] = pvalue_logrank
        
        mask_group = mask_aas
        results = logrank_test(survival[mask_group], survival[~mask_group] , status[mask_group], status[~mask_group], alpha=.95)
        pvalue_logrank = (results.summary["p"].values)[0]
        logrank_pvalues_groups[k, 2] = pvalue_logrank
        
        mask_group = mask_ecog
        results = logrank_test(survival[mask_group], survival[~mask_group] , status[mask_group], status[~mask_group], alpha=.95)
        pvalue_logrank = (results.summary["p"].values)[0]
        logrank_pvalues_groups[k, 3] = pvalue_logrank
        
        mask_group = mask_extranodal_site
        results = logrank_test(survival[mask_group], survival[~mask_group] , status[mask_group], status[~mask_group], alpha=.95)
        pvalue_logrank = (results.summary["p"].values)[0]
        logrank_pvalues_groups[k, 4] = pvalue_logrank
        
        mask_group = mask_LDH
        results = logrank_test(survival[mask_group], survival[~mask_group] , status[mask_group], status[~mask_group], alpha=.95)
        pvalue_logrank = (results.summary["p"].values)[0]
        logrank_pvalues_groups[k, 5] = pvalue_logrank
        
    return(logrank_pvalues_groups)



def survival_features_comparison(parameters):
    """
    apply survival_pipeline function to different groups of features ("clinical", "classical", "radiomics", "clinical+classical", "clinical+radiomics", "clinical+classical+radiomics")
    
    Then combine p_values of each features groups with harmonic mean pvalue and fischer method
    
    Compare features groups pvalues with wilcoxon ranked test
    
    same thing with clinical features alone (pvalues combination) and wilcoxon of clinical features regression score versus each clinical features alone

    Parameters
    ----------
    parameters : class parameters
        parameters script

    Returns
    -------
    logrank_pvalues_groups, hmp_groups, fcomb_groups, wilcoxon_less_groups, logrank_pvalues_clinical, hmp_clinical, fcomb_clinical, wilcoxon_less_clinical
    
    """
    
    # initialization
    groups = ["clinical", "consolidation", "conventional", "radiomics", "clinical+conventional", "clinical+radiomics", "clinical+conventional+radiomics", "clinical + consolidation", "consolidation+conventional", "consolidation+radiomics", "clinical+consolidation+conventional", "clinical+consolidation+radiomics", "clinical+consolidation+conventional+radiomics"]
    n_groups = len(groups) # clin, class, rad, clin+class, clin+rad, clin+class+rad
    
    # output p_values of survival =_pipeline for each features groups
    if parameters.compute_pvalues_survival_comparison:
        
        parameters.standardization = True # reset standardization 
        
        # clinical features
        parameters.aaIPI = False
        parameters.clinical_features = True
        parameters.classical_features = False
        parameters.radiomics_features = False
        parameters.consolidation_features = False
        parameters.outcome = "pfs"
        
        X_largest, features_name_largest, survival_data, patients_id = features_processing(parameters)
        logrank_pvalues, logrank_stats = survival_pipeline(survival_data, X_largest, features_name_largest, parameters.dict_params_Cox, n_loop=parameters.n_loop_survival, correction_method=parameters.correction_method, multivariate_significant=parameters.multivariate_significant, plot_survival=False) 
        n_test = np.shape(logrank_pvalues)[0]
        logrank_pvalues_groups = np.zeros((n_test, n_groups))
        logrank_pvalues_groups[:, 0] = logrank_pvalues

        # consolidation features
        parameters.aaIPI = False
        parameters.clinical_features = False
        parameters.classical_features = False
        parameters.radiomics_features = False
        parameters.consolidation_features = True
        parameters.outcome = "pfs"
        
        X_largest, features_name_largest, survival_data, patients_id = features_processing(parameters)
        logrank_pvalues, logrank_stats = survival_pipeline(survival_data, X_largest, features_name_largest, parameters.dict_params_Cox, n_loop=parameters.n_loop_survival, correction_method=parameters.correction_method, multivariate_significant=parameters.multivariate_significant, plot_survival=False) 
        logrank_pvalues_groups[:, 1] = logrank_pvalues
        
        
        # conventional features
        parameters.aaIPI = False
        parameters.clinical_features = False
        parameters.classical_features = True
        parameters.radiomics_features = False
        parameters.consolidation_features = False
        parameters.outcome = "pfs"
        
        X_largest, features_name_largest, survival_data2, patients_id = features_processing(parameters)
        logrank_pvalues, logrank_stats = survival_pipeline(survival_data, X_largest, features_name_largest, parameters.dict_params_Cox, n_loop=parameters.n_loop_survival, correction_method=parameters.correction_method,multivariate_significant=parameters.multivariate_significant, plot_survival=False) 
        logrank_pvalues_groups[:, 2] = logrank_pvalues
        
        # radiomics features
        parameters.aaIPI = False
        parameters.clinical_features = False
        parameters.classical_features = False
        parameters.radiomics_features = True
        parameters.consolidation_features = False
        parameters.outcome = "pfs"
        
        X_largest, features_name_largest, survival_data, patients_id = features_processing(parameters)
        logrank_pvalues, logrank_stats = survival_pipeline(survival_data, X_largest, features_name_largest, parameters.dict_params_Cox, n_loop=parameters.n_loop_survival, correction_method=parameters.correction_method, multivariate_significant=parameters.multivariate_significant, plot_survival=False) 
        logrank_pvalues_groups[:, 3] = logrank_pvalues    
        
        # clinical+conventinal features
        parameters.aaIPI = False
        parameters.clinical_features = True
        parameters.classical_features = True
        parameters.radiomics_features = False
        parameters.consolidation_features = False
        parameters.outcome = "pfs"
        
        X_largest, features_name_largest, survival_data, patients_id = features_processing(parameters)
        logrank_pvalues, logrank_stats = survival_pipeline(survival_data, X_largest, features_name_largest, parameters.dict_params_Cox, n_loop=parameters.n_loop_survival, correction_method=parameters.correction_method, multivariate_significant=parameters.multivariate_significant, plot_survival=False) 
        logrank_pvalues_groups[:, 4] = logrank_pvalues    
            
        # clinical+radiomics features
        parameters.aaIPI = False
        parameters.clinical_features = True
        parameters.classical_features = False
        parameters.radiomics_features = True
        parameters.consolidation_features = False
        parameters.outcome = "pfs"
        
        X_largest, features_name_largest, survival_data, patients_id = features_processing(parameters)
        logrank_pvalues, logrank_stats = survival_pipeline(survival_data, X_largest, features_name_largest, parameters.dict_params_Cox, n_loop=parameters.n_loop_survival, correction_method=parameters.correction_method, multivariate_significant=parameters.multivariate_significant, plot_survival=False) 
        logrank_pvalues_groups[:, 5] = logrank_pvalues    
        
        # clinical + conventinal + radiomics features
        parameters.aaIPI = False
        parameters.clinical_features = True
        parameters.classical_features = True
        parameters.radiomics_features = True
        parameters.consolidation_features = False
        parameters.outcome = "pfs"
        
        X_largest, features_name_largest, survival_data, patients_id = features_processing(parameters)
        logrank_pvalues, logrank_stats = survival_pipeline(survival_data, X_largest, features_name_largest, parameters.dict_params_Cox, n_loop=parameters.n_loop_survival, correction_method=parameters.correction_method, multivariate_significant=parameters.multivariate_significant, plot_survival=False) 
        logrank_pvalues_groups[:, 6] = logrank_pvalues      






        # clinical + consolidation features
        parameters.aaIPI = False
        parameters.clinical_features = True
        parameters.classical_features = False
        parameters.radiomics_features = False
        parameters.consolidation_features = True
        parameters.outcome = "pfs"
        
        X_largest, features_name_largest, survival_data, patients_id = features_processing(parameters)
        logrank_pvalues, logrank_stats = survival_pipeline(survival_data, X_largest, features_name_largest, parameters.dict_params_Cox, n_loop=parameters.n_loop_survival, correction_method=parameters.correction_method, multivariate_significant=parameters.multivariate_significant, plot_survival=False) 
        logrank_pvalues_groups[:, 7] = logrank_pvalues
    
        # consolidation + conventional features
        parameters.aaIPI = False
        parameters.clinical_features = False
        parameters.classical_features = True
        parameters.radiomics_features = False
        parameters.consolidation_features = True
        parameters.outcome = "pfs"
        
        X_largest, features_name_largest, survival_data2, patients_id = features_processing(parameters)
        logrank_pvalues, logrank_stats = survival_pipeline(survival_data, X_largest, features_name_largest, parameters.dict_params_Cox, n_loop=parameters.n_loop_survival, correction_method=parameters.correction_method,multivariate_significant=parameters.multivariate_significant, plot_survival=False) 
        logrank_pvalues_groups[:, 8] = logrank_pvalues
        
        # consolidation + radiomics features
        parameters.aaIPI = False
        parameters.clinical_features = False
        parameters.classical_features = False
        parameters.radiomics_features = True
        parameters.consolidation_features = True
        parameters.outcome = "pfs"
        
        X_largest, features_name_largest, survival_data, patients_id = features_processing(parameters)
        logrank_pvalues, logrank_stats = survival_pipeline(survival_data, X_largest, features_name_largest, parameters.dict_params_Cox, n_loop=parameters.n_loop_survival, correction_method=parameters.correction_method, multivariate_significant=parameters.multivariate_significant, plot_survival=False) 
        logrank_pvalues_groups[:, 9] = logrank_pvalues    
        
        # clinical+ consolidation + conventional features
        parameters.aaIPI = False
        parameters.clinical_features = True
        parameters.classical_features = True
        parameters.radiomics_features = False
        parameters.consolidation_features = True
        parameters.outcome = "pfs"
        
        X_largest, features_name_largest, survival_data, patients_id = features_processing(parameters)
        logrank_pvalues, logrank_stats = survival_pipeline(survival_data, X_largest, features_name_largest, parameters.dict_params_Cox, n_loop=parameters.n_loop_survival, correction_method=parameters.correction_method, multivariate_significant=parameters.multivariate_significant, plot_survival=False) 
        logrank_pvalues_groups[:, 10] = logrank_pvalues    
            
        # clinical+ consolidation + radiomics features
        parameters.aaIPI = False
        parameters.clinical_features = True
        parameters.classical_features = False
        parameters.radiomics_features = True
        parameters.consolidation_features = True
        parameters.outcome = "pfs"
        
        X_largest, features_name_largest, survival_data, patients_id = features_processing(parameters)
        logrank_pvalues, logrank_stats = survival_pipeline(survival_data, X_largest, features_name_largest, parameters.dict_params_Cox, n_loop=parameters.n_loop_survival, correction_method=parameters.correction_method, multivariate_significant=parameters.multivariate_significant, plot_survival=False) 
        logrank_pvalues_groups[:, 11] = logrank_pvalues    
        
        # clinical + consolidation  + conventinal + radiomics features
        parameters.aaIPI = False
        parameters.clinical_features = True
        parameters.classical_features = True
        parameters.radiomics_features = True
        parameters.consolidation_features = True
        parameters.outcome = "pfs"
        
        X_largest, features_name_largest, survival_data, patients_id = features_processing(parameters)
        logrank_pvalues, logrank_stats = survival_pipeline(survival_data, X_largest, features_name_largest, parameters.dict_params_Cox, n_loop=parameters.n_loop_survival, correction_method=parameters.correction_method, multivariate_significant=parameters.multivariate_significant, plot_survival=False) 
        logrank_pvalues_groups[:, 12] = logrank_pvalues    
        
        
        
        
        # clinical features one by one with  specific threshold
        logrank_pvalues_clinical = survival_clinical_features_test(parameters, test_size = 0.2)
        
        # save pvalues vectors
        if parameters.save_pvalues_survival_comparison:
            np.save(parameters.savename_pvalues_survival_comparison, logrank_pvalues_groups)
            np.save(parameters.savename_pvalues_survival_comparison + "_clinical", logrank_pvalues_clinical)
    
    #laod pvalues vectors
    else:
        logrank_pvalues_groups = np.load(parameters.savename_pvalues_survival_comparison + ".npy")
        logrank_pvalues_clinical = np.load(parameters.savename_pvalues_survival_comparison + "_clinical.npy")
        
        # groups = ["clinical",  "conventional", "radiomics", "clinical+conventional", "clinical+radiomics", "clinical+conventional+radiomics", "clinical + consolidation", "consolidation+conventional", "consolidation+radiomics", "clinical+consolidation+conventional", "clinical+consolidation+radiomics", "clinical+consolidation+conventional+radiomics"]
        # logrank_pvalues_groups = np.delete(logrank_pvalues_groups, 1, 1)
        # n_groups =n_groups -1
        
        
        
    ## features groups analysis: p_values combination (fischer and hmp) and wilcoxon ranked test  
    
    hmp_groups = np.ones(n_groups, dtype="float")   # initalize harmonic mean pvalue groups  
    fcomb_groups = np.ones(n_groups, dtype="float")  # initalize pvalue fischer combinaiton groups  
    wilcoxon_less_groups = np.ones((n_groups, n_groups), dtype="float") # initalize  pvalue wilcoxon ranked test alternative=less
    for k in range(n_groups):
        hmp_groups[k] = hmean(logrank_pvalues_groups[:, k]).round(5) # harmonic mean pvalue
        stat, fcomb = combine_pvalues(logrank_pvalues_groups[:, k], method='fisher', weights=None) #fischer combination
        fcomb_groups[k] = fcomb.round(5)
        plot_histogram(logrank_pvalues_groups[:, k], groups[k],  n_bins=40)
        
        
        for l in range(n_groups):
            if l!=k:
                wilc_stat, wilc_pvalue = wilcoxon(logrank_pvalues_groups[:, k], logrank_pvalues_groups[:, l], alternative = 'less') #wilcoxon ranked test  
                wilcoxon_less_groups[k,l] = wilc_pvalue.round(5)

    # put results in dataframe
    hmp_groups = DataFrame([hmp_groups], columns=groups, index=["hmp"])
    fcomb_groups = DataFrame([fcomb_groups], columns=groups, index=["fcomb"])
    wilcoxon_less_groups = DataFrame(wilcoxon_less_groups, columns=groups,  index=groups)
    
    # plot boxplot    
    boxplot(logrank_pvalues_groups, groups) 
    
    ## clinical features analysis: p_values combination (fischer and hmp), wilcoxon ranked test  
    
    features_name = ["aaIPI", "age", "aas", "ecog", "n_extranodal", "LDH"]
    n_clin_features = np.shape(logrank_pvalues_clinical)[1]
    hmp_clinical = np.ones(n_clin_features, dtype="float")   # initalize harmonic mean pvalue groups  
    fcomb_clinical = np.ones(n_clin_features, dtype="float")  # initalize pvalue fischer combinaiton groups  
    wilcoxon_less_clinical = np.ones(n_clin_features, dtype="float") # initalize  pvalue wilcoxon ranked test alternative=less
    for k in range(n_clin_features):
        hmp_clinical[k] = hmean(logrank_pvalues_clinical[:, k]).round(4) # harmonic mean pvalue
        stat, fcomb = combine_pvalues(logrank_pvalues_clinical[:, k], method='fisher', weights=None) #fischer combination
        fcomb_clinical[k] = fcomb.round(4)
        plot_histogram(logrank_pvalues_clinical[:, k], features_name[k],  n_bins=40)
        
        wilc_stat, wilc_pvalue = wilcoxon(logrank_pvalues_groups[:, 0], logrank_pvalues_clinical[:, k], alternative = 'less') #wilcoxon ranked test  
        wilcoxon_less_clinical[k] = wilc_pvalue.round(5)

    # put results in dataframe
    hmp_clinical = DataFrame([hmp_clinical], columns=features_name, index=["hmp"])
    fcomb_clinical = DataFrame([fcomb_clinical], columns=features_name, index=["fcomb"])    
    wilcoxon_less_clinical = DataFrame([wilcoxon_less_clinical], columns=features_name,  index=["RS clinical"])         
       
    # plot boxplot    
    boxplot(logrank_pvalues_clinical, features_name) 
    
    return(logrank_pvalues_groups, hmp_groups, fcomb_groups, wilcoxon_less_groups, logrank_pvalues_clinical, hmp_clinical, fcomb_clinical, wilcoxon_less_clinical)




def plot_histogram(values, name,  n_bins):
    
    # plot pvalues histogram 
    plt.figure()
    plt.hist(values, bins=n_bins, range=[0, 1]) #, density=True) 
    plt.ylim([0, 50])
    plt.title("histogram p values - " + name + " features" )
    plt.xlabel("pvalues")
    plt.ylabel("N")
    plt.show()

def boxplot(values, groups):
    
    n_groups=len(groups)
    # boxplot groups p values 
    positions = range(n_groups)
    fig = plt.figure()
    fig.suptitle('logrank test p values' )
    ax = fig.add_subplot(111)
    bp0 = ax.boxplot(values, positions=positions, patch_artist=True, showfliers=False)
    plt.setp(bp0['boxes'], color='darkorange', facecolor='moccasin')
    plt.setp(bp0['medians'], color='darkorange', linewidth=2)
    plt.setp(bp0['whiskers'], linestyle=':')
    plt.setp(bp0['caps'], color='darkorange')
    ax.set_xticklabels(groups)
    plt.ylim([1e-5, 1])
    plt.yscale('log')

    #plt.legend([bp0["boxes"][0]], ["p values"])
    plt.xticks(rotation=70)
    plt.subplots_adjust(top=0.92, bottom=0.46)
    plt.xlabel("Groups")
    plt.ylabel("p values")
    plt.show()
    
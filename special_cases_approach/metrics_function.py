# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 10:46:11 2022

@author: gafrecon
"""



# import rpy2.robjects as ro
# import rpy2.robjects.numpy2ri
# rpy2.robjects.numpy2ri.activate()
import numpy as np
from sklearn.metrics import roc_curve
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, recall_score, f1_score, accuracy_score, roc_auc_score, balanced_accuracy_score, roc_curve, RocCurveDisplay, precision_recall_curve



def compute_metrics_sklearn(y_proba, y, threshold=0.5):

    # y pred
    y_pred = np.array(y_proba>0.5, dtype=int)
    
    
    # metrics computes y=known classifictions, y_pred=classifications predicted by the LOO cv
    auc = roc_auc_score(y, y_proba).round(2)
    accuracy = accuracy_score(y, y_pred).round(2)
    f1_score_positive = f1_score(y, y_pred, pos_label=1).round(2)
    f1_score_negative = f1_score(y, y_pred, pos_label=0, zero_division=0).round(2)
    sensitivity = recall_score(y, y_pred, pos_label=1).round(2)
    specificity = recall_score(y, y_pred, pos_label=0, zero_division=0).round(2)
    balanced_accuracy = balanced_accuracy_score(y, y_pred).round(2)
    
    return y_pred, accuracy, specificity, sensitivity, f1_score_positive, f1_score_negative, balanced_accuracy, auc


def compute_metrics_option(y, y_proba, specificity_threshold, bool_plot):
    """
    compute metrics with respect to two option.
        - specificity_threshold = False: we don't fix the specificity so we compute '

    Parameters
    ----------
    y : array
        outcome
    y_proba :array
        float proba array
    specificity_threshold : Bool or float
        False for sklearn metrics compûting, float to fix the spec threshold (smooth roc curve)
    bool_plot : bool

    Returns
    -------
    None.

    """
    
    tfpr, pr, thresholds = roc_curve(y, y_proba)
    n_thresholds = len(thresholds)
    smooth_possible = n_thresholds>4
    
    if not(specificity_threshold) or not(smooth_possible):
        y_pred, accuracy, specificity, sensitivity, f1_score_positive, f1_score_negative, balanced_accuracy, auc = compute_metrics_sklearn(y_proba, y, threshold=0.5)
   
    else:    
        y_pred, accuracy, specificity, sensitivity, f1_score_positive, f1_score_negative, balanced_accuracy, auc = R_metric_smooth_roc(y, y_proba, specificity_threshold, bool_plot)

    return y_pred, accuracy, specificity, sensitivity, f1_score_positive, f1_score_negative, balanced_accuracy, auc



def R_metric_smooth_roc(y, y_proba, specificity_threshold, plot):
    """
    

    Parameters
    ----------
    y : R vector
        label
    y_proba : R vector
        proba
    specifity_threshold : float
        specificity threshold ex: 0.7 or 0.9
    plot : boolTYPE
        plot or not roc curves smoothed and non smoothed

    Returns
    -------
    metrics : R float vector
        accuracy, specificity, sensitivity, f1_positive, f1_negative, auc

    """
    r = ro.r
    path="C:/Users/gafrecon/Documents/TFE_CHU/Infective_endocharditis/codes/" #"../"
    r.source(path+"metric_smooth_roc.R")
    y_pred, metrics = r.metric_spec_theshold(y, y_proba, specificity_threshold, plot)
    accuracy, specificity, sensitivity, f1_positive, f1_negative, balanced_accuracy, auc = metrics
    return y_pred, accuracy, specificity, sensitivity, f1_positive, f1_negative, balanced_accuracy, auc


if __name__ == "__main__":
    
    y=np.array([1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1])
    y_proba = np.array([0.67053719, 0.5255298 , 0.14629644, 0.30809895, 0.17024713,
      0.35937919, 0.86047918, 0.17157546, 0.84428904, 0.37784014,
        0.32219943, 0.27392888, 0.89798804, 0.6091058 , 0.70389107,
        0.18960525, 0.72382441, 0.43375593, 0.45725107, 0.16753594,
        0.96477687, 0.17132162])


    
    y_pred, accuracy, specificity, sensitivity, f1_positive, f1_negative, balanced_accuracy, auc = R_metric_smooth_roc(y, y_proba, 0.9, True)   # calling the function with passing arguments

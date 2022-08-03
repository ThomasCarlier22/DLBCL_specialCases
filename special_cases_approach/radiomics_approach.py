# -*- coding: utf-8 -*-
"""
Created on Tue May 11 17:10:03 2021

@author: GauthierFrecon
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 08:24:39 2021

@author: GauthierFrecon
"""

#### Internship- Infective Endocharditis Classification #####


# =============================================================================
# First approach - Radiomics approach
# =============================================================================

import os 

# script_path = "C:/Users/gfrecon/Documents/TFE_CHU/stage_diagnostic_endochardites/codes/radiomics_approach/"
script_path="/home/kouame/Bureau/Dossiers_stage/special_cases_approach"
os.chdir(script_path)



import pandas as pd
from pandas import DataFrame 
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from radiomics_computation import radiomics_extraction, radiomics_pipeline, PCA_analysis, spearman_analysis, processing_features_special_cases
from classification_pipeline import LOO_cross_validation, gridsearch_loo, gridsearch_models, models_analysis, feature_importance, outcome_analysis, test_analysis_LOO, test_analysis_hold_out
from collections import Counter
import SimpleITK as sitk
from mpl_toolkits.mplot3d import Axes3D

from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.pipeline import Pipeline

from scipy.stats import spearmanr
import scipy
from scipy.stats import fisher_exact
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, LeaveOneOut, cross_val_predict, ParameterGrid
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, recall_score, f1_score, accuracy_score, make_scorer, roc_curve, roc_auc_score
from sklearn.cluster import FeatureAgglomeration


# models
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB 
from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
#from xgboost import XGBClassifier
import pydicom


from radiomics_computation import correlation_selection
from survival import survival_pipeline



import openpyxl as xl

# =============================================================================
# pyradiomics extractions setings
# =============================================================================

settings_pyradiomics = {}
settings_pyradiomics['binCount'] = 64
#settings_pyradiomics['binWidth'] = 0.3
settings_pyradiomics['resampledPixelSpacing'] = [2,2,2] #[4,4,2]
settings_pyradiomics['interpolator'] = sitk.sitkBSpline
settings_pyradiomics['preCrop'] = True
settings_pyradiomics['weightingNorm'] = 'no_weighting'
settings_pyradiomics['sigma'] = [2, 6]


# =============================================================================
# Parameters
# =============================================================================

class parameters:
    
    def __init__(self, settings_pyradiomics):
        
        self.remove_nosuspicion = False
        self.valid_exclude_nosuspicion = False #set false if remove_nosuspicion is true

        # processing
        
        self.test_size =  0.2
        self.hierarchical_clustering = True
        self.correlation_threshold = 0.9
        self.dendro_plot = False
        self.p_dendrogram=12
        self.leaf_font_size=9
        self.no_labels = False   

    
        self.pca_analysis = False
        if  self.pca_analysis:
            self.test_analysis_pca = True
            self.pca_replacement = False
        
        
        self.optim_metric_name = "balanced_accuracy" # "auc" or "balanced_accuracy
        self.specificity_threshold = False #0.7 #0.7 # False or 0.7  ,  float that set specificity to fixed the threshold probability 
        
        
        
        ## classification train/validaiton

        self.grid_search_model = True ################################### Mettre à false#############################################
        if self.grid_search_model:
            self.loop_evaluation = 3
        

        
        ## analysis best models and test classification
      
        self.test_results = True ######################################## Mettre à true############################################
        if self.test_results:
            self.loop_test = 100  ######################################## Mettre à 100###########################################
            self.test_plot = True
            self.test_type ="hold_out" #"LOO" #LOO or hold out test type
            
        ## other analysis    
        self.feature_importance = True ########################################## Mettre à false#################################
        self.outcome_correlation = False ########################################## Mettre à false#################################
        if self. outcome_correlation:
            self.alternative = "less" #alternative hypothesis, greater: negative dist greater than positive, "less": positive greater than negative distr, "two-sided": distr not equal
        

        ## Parameters for analysis

        self.sampling_ridge = SMOTE()
        self.C_ridge= 0.00001



        ################ survival ################ 
        self.survival_prediction = True #Mettre à True
        self.dict_params_Cox= {
                            "penalizer": 10.0 ** np.arange(-4, 0),
                            "l1_ratio": [0]

                            }
        self.n_loop_survival = 100 # change to 100
        self.correction_method = "" # correction to apply in univariate analysis ""= no correciton, "bonferroni", or "fdr_bh"
        self.multivariate_significant = False
        self.plot_survival = True


parameters = parameters(settings_pyradiomics)



# =============================================================================
# Extraction of features
# =============================================================================



plt.close('all')

X_train_test = np.load("./data/features.npy")
y_train_test = np.load("./data/PFS_outcome.npy")
survival_train_test = np.load("./data/survival_data.npy")  # to modifief
features_name = np.load("./data/features_name.npy")



X_train_test_processed, X, X_test, y, y_test, features_name = processing_features_special_cases(X_train_test, y_train_test,  features_name, parameters)
n_patients, n_features = np.shape(X)## Training and Validation patients

n_patients_test, n_features = np.shape(X_test) #Test patients
list_patients_int = np.arange(n_patients)
list_patients_test = np.arange(n_patients_test)

mask_nosuspicion = np.zeros(n_patients, dtype=bool)    



# =============================================================================
# classification
# =============================================================================



if __name__=="__main__":

    
    
    ## grid params:
    sampling_opt = [SMOTE(),'passthrough', ADASYN()] 
    
    #LR
    C_opt_ridge = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]

    
    
    #direct grid
    LR_ridge_dict = {'sampling': sampling_opt,
                'clf': [LogisticRegression()],
                'clf__C': C_opt_ridge,
                'clf__penalty': ['l2']
                }   

     
    
    if  parameters.grid_search_model:
        pipeline = Pipeline([('sampling', 'passthrough'),
                             ('clf', LogisticRegression()) ])
        
        metrics, ind = gridsearch_loo(X, y, parameters.remove_nosuspicion, parameters.valid_exclude_nosuspicion, mask_nosuspicion, pipeline, LR_ridge_dict, list_patients_int, parameters.loop_evaluation, parameters.optim_metric_name, parameters.specificity_threshold, additional_comparison=True)
        df_LR = DataFrame(metrics, columns=['model', 'sampling', 'params', 'accuracy', "f1_score_positive", "f1_score_negative", 'sensitivity', 'specificity', 'balanced_accuracy', 'auc', 'std']) 
    
    
    
    

    if parameters.feature_importance:
    
        ##concatenation with the test database
        X = np.concatenate((X, X_test), axis=0)
        y = np.concatenate((y, y_test), axis=0)
    


        # LR ridge
        pipeline1 = Pipeline([('sampling', SMOTE()),    
                          ('clf', LogisticRegression(C=0.1))])
        
        # LR lasso
        pipeline2 = Pipeline([('sampling', SMOTE()),    
                          ('clf', LogisticRegression(penalty='l1', C=0.5, solver='liblinear'))])
        
        # RF
        pipeline3 = Pipeline([('sampling', SMOTE()),    
                              ('clf', RandomForestClassifier(n_estimators=50, min_samples_split=2, max_depth=30, n_jobs=-1))])
        
        importance1 = feature_importance(X, y, features_name, pipeline1, n_loop=50)
        importance2 = feature_importance(X, y, features_name, pipeline2, n_loop=50)
        importance3 = feature_importance(X, y, features_name, pipeline3, n_loop=50)
    
    
        ## to do properly, in a new function define in classification_pipeline.py 

        #importance normalized by the coefficients sum
        importance1 = importance1/np.sum(importance1)
        importance2 = importance2/np.sum(importance2)
        importance3 = importance3/np.sum(importance3)
        
        #bar plot comparison features importance                      
        positions = np.array(range(n_features))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.bar(positions-0.2, importance1, width=0.2, label="LR ridge")
        ax.bar(positions, importance2, width=0.2, label="LR lasso")
        ax.bar(positions+0.2, importance3, width=0.2, label="RF")
        ax.set_xticks(positions)
        ax.set_xticklabels(features_name)
        plt.ylabel("importance / importance max")
        plt.xticks(rotation=90)
        plt.subplots_adjust(top=0.960, bottom=0.40)
        plt.title("features importance")
        plt.legend(loc='best')
        plt.show()
    
    
    
    
    if parameters.outcome_correlation:

            
        
        vect_stat, vect_pvalue = outcome_analysis(X, y, features_name)
            

    
    if parameters.test_results:
        
        LR_ridge_dict = {'sampling': [parameters.sampling_ridge], 'clf': [LogisticRegression()], 'clf__C': [parameters.C_ridge], 'clf__penalty': ['l2']}  
        dict_models = [LR_ridge_dict] 
        
        if parameters.test_type=="hold_out":
            df_metrics_hold_out, metrics, df_pred_models, df_metrics_sklearn = test_analysis_hold_out(X, y, X_test, y_test,  list_patients_test, parameters.remove_nosuspicion, mask_nosuspicion, dict_models, parameters.loop_test, parameters.specificity_threshold, parameters.test_plot)
            
            # print metrics       
            metrics_mean = np.mean(metrics, axis=1).round(2)
            metrics_std = np.std(metrics, axis=1).round(2)
            print(df_metrics_hold_out.columns)
            print("metrics \n", metrics_mean)
            print("")
            print("std \n", metrics_std)
            
            
            # Creating Excel Writer Object from Pandas  
            path_outcome = "./files_to_load/" 
            excel_name = path_outcome +  "prediction_models.xlsx"
            
            try:
                os.remove(excel_name)
            except OSError:
                pass
            
            # writer = pd.ExcelWriter(excel_name, mode='w', engine='openpyxl')
            # # writer = pd.ExcelWriter(excel_name, mode='w', engine='openpyxl', if_sheet_exists='replace')
            
            with pd.ExcelWriter(excel_name, mode='w') as writer: 
                df_pred_models.to_excel(writer)
                df_metrics_sklearn.to_excel(writer, startcol=8)
                df_metrics_hold_out.to_excel(writer, startcol=16)
                                           
            # save excel 
            writer.save()          
        
    # Survival approach, totally different

  
    if parameters.survival_prediction:
        
        ## Cox survival 
        
        logrank_pvalues, logrank_stats = survival_pipeline(survival_train_test, X_train_test_processed, features_name, parameters.dict_params_Cox, n_loop = parameters.n_loop_survival, correction_method=parameters.correction_method, multivariate_significant=parameters.multivariate_significant, plot_survival=parameters.plot_survival)
        
        mean_pvalue = np.mean(logrank_pvalues)
        std_pvalue = np.std(logrank_pvalues)
        mean_stat = np.mean(logrank_stats)
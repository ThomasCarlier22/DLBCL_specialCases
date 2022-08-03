# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 08:35:45 2021

@author: GauthierFrecon
"""
import os 
import SimpleITK as sitk
import radiomics
import numpy as np
import matplotlib.pyplot as plt
from radiomics import featureextractor 
from radiomics import imageoperations #firstorder, getTestCase, glcm, glrlm, glszm, shape
import random as rnd
from pandas import DataFrame 
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, pearsonr
import pydicom
from mpl_toolkits.mplot3d import Axes3D
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import FeatureAgglomeration
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(1, '../')

#from data_importation_conversion import VG_path, systemModelPET, convertToSUV, valve_path




def radiomics_extraction(data_path, list_patients, settings, SUVconversion=True, resampling=True, verbosity=40, loop_textural_analysis=0):
    '''
    

    Parameters
    ----------
    data_path : string
        data path to "EI" folder
    list_patients : list (string)
        patient number for who compute radiomics
    settings : dict
        settings for the radiomics extractor of pyradiomics
    SUVconversion : Boolean, optional
        True if conversion to SUV (image multiply by SUV factor. The default is True.
    resampling : Boolean, optional
        True if th volume is resempled in 2*2*2 (original ~4*4*2). The default is True.
    verbosity : int, optional
        pyradiomics verbosity. The default is 40.
    loop_textural_analysis : int, optional
        iterations number of random feature (i.e pixels shuffling before computation) to calculate, if 0, no random feature computation. The default is 0.

    Returns
    -------
    rad_features: array (n_patients, n_features)
        array of features
    rad_features_random: array (n_patients, loop_textural_analysis, n_features)
        array of random features, inspiration of "Plausibility and redundancy analysis to select FDG-PET textural features in non-small cell lung cancer". Elisabeth Pfaehler
    features_names: list shape (n_features)
        names of features

    '''
    
    
    n_patients = len(list_patients) 
    radiomics.setVerbosity(verbosity)
    
    # features_extractor
    texturalFeatures = featureextractor.RadiomicsFeatureExtractor(**settings)
    texturalFeatures.disableAllFeatures()
    texturalFeatures.enableImageTypes(Original={}) #, LoG={}, Wavelet={})
    list_image_type = ("original") #, "log", "wavelet")
    texturalFeatures.enableFeaturesByName(firstorder=['Maximum', 'Mean', 'Energy'], glcm=['JointEntropy', 'Id', 'DifferenceAverage', 'JointAverage', 'DifferenceEntropy', 'ClusterShade'], glrlm=['HighGrayLevelRunEmphasis', 'LongRunHighGrayLevelEmphasis', 'RunLengthNonUniformity', 'ShortRunHighGrayLevelEmphasis', 'ShortRunLowGrayLevelEmphasis', 'LowGrayLevelRunEmphasis', 'GrayLevelNonUniformity'], glszm=['SizeZoneNonUniformity', 'SmallAreaEmphasis', 'LargeAreaEmphasis', 'ZonePercentage'], gldm=['DependenceNonUniformity','SmallDependenceHighGrayLevelEmphasis', 'LargeDependenceLowGrayLevelEmphasis', 'SmallDependenceEmphasis', 'LowGrayLevelEmphasis', 'LargeDependenceHighGrayLevelEmphasis', 'HighGrayLevelEmphasis', 'GrayLevelNonUniformity'], ngtdm=['Coarseness', 'Complexity', 'Strength'], shape=['MeshVolume', 'VoxelVolume', 'SurfaceArea', 'SurfaceVolumeRatio', 'Sphericity', 'Elongation', 'Flatness', 'Maximum3DDiameter', 'MajorAxisLength', 'MinorAxisLength'])
    enabled_features = texturalFeatures.enabledFeatures
    features_classes = [key for key, value in enabled_features.items()]
    features_names = [classes + "_" + name_feature  for classes, classes_value in enabled_features.items() for name_feature in classes_value]
    n_features = len(features_names)     
    rad_features = np.zeros((n_patients, n_features))
    rad_features_random = np.zeros((n_patients, loop_textural_analysis, n_features)) # needed only if textural analysis(loop_textural_analysios>0)
                                   
    list_suv_max = []   # debug or SUV analysis
    list_suv_factor=[]    
                  
    for i in range(n_patients):
        
        patient = list_patients[i]
        print("patient: ", patient)
        imagePath = data_path+"/"+patient+"/"+"patient.nii"
        petPath = data_path+"/"+patient+"/PT"
        labelPath = valve_path(data_path, patient)
        image = sitk.ReadImage(imagePath)
        
        #SUV conversion
        if SUVconversion:
            image, suv_factor = convertToSUV(petPath, image) # conversion to SUV
            list_suv_factor.append(suv_factor)
            
        newMask = sitk.ReadImage(labelPath) 
        newMask.SetSpacing(image.GetSpacing())
        newMask.SetOrigin(image.GetOrigin())
        
        #resampling 
        if resampling:
            image, newMask = imageoperations.resampleImage(image, newMask, **settings)
        origin =  image.GetOrigin()
        spacing = image.GetSpacing()        
               
  
        image_arr = sitk.GetArrayFromImage(image) 
        list_suv_max.append(np.max(image_arr))

        #radiomics computation
        result = texturalFeatures.execute(image, newMask, label=1) #feature extraction
        rad_features[i, :] = np.array([value for key, value in result.items() if key.startswith(list_image_type)])
     
        #random features computation    
        if loop_textural_analysis != 0:
            for k in range(loop_textural_analysis):
                print("__")
                image_arr = sitk.GetArrayFromImage(image)
                label_arr = sitk.GetArrayFromImage(newMask)
                pixels = image_arr[label_arr==1]
                rnd.shuffle(pixels)  #pixel shuffled before radiomics computation
                image_arr[label_arr==1] = pixels
                
                image = sitk.GetImageFromArray(image_arr)
                image.SetSpacing(spacing)
                image.SetOrigin(origin)       
                   
                result = texturalFeatures.execute(image, newMask, label=1)            
                rad_features_random[i, k, :] = np.array([value for key, value in result.items() if key.startswith(list_image_type)])
     
    features_names = [key for key, value in result.items() if key.startswith(list_image_type)]
        
    
    #return(list_suv_max, list_suv_factor) #  for  SUV analysis
    return(rad_features, rad_features_random, features_names) 
    



def textural_selection(rad_features, rad_features_random, features_name, treshold_textural_selection = 0.1, plot_textural_selection = False):
    '''
    textural selection, inspiration from "Plausibility and redundancy analysis to select FDG-PET textural features in non-small cell lung cancer". Elisabeth Pfaehler
    For each patient, and each feature calculate if non random feature is inside the 95 % confidence interval  [mean_feat - 1.96*std_feat, mean_feat + 1.96*std_feat] (mean and std_feat calculate on rad_features_random[patient, :, feature].
    The feature is to remove if she is out the confidence interval for  less than treshold_textural_selection of patients.
                                                                                                                                  

    Parameters
    ----------
    rad_features : array (n_patients, n_feautres)
        contains radiomics features
    rad_features_random : array shape (n_patients, loop_textural_analysis, n_features)
        contains all random features compute after that pixels inside segmentation has been shuffled randomly (iterations= loop_textural_analysis)
    features_name : list shape n_features
        contains names of features       
    treshold_textural_selection : float, optional
        DESCRIPTION. The default is 0.1.
    plot_textural_selection : boolean, optional
        Plot figure to show boxplot of random features versus the non random feature. The default is False.

    Returns
    -------
    mask_selection_textural: mask of boolean 
        True: we keep the feature

    '''

    n_patients, n_features = np.shape(rad_features)
    
    mean_feat = np.mean(rad_features_random, axis=1) #mean on random feature for each patient and each feature 
    std_feat = np.std(rad_features_random, axis=1) #std on random feature for each patient and each feature
    inf_bound = mean_feat - 1.96*std_feat #inf bound of the confidence interval for each patient and each feature
    sup_bound = mean_feat + 1.96*std_feat  #sup bound of the confidence interval for each patient and each feature
    
    bool_outside = (inf_bound > rad_features) | (sup_bound < rad_features) #or operator, is feature inside of the confidence interval for each patient and features ?
    features_to_remove = [name for name in features_name[np.mean(bool_outside, axis=0)<treshold_textural_selection] if not name.startswith(("shape", "firstorder"))] #selection if the proportion of patients for who the feature is inside the confidence interval <treshold, But we preserve shape features and first order features 
    ind_features_to_remove = [np.where(features_name == name)[0][0] for name in features_to_remove]
    mask_selection_textural = np.ones(n_features, dtype=bool)
    mask_selection_textural[ind_features_to_remove] = False
    
    
    #illustrate the selection
    if plot_textural_selection:
        ## Box plot random textural selection 
        num_patient = 3
        ind_feat = 16
        
        plt.figure()    
        plt.boxplot(rad_features_random[num_patient,:,ind_feat] , labels=[features_name[ind_feat]], showfliers=False)
        plt.plot([1], rad_features[num_patient,ind_feat], 'o', c="r", ms=5, label="non random features")
        plt.title("Boxplot random features ")
        plt.legend()
        plt.show() 
        
        for i in ind_features_to_remove:
            ind_feat = i
            plt.figure()    
            plt.boxplot(rad_features_random[num_patient,:,ind_feat] , labels=[features_name[ind_feat]], showfliers=False)
            plt.plot([1], rad_features[num_patient,ind_feat], 'o', c="r", ms=5, label="non random features")
            plt.title("Boxplot random features")
            plt.legend()
            plt.show() 
            
        print("features non informative for texture:\n", features_to_remove)
        print("random variances of this features:\n", std_feat[num_patient, ind_features_to_remove])

    return(mask_selection_textural)




def variance_selection(rad_features, features_name, treshold_var, plot_variance_selection):
    '''
    Finally never used in the pipeline
    

    Parameters
    ----------
    rad_features : array (n_patients, n_feautres)
        contains radiomics features
    features_name : list shape n_features
        contains names of features 
    treshold_var : float
        treshold for variance var/mean> treshold to delete
    plot_variance_selection : boolean
     variance barplot

    Returns
    -------
    mask_selection_variance: mask of boolean 
        True: we keep the feature
    '''
     
    n_patients, n_features = np.shape(rad_features)
    var_features = np.std(rad_features, axis=0)
    mean_features = np.mean(rad_features, axis=0)
    ratio_var_mean = var_features/mean_features 
    
    positions = [i for i in range(n_features)]
    
    if plot_variance_selection:
        fig = plt.figure()
        fig.suptitle('Features variances divided by features means' )
        ax = fig.add_subplot(111)
        plt.bar(positions, ratio_var_mean)
        ax.set_xticks(positions)
        ax.set_xticklabels(features_name)
        plt.xticks(rotation=90)
        plt.xlabel("Features")
        plt.ylim([0,100])
        plt.ylabel("ratio variances/means")
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.35)
        plt.show()
    
    mask_selection_variance =  ratio_var_mean> treshold_var # feature xhare var/mean> treshold to keep
    
    return(mask_selection_variance)




def spearman_analysis(X, features_name):
    '''
    compute the spearman correlation of the radiomics and plot correlation > 0.95

    Parameters
    ----------
    X : array (n_patients, n_features)
        radiomics array
    features_name : array 
        name of radiomics features

    Returns 
    -------
    None.

    '''
    n_patients, n_features = np.shape(X)
    corr_spearman, p_value = spearmanr(X)   
    print("mean absolute searman correlation:", np.mean(np.abs(corr_spearman)))
    positions = [i for i in range(n_features)]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #plt.imshow(np.abs(corr_spearman)>0.95)
    plt.imshow(np.abs(corr_spearman))
    ax.set_yticks(positions)
    ax.set_yticklabels(features_name)
    #plt.xticks(rotation=90)
    ax.set_xticks(positions)
    ax.set_xticklabels(features_name)
    plt.xticks(rotation=90)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.35)
    plt.colorbar()
    plt.title("absolute spearman correlation")
    
def reg_coef(x,y,label=None,color=None,**kwargs):
    '''
    Function to use inside correlation_analysis as input of sns.PairGrid
    calculate correlation and spearman correlation of two features x and y.

    Parameters
    ----------
    x : array shape n_patients
        feature vect
    y : array shape n_patients
        feature vect
    label : TYPE, optional
        classification, 1 presence of EI
    color : TYPE, optional
        DESCRIPTION. The default is None.
    **kwargs : array shape n_patients
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    ax = plt.gca()
    r,p = pearsonr(x,y)
    ax.annotate('r = {:.2f}'.format(r), xy=(0.5,0.55), xycoords='axes fraction', ha='center')
    r,p = spearmanr(x,y)
    ax.annotate('rs = {:.2f}'.format(r), xy=(0.5,0.45), xycoords='axes fraction', ha='center')
    ax.set_axis_off()


def correlation_analysis(X, y, features_name):
    '''
    Plot pair plot analysis for range of features (ind to ind+3). ind to adapt.

    Parameters
    ----------
    X :  float array (n_patients, n_features)
        features
    y : int array 
        classification, 1 positive to EI
    features_name : list string
        features names

    Returns
    -------
    None.

    '''
    
    ind=6 # 0 first order, 6 GLCM
    ind_end=ind+3 #
    
    n_patients, n_features = np.shape(X)
    
    corr = np.corrcoef(X, rowvar=False)
    
    #fig1, pairwise correlation image
    positions = [i for i in range(n_features)]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(np.abs(corr))
    plt.colorbar()
    ax.set_yticks(positions)
    ax.set_yticklabels(features_name)
    ax.set_xticks(positions)
    ax.set_xticklabels(features_name)
    plt.xticks(rotation=90)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.35)
    plt.title("absolute correlation coefficient")
    
    X_df = DataFrame(X,columns=features_name)
    corr = X_df.corr()
    
    #fig2, pairplot 1
    x = X[:,ind:ind_end]
    x_df = DataFrame(x,columns=features_name[ind:ind_end])
    x_df['diagnostic'] = y
    g = sns.PairGrid(x_df, hue="diagnostic")
    g.map_lower(sns.scatterplot)
    g.map_upper(sns.scatterplot)
    g.map_diag(sns.histplot)
    #g.map_upper(reg_coef)
    g.add_legend()
    g.fig.suptitle("Pair plot")
    
    #fig3 pari plot with regression coeff in the upper diag
    x = X[:,ind:ind_end]
    x_df = DataFrame(x,columns=features_name[ind:ind_end])
    g = sns.PairGrid(x_df) 
    g.map_diag(sns.histplot)
    g.map_lower(sns.regplot)
    g.map_upper(reg_coef)
    g.add_legend()
    g.fig.suptitle("Regression plot, and correlation coefficient")
    
    

def VIF_analysis(X, features_name):
    '''
    Plot VIF analysis of features X

    Parameters
    ----------
    X : features float array
        DESCRIPTION.
    features_name : features name list of string

    Returns
    -------
    VIF

    '''
    
    n_features = np.shape(X)[1]
    X_df = DataFrame(X, columns=features_name)
    X_df = X_df.assign(const=1)
    VIF =[variance_inflation_factor(X_df.values, i)  for i in range(X_df.shape[1])]
    
    positions = np.array(range(n_features))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(positions, VIF[0:-1], 'k--')
    ax.plot(positions, VIF[0:-1], 'None', marker='o',markerfacecolor='b')
    ax.set_xticks(positions)
    ax.set_xticklabels(features_name)
    plt.xticks(rotation=90)
    plt.subplots_adjust(top=0.960, bottom=0.40)
    plt.ylabel("VIF")
    plt.legend(loc='best')
    plt.title("VIF")
    plt.yscale('log')
    plt.show()    
    
    VIF_df = pd.Series(VIF, index=X_df.columns)
    print(VIF_df)
    return(VIF)


def correlation_selection(X, features_name, treshold=0.95):
    '''
     run among features and remove highly correlated features if feature not in feature_to_remove   

    Parameters
    ----------
    X : features float array
    features_name : features name list of string
    treshold :  treshold float
    
    Returns 
    -------
    mask: selection mask for correlation selection, 1 we keep the features, 0 we remove the feature

    '''
    
    n_patients, n_features = np.shape(X)
    f_keep = []
    corr_spearman, p_value = spearmanr(X)
    abs_corr = np.abs(corr_spearman)
    index_to_remove = []
    
    # n_highly_correlated = np.sum(abs_corr, axis=0)
    # index_max = np.argmax(n_highly_correlated)
    # f_keep.append(features_name[index_max])
    
    for i in range(n_features):
        if i not in index_to_remove: #if i in index_to_remove, analyze not to do because i already remove
            index_correlation = np.where(abs_corr[i, :]>treshold)[0] #index highly correlated
            print("\nfeatures correlation: ", features_name[i])
            print("highly correlated to: ", [features_name[k] for k in index_correlation])
            for ind in index_correlation:
                if ind!=i and ind not in index_to_remove:  #condition before removal
                    index_to_remove.append(ind)  
                    print("remove", features_name[ind])
    
    mask = np.ones(n_features, dtype=bool)
    mask[index_to_remove] = False
    
    return(mask)
    

def VIF_selection(X, X_test, features_name, treshold=10):  
    '''
    
    Remove features with the highest VIF while VIF > treshold
    apply same change to X_test
    
    Parameters
    ----------
    X : features float array
    X_test: features float array for test databse
    features_name : features name list of string
    treshold : VIF treshold float
    
    Returns 
    -------
    X features array selected, 
    features_name

    '''
    n_features = np.shape(X)[1]
    X_df = DataFrame(X, columns=features_name)
    X_df = X_df.assign(const=1)
    X_test_df = DataFrame(X_test, columns=features_name)
    X_test_df = X_test_df.assign(const=1)
    
    #initialization
    VIF = [variance_inflation_factor(X_df.values, i)  for i in range(X_df.shape[1])]
    ind_max = np.argmax(VIF)
    vif_max = VIF[ind_max]
    print("max VIF: ", (X_df.columns[ind_max], VIF[ind_max]))
    
    while vif_max>treshold:  #VIF >treshold indicate multicolinearity
        print("drop: ",X_df.columns[ind_max])
        X_df = X_df.drop(X_df.columns[ind_max], axis=1) #remove features
        X_test_df = X_test_df.drop(X_test_df.columns[ind_max], axis=1) #remove features
        #update VIF
        VIF =  [variance_inflation_factor(X_df.values, i)  for i in range(X_df.shape[1])] 
        ind_max = np.argmax(VIF)
        vif_max = VIF[ind_max]
        print("max VIF: ", (X_df.columns[ind_max], VIF[ind_max]))
        
    print("VIF<10")     
           
    features_name = X_df.columns        
    X = X_df.to_numpy()   
    X_test = X_test_df.to_numpy()      
    return(X, X_test, features_name)





def spearman_affinity(X):
    
   corr_spearman, p_value = spearmanr(X.T)
   
   return 1 - np.abs(corr_spearman)


def pooling_func_medoid(X, axis=1):
    
    n_features = np.shape(X)[1] 
    if n_features<=2:  # one or two features
        index_medoid=0 # medoid = first features

    else:
        corr_spearman, p_value = spearmanr(X)
        distMatrix = 1 - np.abs(corr_spearman)
        index_medoid = np.argmin(distMatrix.sum(axis=axis))
    medoid = X[:, index_medoid]
    return medoid

def spearman_dist_medoid(X, features_name_cluster, axis=1):
    
    n_features = np.shape(X)[1] 
    if n_features<=2:  # one or two features
        index_medoid=0 # medoid = first features

    else:
        corr_spearman, p_value = spearmanr(X)
        distMatrix = 1 - np.abs(corr_spearman)
        index_medoid = np.argmin(distMatrix.sum(axis=axis))
    medoid = X[:, index_medoid]
    feature_name  = features_name_cluster[index_medoid]
    return medoid, feature_name 

def medoid_transform(cluster, features, features_name):
    
    n_patients, n_features = np.shape(features)
    labels = cluster.labels_
    n_clusters = cluster.n_clusters_
    features_transform = np.zeros((n_patients, n_clusters))
    features_name_transform = np.zeros(n_clusters, dtype="object")
    
    for k in range(n_clusters):
        cluster_k = features[:, labels==k]
        features_name_cluster_k = features_name[labels==k]
        medoid_k, feature_k = spearman_dist_medoid(cluster_k, features_name_cluster_k)
        features_transform[:, k] = medoid_k
        features_name_transform[k] = feature_k
          
    return(features_transform, features_name_transform)





def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    plt.figure()
    plt.title("Hierarchical Clustering Dendrogram")
    dendrogram(linkage_matrix,  **kwargs) 
    plt.subplots_adjust(bottom=0.31)
    plt.ylabel("Correlation distance")
    plt.xlabel("Features names or number of features in node.")
    plt.show()





def PCA_analysis(X, y, features_name, pca_misclassification, list_patients, patients_misclassified=[], sampling = False, X_smote=[], y_smote=[], pca_nosuspicion=False, mask_nosuspicion=[], X_test=[], y_test=[], test_analysis=False):
    '''
    PCA analysis of the radiomics. and PCA projections. feature plot (to set True manually)

    Parameters
    ----------
    X : array shape:(n_patients, n_features)
        radiomics
    y : array shape:(n_patients,)
        labels of patients
    features_name : list of string
        features name 
    pca_misclassification : bool
       bool = true if analysis of patients misclassified
    list_patients : list or array
        patients names (int)
    patients_misclassified : list or array, optional
        patients misclassified. The default is [].
    sampling : bool, optional
        True if we plot points of data augmentation. The default is False.
    X_smote : array, optional
        data augmenatatin points. The default is [].
    y_smote : list or array, optional
        data augmentation points labelization. The default is [].
    pca_nosuspicion: bool
        True: highlight nosuspicion patients in pca plots
    mask_nosuspicion: array of bool
        True: nosuspicion patients => include in valid set, False: nosuspicion patients => exclude of valid set
    X_test : array shape:(n_patients, n_features)
        radiomics test database
    y_test : array shape:(n_patients,)
        labels of patients test database
    test_analysis: bool
        apply or not analysis with test database (to implement)
        
    Returns
    -------
    projections of radiomics on the n_PCA (code brut!) directions with highest variance

    '''
    n_PCA = 6
    feature_space_plot = False #hard coded plot end of the function
    
    n_patients, n_features = np.shape(X)
    
    pca = PCA()
    pca.fit(X)
    print("pca.explained_variance_ratio_:", pca.explained_variance_ratio_[0:n_PCA])
    print("explained variance with 6 directions:, ", np.sum(pca.explained_variance_ratio_[0:6]))
    plt.figure()
    plt.bar(range(n_PCA), pca.explained_variance_ratio_[0:n_PCA])
    plt.xlabel("PCA components")
    plt.ylabel("Variance ratio")
    plt.title("PCA explained variance ratio")
    plt.show()
    
    if sampling:
        X, y = X_smote, y_smote
    
    rad_features_pca =  pca.transform(X) # transformer (prediction) les données features_nor pour le modele PCA
    rad_features_pca_test = pca.transform(X_test)
    
    ACP0 = rad_features_pca[:,0] # extraction premiere composante
    ACP1 = rad_features_pca[:,1] # extraction deuxieme composante
    ACP2 = rad_features_pca[:,2] # extraction troisième composante
    
    ACP0_test = rad_features_pca_test[:,0] # extraction premiere composante
    ACP1_test = rad_features_pca_test[:,1] # extraction deuxieme composante
    ACP2_test = rad_features_pca_test[:,2] # extraction troisième composante    
    
    print(ACP0.shape)
    
    classif_names = ["Negative IE", "Positive IE"]
    
    #plot 2D 
    plt.figure(figsize=(12, 12))
    for c, i, clf in zip("rb", range(0,2), range(0,2)):
        plt.scatter(ACP0[y==i], ACP1[y==i], c=c, label=classif_names[clf])
        
    if test_analysis:
         plt.scatter(ACP0_test, ACP1_test, c="g", label="Test samples")
            
    if pca_misclassification:
        for patient in patients_misclassified:
            list_patients_int = [int(patient) for patient in list_patients] 
            pos = list_patients_int.index(patient)
            plt.scatter(ACP0[pos], ACP1[pos], s=600, facecolors='none', edgecolors='black')
    
    if pca_nosuspicion:
        plt.scatter(ACP0[~mask_nosuspicion], ACP1[~mask_nosuspicion], c="r", edgecolors='black', label="Negative IE No suspicion")
        
    for k in range(n_patients):
        plt.annotate(list_patients[k], (ACP0[k]+0.1, ACP1[k]+0.1)) 
    plt.legend()
    plt.xlabel("PCA0")
    plt.ylabel("PCA1")
    plt.title("Patients Points projection on the two highest variance directions")
    
    
    #plot 3D 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for c, i, clf in zip("rb", range(0,2), range(0,2)):
        ax.scatter3D(ACP0[y==i], ACP1[y==i], ACP2[y==i],  c=c, label=classif_names[clf])
    
    if test_analysis:
        ax.scatter3D(ACP0_test, ACP1_test, ACP2_test,  c="g", label="Test samples")
    
    if pca_misclassification:
        for patient in patients_misclassified:
            list_patients_int = [int(patient) for patient in list_patients] 
            for pos in [list_patients_int.index(patient)]:       
                ax.scatter3D(ACP0[pos], ACP1[pos], ACP2[pos],  '*', edgecolors='black', label='misclassified') 
                ax.text(ACP0[pos]+0.1, ACP1[pos]+0.1, ACP2[pos]+0.1, list_patients[pos])
    if pca_nosuspicion:
        ax.scatter3D(ACP0[~mask_nosuspicion], ACP1[~mask_nosuspicion], ACP2[~mask_nosuspicion], c="r", edgecolors='black', label="Negative IE nosuspicion")
    
    plt.legend() 
    plt.xlabel("PCA0")
    plt.ylabel("PCA1")  
    plt.title("Patients Points projection on the three highest variance directions")
    
    
    ## other plot features space no PCA
    if feature_space_plot:
        classif_names = ["Negative EI", "Positive EI"]
    
        #plot 1D 
        positions =[0]
        xlabel = [features_name[0]]
        fig=plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111)
        
        for c, i, clf in zip("rb", range(0,2), range(0,2)):
            value = X[y==i,0]
            pos = np.zeros(np.shape(value)[0])
            plt.plot(pos, value, 'o', c=c,  ms=2.5, label=classif_names[clf])
        ax.set_xticks(positions)
        ax.set_xticklabels(xlabel)
        plt.xticks(rotation=70)
        plt.subplots_adjust(top=0.960, bottom=0.225)    
        plt.ylabel("Value")    
        plt.title("features space")
        plt.legend()
        plt.show()
        
        #plot 2D feature space 2 feature space
        fig=plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111)
        for c, i, clf in zip("rb", range(0,2), range(0,2)):
            value1 = X[y==i,0]
            value2 = X[y==i,1]
            pos = np.zeros(np.shape(value)[0])
            plt.plot(value1, value2, 'o', c=c,  ms=2.5, label=classif_names[clf])
        plt.subplots_adjust(top=0.960, bottom=0.225)    
        plt.ylabel(features_name[0])    
        plt.xlabel(features_name[1])
        plt.title("features space")
        plt.legend()
        plt.show()
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for c, i, clf in zip("rb", range(0,2), range(0,2)):
            value1 = X[y==i,0]
            value2 = X[y==i,1]
            pos = np.zeros(np.shape(value)[0])
            value3 = X[y==i,-1]
            ax.scatter3D(value1, value2, value3, c=c, label=classif_names[clf])
        plt.subplots_adjust(top=0.960, bottom=0.225)    
        plt.ylabel(features_name[0])    
        plt.xlabel(features_name[1])
        plt.title("features space")
        plt.legend()
        plt.show()

    return(rad_features_pca[:,0:n_PCA], rad_features_pca_test[:,0:n_PCA])
      
def radiomics_pipeline(data_path, list_patients, y, parameters, clinical_features, clinical_features_name, test_path, list_patients_test, y_test, clinical_features_test):
    
    
    '''
    return pyradiomics features selected, and processed w.r.t parameters, return also features names 
    clinical features are concatenate with radiomics features if parameters.compute_clinical_features=True


    Parameters
    ----------
    data_path : string
        Path to the database ("EI" folder) 
    list_patients: list
        patients 
    y: array int
        0 negative EI, 1 positive EI    
    parameters : class parameters
        parameters of the radiomics extraction
    radiomics: array of int n_patients*n_clinical_features
        array of clinical features data convert from excel data
    clinical_features_name: float np array
        name of clinical features
    test_path: string
        Path to the test database
    list_patients_test:list
        test patients
    y_test: array int
        0 negative EI, 1 positive EI    
    clinical_features_test: float np array
        array of clinical features data convert from excel data for test database
    
    Returns 
    -------
    X array
        radiomics (n_patients, n_features). Extracted, selected and then normalized with respect to parameters options

    features_names list
        name of features

    '''
    
    # rad_features_test, rad_features_random_test, features_name_test = radiomics_extraction(test_path, list_patients_test, parameters.settings_pyradiomics, parameters.SUVconversion, parameters.resampling, parameters.verbosity, loop_textural_analysis=0)  ## rad_fdeatures_rand = np.zeros array
    # np.save("./files_to_load/rad_features_test" + parameters.save_name + ".npy", rad_features_test)
    
    
    ## radiomics computation and random radiomics computation
    if parameters.features_computation and parameters.textural_computation: 
        print("compute features and random features")
        rad_features, rad_features_random, features_name = radiomics_extraction(data_path, list_patients, parameters.settings_pyradiomics, parameters.SUVconversion, parameters.resampling, parameters.verbosity, loop_textural_analysis=parameters.loop_textural_analysis)
        if parameters.save_computation:
            np.save("./files_to_load/rad_features" + parameters.save_name + ".npy", rad_features)
            np.save("./files_to_load/features_name.npy", features_name)
            #np.save("./rad_features_random_suv.npy", rad_features_random)
        

    # radiomics computation           
    elif not(parameters.textural_computation) and parameters.features_computation:
        print("compute features")
        rad_features, rad_features_random, features_name = radiomics_extraction(data_path, list_patients, parameters.settings_pyradiomics, parameters.SUVconversion, parameters.resampling, parameters.verbosity, loop_textural_analysis=0)  ## rad_fdeatures_rand = np.zeros array
        rad_features_test, rad_features_random_test, features_name_test = radiomics_extraction(test_path, list_patients_test, parameters.settings_pyradiomics, parameters.SUVconversion, parameters.resampling, parameters.verbosity, loop_textural_analysis=0)  ## rad_fdeatures_rand = np.zeros array
        if parameters.save_computation:
            np.save("./files_to_load/rad_features" + parameters.save_name + ".npy", rad_features)
            np.save("./files_to_load/features_name" + parameters.save_name + ".npy", features_name)
            np.save("./files_to_load/rad_features_test" + parameters.save_name + ".npy", rad_features_test)
    
    #download radiomics and random radiomics to avoid computation time
    else:
        print("download features")
        if parameters.SUVconversion:
            rad_features = np.load("./files_to_load/rad_features" + parameters.load_name + ".npy")
            rad_features_test = np.load("./files_to_load/rad_features_test" + parameters.load_name + ".npy")
            #rad_features_random = np.load("./rad_features_random.npy")
            features_name = np.load("./files_to_load/features_name" + parameters.load_name + ".npy")
            
        else:
            print("!!!! radiomics without SUV TO compute !!!!!")
            # rad_features = np.load("./rad_features.npy")
            # rad_features_random = np.load("./rad_features_random.npy")
            # features_name = np.load("./features_name.npy")
    features_name = [name[9:len(name)+1] for name in features_name] #remove "original_"        
    n_patients, n_features = np.shape(rad_features)
    features_name = np.array(features_name)


    ## textural selection, inspiration from pfalher paper
    mask_selection_textural = np.ones(n_features, dtype=bool)
    if parameters.textural_selection:
        #automatization:
        #mask_selection_textural = textural_selection(rad_features, rad_features_random, features_name, parameters.treshold_textural_selection, parameters.plot_textural_selection)
        
        #Manually: (we skip the computation of this step, to avoid computation)
        features_to_remove = ['gldm_LowGrayLevelEmphasis', 'gldm_HighGrayLevelEmphasis', 'gldm_GrayLevelNonUniformity']
        ind_textural = [i for i in range(n_features) if features_name[i] in features_to_remove]
        mask_selection_textural[ind_textural] = False
    
    ## variance selection, not to do in majority of cases
    mask_selection_variance = np.ones(n_features, dtype=bool)
    if parameters.variance_selection:
        mask_selection_variance = variance_selection(rad_features, features_name, parameters.threshold_var, parameters.plot_variance_selection)
        
    ## shape selection    
    mask_selection_shape = np.ones(n_features, dtype=bool)
    if parameters.shape_selection:
        ind_shape = [i for i in range(n_features) if features_name[i].startswith("shape")] #and features_name[i]!='shape_VoxelVolume' 
        mask_selection_shape[ind_shape] = False

    ## combination of shape, textural (and variance selection) selection (finally only textural+shape to remove)
    mask_selection = mask_selection_shape & mask_selection_textural  & mask_selection_variance 
    rad_features = rad_features[:, mask_selection]
    rad_features_test = rad_features_test[:, mask_selection]
    features_name = features_name[ mask_selection]
    n_features = len(features_name)
    
    
    #correlation analyis
    if parameters.correlation_analysis:
        correlation_analysis(rad_features, y, features_name)
        
    ## spearman analysis
    if parameters.spearman_analysis:
        spearman_analysis(rad_features, features_name)

    ## VIF
    if parameters.VIF_analysis:
        VIF_analysis(rad_features, features_name)
    
    ## correlation selection come after precedent selection
    if parameters.correlation_selection:
        mask_selection_correlation = correlation_selection(rad_features, features_name, parameters.correlation_treshold)
        rad_features = rad_features[:, mask_selection_correlation]
        rad_features_test = rad_features_test[:, mask_selection_correlation]
        features_name = features_name[ mask_selection_correlation]
        n_features = len(features_name)
        
        
    if parameters.VIF_selection:     
        rad_features, rad_features_test, features_name = VIF_selection(rad_features, rad_features_test, features_name, treshold=parameters.VIF_treshold)
        n_features = len(features_name)
        
    ## VIF after correlation selection
    if parameters.VIF_analysis:
        VIF_analysis(rad_features, features_name)
        
    if parameters.spearman_analysis:
        spearman_analysis(rad_features, features_name)   
    
    #include or not clinical features
    if parameters.add_clinical_features:
        X = np.concatenate((rad_features, clinical_features), axis=1) #concatenation features, patients in axis 0
        X_test = np.concatenate((rad_features_test, clinical_features_test), axis=1) #concatenation features, patients in axis 0
        features_name = np.concatenate((features_name, clinical_features_name), axis=0) #concatenate features _name
        
    else:
        X = rad_features
        X_test = rad_features_test
      
    #VIF analysis with others features
    if parameters.add_clinical_features and parameters.clinical_features_analysis:
        VIF_analysis(X[:, 0:-3], features_name[0:-3])   #### Hard coded to remove categorical variable###### !!!
        spearman_analysis(X[:, 0:-3], features_name[0:-3])   #### Hard coded to remove categorical variable###### !!!
      
    ## Cholesky decorrelation: to have mahalanobis distance inside SVM
    if parameters.cholesky_decorrelation:
        cov= np.cov(X, rowvar=False) 
        p = np.shape(X)[1]
        eye_matrix =  1e-06*np.eye(p)  # to brute force definite piositive matrix
        L = np.linalg.cholesky(cov + eye_matrix)
        L_inv = np.linalg.inv(L)
        X_mahalanobis_T = np.matmul(L_inv, X.T)
        X_mahalanobis = X_mahalanobis_T.T 
        X = X_mahalanobis
        
        #same change to X_test
        X_test_mahalanobis_T = np.matmul(L_inv, X_test.T)
        X_test_mahalanobis = X_test_mahalanobis_T.T 
        X_test = X_test_mahalanobis
 
    ## Normalization
    if parameters.standardization:
        scaler = StandardScaler()
        X_concat = np.concatenate((X, X_test))
        scaler.fit(X_concat)
        X_scale = scaler.transform(X)  
        X = X_scale          
        X_test_scale = scaler.transform(X_test)  
        X_test = X_test_scale                                 

    ## pca analysis
    if parameters.pca_analysis:
        rad_features_pca, rad_features_pca_test = PCA_analysis(X, y, features_name, False, list_patients, X_test=X_test, y_test=y_test, test_analysis=parameters.test_analysis_pca) 
        if parameters.pca_replacement:
            X = rad_features_pca
            X_test = rad_features_pca_test
     
        
     
        
    return(X, X_test, features_name)    



def processing_features_special_cases(X, y, features_name, parameters):
    """
    clustering + standardization +split

    Parameters
    ----------
    X : features n_patients*n_features
    y : outcome
    features_name : name
    parameters : see radiomics_apoproach

    Returns
    -------
    None.

    """
    
    
    # hierarchical_clustering
    if parameters.hierarchical_clustering:
        
        cluster = FeatureAgglomeration(n_clusters=None,  linkage='average', affinity=spearman_affinity, distance_threshold=1-parameters.correlation_threshold, pooling_func=pooling_func_medoid)
        cluster.fit(X)
        if parameters.dendro_plot:
            plot_dendrogram(cluster, p=parameters.p_dendrogram, truncate_mode="level", color_threshold=1-parameters.correlation_threshold, labels = features_name, no_labels= parameters.no_labels, leaf_font_size=parameters.leaf_font_size)
        
        X, features_name = medoid_transform(cluster, X, features_name)
        
        n_clusters = cluster.n_clusters_
        print("n clusters: ", n_clusters)
        
    # standardization    
    scaler = StandardScaler()
    scaler.fit(X)
    X_scale = scaler.transform(X)  
    X_processed = X_scale     

    # split 
    
    X_train, X_test, y_train, y_test = train_test_split(X_processed , y, test_size=parameters.test_size, stratify=y, random_state=0)
    
    return  X_processed, X_train, X_test, y_train, y_test, features_name
    


if __name__ == '__main__':
    
    data_path = "D:\TFE_CHU_data\EI"
    list_patients = os.listdir(data_path) # ["1","10"] 
    patients_excluded = ["23", "26", "46", "49", "50", "59"] # remove patients with problems 
    list_patients = [patient for patient in list_patients if patient not in patients_excluded] 
    list_patients_int = [int(patient) for patient in list_patients] 
    list_classification = [patient<=46 for patient in list_patients_int] 
    y = np.array(list_classification, dtype=int)
    
    n_patients = len(list_patients)
    verbosity = 40
    settings = {}
    settings['binCount'] = 64
    #settings['binWidth'] = 0.3
    #settings['resampledPixelSpacing'] = [2/voxel3D[2],2/voxel3D[0],2/voxel3D[1]]  # OK Verified 19112020
    settings['resampledPixelSpacing'] = [2,2,2]
    settings['interpolator'] = sitk.sitkBSpline
    settings['weightingNorm'] = 'no_weighting'
    settings['sigma'] = [2, 6]

    
    #list_patients = ["1", "10", "11"]
    
    
    list_suv_max, list_suv_factor = radiomics_extraction(data_path, list_patients, settings, verbosity, loop_textural_analysis=0)
    
    
    #rad_features = radiomics_extraction(data_path, list_patients, settings, verbosity, loop_textural_analysis=0)
    
   

    
        

    
    
    


# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 10:19:17 2021

@author: GFRECON
"""

import numpy as np
from scipy.stats import spearmanr, pearsonr, mannwhitneyu
import matplotlib.pyplot as plt


metrics_valid_exclud = np.load('./files_to_load/metrics_models/valid_exclude.npy')
metrics_nosuspicion_removed = np.load('./files_to_load/metrics_models/no_suspicion_removed.npy')
metrics_with_clinical_features = np.load('./files_to_load/metrics_models/with_clinical_features.npy')
        
pvalue_arr = np.zeros(6)
for k in range(6):
    (stat, pvalue) = mannwhitneyu( metrics_nosuspicion_removed[3, :, k], metrics_valid_exclud[3, :, k])
    pvalue_arr[k] = pvalue

pvalue_arr = np.zeros((4, 4))

#model vs model, evaluation fixed
for i in range(4):
    for j in range(4):
        (stat, pvalue) = mannwhitneyu( metrics_with_clinical_features[i, :, 5], metrics_with_clinical_features[j, :, 5], alternative='greater')
        pvalue_arr[i, j] = pvalue

pvalue_arr = np.zeros((4,6))
#evaluation vs evaluation, model fixed
for i in range(4):
    for m in range(6):
        (stat, pvalue) = mannwhitneyu(metrics_valid_exclud[i, :, m], metrics_nosuspicion_removed[i, :, m], alternative='greater' )
        pvalue_arr[i, m] = pvalue
        
        
        




Name = ["LR ridge - Radiomics", "LR ridge - Radiomics+clinical features", "LR lasso - Radiomics", "LR lasso - Radiomics+clinical features", "SVC - Radiomics", "SVC- Radiomics+clinical features", "RF - Radiomics", "RF - Radiomics+clinical features"]
metrics_with_without_clinical = np.zeros((8, 20, 6))
for i in range(4):
    metrics_with_without_clinical[2*i, :, :] = metrics_nosuspicion_removed[i, : ,:]
    metrics_with_without_clinical[2*i+1, :, :] = metrics_with_clinical_features[i, : , :]
    

positions = range(8)
# AUC
fig = plt.figure()
fig.suptitle('Models Comparison - AUC' )
ax = fig.add_subplot(111)
bp0 = ax.boxplot(metrics_with_without_clinical[:,:,5].T, positions=positions, patch_artist=True, showfliers=False)
plt.setp(bp0['boxes'], color='darkorange', facecolor='moccasin')
plt.setp(bp0['medians'], color='darkorange', linewidth=2)
plt.setp(bp0['whiskers'], linestyle=':')
plt.setp(bp0['caps'], color='darkorange')
ax.set_xticklabels(Name)
plt.legend([bp0["boxes"][0]], ["AUC"])
plt.xticks(rotation=70)
plt.subplots_adjust(top=0.960, bottom=0.45)
plt.xlabel("Models")
plt.ylabel("AUC")
plt.show()

positions = range(8)
# Acc
fig = plt.figure()
fig.suptitle('Models Comparison - Accuracy' )
ax = fig.add_subplot(111)
bp0 = ax.boxplot(metrics_with_without_clinical[:,:,0].T, positions=positions, patch_artist=True, showfliers=False)
plt.setp(bp0['boxes'], color='darkorange', facecolor='moccasin')
plt.setp(bp0['medians'], color='darkorange', linewidth=2)
plt.setp(bp0['whiskers'], linestyle=':')
plt.setp(bp0['caps'], color='darkorange')
ax.set_xticklabels(Name)
plt.legend([bp0["boxes"][0]], ["Accuracy"])
plt.xticks(rotation=70)
plt.subplots_adjust(top=0.960, bottom=0.45)
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.show()
        
        



Name = ["LR ridge - no suspicion patients in training", "LR ridge - no suspicion patients removed", "LR lasso - no suspicion patients in training", "LR lasso - no suspicion patients removed", "SVC - no suspicion patients in training", "SVC - no suspicion patients removed", "RF - no suspicion patients in training", "RF - no suspicion patients removed"]
metrics_exclude_removed = np.zeros((8, 20, 6))
for i in range(4):
    metrics_exclude_removed[2*i, :, :] = metrics_valid_exclud[i, : ,:]
    metrics_exclude_removed[2*i+1, :, :] = metrics_nosuspicion_removed[i, : ,:]
    

positions = range(8)
# AUC
fig = plt.figure()
fig.suptitle('Models Comparison - AUC' )
ax = fig.add_subplot(111)
bp0 = ax.boxplot(metrics_exclude_removed[:,:,5].T, positions=positions, patch_artist=True, showfliers=False)
plt.setp(bp0['boxes'], color='darkorange', facecolor='moccasin')
plt.setp(bp0['medians'], color='darkorange', linewidth=2)
plt.setp(bp0['whiskers'], linestyle=':')
plt.setp(bp0['caps'], color='darkorange')
ax.set_xticklabels(Name)
plt.legend([bp0["boxes"][0]], ["AUC"])
plt.xticks(rotation=70)
plt.subplots_adjust(top=0.960, bottom=0.45)
plt.xlabel("Models")
plt.ylabel("AUC")
plt.show()     

positions = range(8)
# Acc
fig = plt.figure()
fig.suptitle('Models Comparison - Accuracy' )
ax = fig.add_subplot(111)
bp0 = ax.boxplot(metrics_exclude_removed[:,:,0].T, positions=positions, patch_artist=True, showfliers=False)
plt.setp(bp0['boxes'], color='darkorange', facecolor='moccasin')
plt.setp(bp0['medians'], color='darkorange', linewidth=2)
plt.setp(bp0['whiskers'], linestyle=':')
plt.setp(bp0['caps'], color='darkorange')
ax.set_xticklabels(Name)
plt.legend([bp0["boxes"][0]], ["Accuracy"])
plt.xticks(rotation=70)
plt.subplots_adjust(top=0.960, bottom=0.45)
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.show()     
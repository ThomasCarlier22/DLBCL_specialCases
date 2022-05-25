# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 17:32:14 2022

@author: Fero
"""
import matplotlib.pyplot as plt
# import matplotlib.table as tbl
import numpy as np
from sklearn import metrics
import pandas as pd
from scipy import stats
from lifelines.statistics import logrank_test

from lifelines.fitters.coxph_fitter import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from lifelines import KaplanMeierFitter
from lifelines import NelsonAalenFitter


############################################### FONCTIONS ############################################## 
def sensivity_specifity_cutoff(y_true, y_score):
    
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score) # 1-spécificités, sensibilités, seuils
    idx = np.argmax(tpr - fpr)
    return  tpr[idx],  1-fpr[idx], thresholds[idx]

def roc_plots():
    plt.clf()
    plt.title('delta_SUV ROC curve for predicting 2y-PFS.')
    plt.plot([0, 1], [0, 1],'r--', label='Première bissectrice (AUC=0.50)')
    plt.plot(fpr, tpr, 'b', label = ('AUC = %0.2f' % roc_auc , 'cutoff = %0.2f ' % cut_off))
    plt.legend(loc = 'best')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('sensibilité')
    plt.xlabel('1-spécificité')
    plt.show()
    #plt.savefig('delta_SUV_PFS_ROC.pdf') #
    return plt

def case():
    a=int(input("Entrez (1) pour selectionner les CAS SPECIAUX, (2) pour les CAS NON SPECIAUX  et (3) pour ne pas différencier: "))
    if a==1:
        donnes= Complet_result.query("Special_Case == 1")
    elif a==2:
        donnes= Complet_result.query("Special_Case == 2")
    else:
        donnes= Complet_result
    return donnes
    
    

########################################## DATAS ################################################
filename ="Complet_result.xlsx"; 
Complet_result=pd.read_excel(filename)
donnes=case()  

PFS_censoring=donnes["PFS censoring"]            #
delta_SUV=donnes["delta_SUV_PET4_adjudication"]


########################################## ROC CURVES PLOT ################################################    
y = np.array(PFS_censoring)
pred = np.array(delta_SUV)
fpr, tpr, thresholds = metrics.roc_curve(y, pred)
roc_auc = metrics.auc(fpr, tpr)
sensitivity, specificity, cut_off = sensivity_specifity_cutoff(y,pred)

roc_plots()
print('seuil:',cut_off )
print('sensibilité:',sensitivity) #
print('spécificité:',specificity) #

############################################ PATIENTS GROUPS DEFINITION ####################################
patients_a_risque= donnes[donnes["delta_SUV_PET4_adjudication"]<cut_off] #
patients_sans_risque= donnes[donnes["delta_SUV_PET4_adjudication"]>=cut_off] #
patients_a_risque.to_excel("patients_a_risque_delta_SUV.xlsx")
patients_sans_risque.to_excel("patients_sans_risque_delta_SUV.xlsx")

########################################### PFS BOXPLOTS #######################################################   
# box_data= [patients_a_risque["PFS (months)"], patients_sans_risque["PFS (months)"]]
# plt.subplot(121)
# plt.boxplot(box_data[0])
# plt.title("PFS des patients à risque")
# plt.ylabel(' PFS (en mois)')
# plt.subplot(122)
# plt.boxplot(box_data[1])
# plt.title("PFS des patients sans risque")
# plt.ylabel('')
# #plt.savefig("delta_SUV_PFS_Boxes.pdf")
# plt.show()

#################################################### STATISTICS ON BOXPLOT DATAS ###############################
#print("patients à risque:  ", pd.DataFrame(box_data[0]).describe())
#print("patients sans risque:  ", pd.DataFrame(box_data[1]).describe())
# lv_test =stats.levene(box_data[0],box_data[1], center='median', proportiontocut=0.05)#test de Levene
# t_test =stats.ttest_ind(box_data[0],box_data[1], equal_var=False, alternative="less") #Ttes: hypothèse des variances inégales
# mw_test =stats.mannwhitneyu(box_data[0],box_data[1], alternative="less")#test de Man Whitney
# ks_test =stats.ks_2samp(box_data[0],box_data[1])#Test de Kolmogorov Smirnov
# print("PFS: ", mw_test)

#################################################### PROGRESSION ANALYSIS (survival fonctions) ##############################################
kmf_par= KaplanMeierFitter()
kmf_psr= KaplanMeierFitter()
kmf_par.fit(durations=patients_a_risque["PFS (months)"], event_observed=patients_a_risque["uns"])
kmf_psr.fit(durations=patients_sans_risque["PFS (months)"], event_observed=patients_sans_risque["uns"])
plt.clf()
plt.title("Kaplan-Meieir estimate of progression")
plt.xlabel("time (months)")
plt.ylabel("non-progression fraction")
kmf_par.plot(label="Patients à risque")
kmf_psr.plot(label="patients sans risque")
plt.show()

##############################################################" LOG RANK TEST ################################################################
print("LOG RANK TEST")
LR_test = logrank_test(patients_a_risque["PFS (months)"], patients_sans_risque["PFS (months)"], event_observed_A=patients_a_risque["uns"], event_observed_B=patients_sans_risque["uns"])
print(LR_test.print_summary())



############################################################ COX REGRESSION ANALYSIS ##########################################################
print("REGRESSION DE COX")
data=donnes[["TMTV_PET0_Vote_Maj", "SUV_max_PET0_Vote_Maj", "TLG_PET0_Vote_Maj",  "Dmax_PET0_Vote_Maj" ,"sex_label", "Treatment_arm_label", "delta_SUV_PET4_adjudication", "Performance Status (ECOG scale)", "Age (years)", "Ann Arbor Stage", "IPI", "LDH (IU/L)", "PFS (months)", "uns"]]
# data=patients_sans_risque[[ "sex_label", "Treatment_arm_label", "delta_SUV_PET4_adjudication", "Performance Status (ECOG scale)", "Age (years)", "Ann Arbor Stage", "IPI", "LDH (IU/L)", "PFS (months)", "uns"]]
cph=CoxPHFitter()
cph.fit(data, "PFS (months)", event_col="uns")
cph.plot()
plt.show()
cph.print_summary()




############################################# COMPARAISON DE LA PROGRESSION DES CAS SPECIAUX ET DES NON CAS SPECIAUX  #################################


if len(donnes)==len(Complet_result):
    print("COMPARAISON DE LA PROGRESSION DES CAS SPECIAUX ET DES NON CAS SPECIAUX")
    
    special_patient= donnes.query("Special_Case == 1")
    non_special_patient= donnes.query("Special_Case == 2")

    kmf_spec= KaplanMeierFitter()
    kmf_non_spec= KaplanMeierFitter()
    kmf_spec.fit(durations=special_patient["PFS (months)"], event_observed=special_patient["uns"])
    kmf_non_spec.fit(durations=non_special_patient["PFS (months)"], event_observed=non_special_patient["uns"])
    plt.clf()
    plt.title("Kaplan-Meieir estimate of progression")
    plt.xlabel("time (months)")
    plt.ylabel("non-progression fraction")
    kmf_spec.plot(label="Special Patient")
    kmf_non_spec.plot(label="Non Special patient")
    plt.show()

    ##############################################################" LOG RANK TEST ################################################################
    print("LOG RANK TEST")
    LR = logrank_test(special_patient["PFS (months)"], non_special_patient["PFS (months)"], event_observed_A=special_patient["uns"], event_observed_B=non_special_patient["uns"])
    print(LR.print_summary())




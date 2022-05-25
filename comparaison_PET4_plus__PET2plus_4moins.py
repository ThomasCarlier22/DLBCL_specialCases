# -*- coding: utf-8 -*-
"""
Created on Mon May 23 10:08:26 2022

@author: Fero
"""

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

    


def case():
    a=int(input("Patients PET2+-->PET4-: Entrez (1) pour selectionner les CAS SPECIAUX et (2) pour les CAS NON SPECIAUX (3) pour ne pas différencier: "))
    if a==1:
        donnes0= PET2_plus_4moins.query("Special_Case == 1")
    elif a==2:
        donnes0= PET2_plus_4moins.query("Special_Case == 2")
    else:
        donnes0= PET2_plus_4moins
        
    b=int(input("Patients PET4+: Entrez (1) pour selectionner les CAS SPECIAUX,(2) pour les CAS NON SPECIAUX  et (3) pour ne pas différencier: "))
    if b==1:
        donnes1= PET4_plus.query("Special_Case == 1")
    elif b==2:
        donnes1= PET4_plus.query("Special_Case == 2")
    else:
        donnes1= PET4_plus
    return donnes0, donnes1

########################################## DATAS ################################################
filename1 ="PET2_plus_PET4moins_global.xlsx"; 
filename2 ="PET4_plus_global.xlsx"
PET2_plus_4moins= pd.read_excel(filename1)
PET4_plus=pd.read_excel(filename2)
donnes=case()


########################################### PFS BOXPLOTS #######################################################   
box_data= [donnes[0]["PFS (months)"],donnes[1]["PFS (months)"]]

plt.subplot(121)
plt.boxplot(box_data[0])
plt.title("PFS des patients PET2+-->PET4-")
plt.ylabel(' PFS (en mois)')
plt.subplot(122)
plt.boxplot(box_data[1])
plt.title("PFS des patients PET4+")
plt.ylabel('')
#plt.savefig("delta_SUV_PFS_Boxes.pdf")
plt.show()

#################################################### STATISTICS ON BOXPLOT DATAS ###############################
print("Patients PET2+-->PET4-:", pd.DataFrame(box_data[0]).describe())
print("Patients PET4+:  ", pd.DataFrame(box_data[1]).describe())
lv_test =stats.levene(box_data[0],box_data[1], center='median', proportiontocut=0.05)#test de Levene
t_test =stats.ttest_ind(box_data[0],box_data[1], equal_var=False, alternative="two-sided") #Ttes: hypothèse des variances inégales
mw_test =stats.mannwhitneyu(box_data[0],box_data[1], alternative="two-sided")#test de Man Whitney
ks_test =stats.ks_2samp(box_data[0],box_data[1])#Test de Kolmogorov Smirnov
print("PFS: ", mw_test)

#################################################### PROGRESSION ANALYSIS (survival fonctions) ##############################################

kmf_PET2plus_4moins= KaplanMeierFitter()
kmf_PET4_plus= KaplanMeierFitter()
kmf_PET2plus_4moins.fit(durations= donnes[0]["PFS (months)"], event_observed=donnes[0]["uns"])
kmf_PET4_plus.fit(durations=donnes[1]["PFS (months)"], event_observed=donnes[1]["uns"])
plt.clf()
plt.title("Kaplan-Meieir estimate of progression")
plt.xlabel("time (months)")
plt.ylabel("Non-progression fraction")
kmf_PET2plus_4moins.plot(label="Patients PET2+-->PET4-")
kmf_PET4_plus.plot(label="Patients PET4+")
plt.show()

##############################################################" LOG RANK TEST ################################################################
print("LOG RANK TEST")
LR_test = logrank_test(donnes[0]["PFS (months)"], donnes[1]["PFS (months)"], event_observed_A = donnes[0]["uns"],  event_observed_B = donnes[1]["uns"])
print(LR_test.print_summary())


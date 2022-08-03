#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 11:48:44 2022

@author: kouame
"""

import pandas as pd
import numpy as np

tableaux= pd.read_excel("PET4_plus.xlsx")  #########A Utiliser#########
tableaux = tableaux.query("lost_patient == 0")
tableaux = tableaux.query("Delta_TMTV_PET4 !=100") #### A Commenter avnt d'exécuter pour prendre tout les patients#####

patients= tableaux.loc[:,["num_inclusion"]]
outcome= 1-tableaux.loc[:,["PFS_2y"]]

pfs_censoring = 1 - tableaux["PFS censoring"].to_numpy(dtype="bool")
pfs = tableaux["PFS (months)"].to_numpy(dtype="float")
survival_data = np.array((pfs_censoring, pfs)).T


features_name=["TMTV(TEP0)","TMTV(TEP4)", "Delta_TMTV(TEP4)", "TLG(TEP0)","TLG(TEP4)", "Delta_TLG(TEP4)","SUV_max(TEP0)","SUV_max(TEP4)","delta_SUV_max(TEP4)","Dmax(TEP0)" , "Cas(spécial ou non)","Age(mois)" , "LDH_(IU/L)","Stade_Ann_Arbor","Performance(ECOG)"] #
features= tableaux.loc[:,["TMTV_PET0_Vote_Maj","TMTV_PET4_Vote_Maj", "Delta_TMTV_PET4", "TLG_PET0_Vote_Maj","TLG_PET4_Vote_Maj", "Delta_TLG_PET4","SUV_max_PET0_Vote_Maj","SUV_max_PET4_Vote_Maj","delta_SUV_PET4_Vote_Maj","Dmax_PET0_Vote_Maj" , "Special_Case","Age(years)" , "LDH_(IU/L)","Ann_Arbor_Stage","Performance_Status_(ECOG scale)"]]#,
np.save("patients.npy",patients)
np.save("PFS_outcome.npy",outcome)
np.save("features_name.npy",features_name)
np.save("features.npy",features)
np.save("survival_data.npy", survival_data)


a= np.load("patients.npy")
b= np.load("PFS_outcome.npy")
c=np.load("features_name.npy")
d=np.load("features.npy")

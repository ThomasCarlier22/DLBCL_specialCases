# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 11:06:11 2021

@author: gafrecon
"""


from parameters import parameters
import numpy as np
import os
from features_extraction import classical_features_extraction
import matplotlib.pyplot as plt


parameters = parameters()



    


plt.close('all')


###### extraction  features ######

if parameters.features_extraction:
    
    # Prepare list of patients patients_id for features extraction
    available_mask = os.listdir(parameters.mask_path)
    available_pet0 = np.array(os.listdir(parameters.nifti_path))
    
    # retain only patients with pet volumes nii andmask
    mask_patients_to_remove = ~np.array([p in parameters.patients_to_remove for p in available_pet0]) #False Patients to remove 
    mask_missing_mask = np.array([p in available_mask for p in available_pet0]) # False Patients to remove 
    mask = mask_patients_to_remove & mask_missing_mask
    
    patients_id = available_pet0[mask]
#    patients_id = patients_id[12:]
      
    # classical features
    classical_features, classical_features_names = classical_features_extraction(parameters.nifti_path, parameters.mask_path, patients_id, parameters.settings_pyradiomics, save_extraction=parameters.save_extraction)
    

    # # largest
    # rad_features, rad_features_random, features_name = radiomics_extraction(parameters.nifti_path, parameters.mask_path, patients_id, parameters.settings_pyradiomics, parameters.extraction_zone, parameters.radiomics_log_and_wlt, parameters.verbosity, loop_textural_analysis=parameters.loop_textural_analysis, save_extraction=parameters.save_extraction, save_name=parameters.save_name)
    

    

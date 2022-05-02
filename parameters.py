# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 16:28:37 2021

@author: gafrecon
"""



import SimpleITK as sitk



class parameters:
    
    def __init__(self):
        
        
        
        ################ path ##################
        
        
        self.excel_clinical_data = "/home/fero/Bureau/features_extration/clinical_features_outcome_02_18.xlsx"
        self.nifti_path = "/home/fero/Bureau/features_extration/dossier_mask/features_extraction_selection_code/Gained_special_patients/GAINED_nifti_patient_ECHEC"     #"./GAINED_nifti_patient" 
        self.mask_path =  "/home/fero/Bureau/features_extration/dossier_mask/features_extraction_selection_code/Gained_special_patients/GAINED_MASK_ECHEC"              #"./GAINED_MASK"
        
        # patients to remove
        self.patients_to_remove = [ "11011106341004", "51011101291001", "51011102181003", "51011102341005", "51011107081003", "51011102161004", "51011104081005", "11011101391006","11011106051002", "51011101851003", "11011101051024", "11011101061002", "51011101301002", "51011101871001", "51011102161003", "51011102171003", "51011101261002", "11011106341003"]  # excel missing mask/rtstruct, small n_slices, missing slices



        ################ computation in main ################
        
        
        self.features_extraction = True # radiomics and classical features extraction

        
        
        ################ Parameters Features extraction ################
        
        
        # parametes pyradiomics
        settings_pyradiomics = {}
        #settings_pyradiomics['binCount'] = 64
        #settings_pyradiomics['binWidth'] = 0.3
        settings_pyradiomics['resampledPixelSpacing'] = [2,2,2] #[4,4,2]
        settings_pyradiomics['interpolator'] = sitk.sitkBSpline
        settings_pyradiomics['preCrop'] = True
        settings_pyradiomics['weightingNorm'] = 'no_weighting'
        settings_pyradiomics['sigma'] = [2, 6]
        self.settings_pyradiomics = settings_pyradiomics
        
        # radiomics extraction and analysis
        self.verbosity = 40
        self.radiomics_log_and_wlt = True # add log and wavelet images for radiomcis extraction in addition to original images
        if self.radiomics_log_and_wlt:
            self.save_name = "_all" # 
        else: 
            self.save_name = "_original"
        self.extraction_zone = "tmtv"
        self.loop_textural_analysis = 0  # n loop random features for textural analysis
        self.save_extraction = True # save radiomics computed






        
if __name__=="__main__":

    a=0

        
        
        
        
        
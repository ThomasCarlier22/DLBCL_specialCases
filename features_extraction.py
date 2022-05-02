# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 11:06:48 2021

@author: gafrecon
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 08:35:45 2021

@author: GauthierFrecon
"""

import os 
import cv2
import SimpleITK as sitk
import radiomics
import numpy as np
from radiomics import featureextractor 
from radiomics import imageoperations #firstorder, getTestCase, glcm, glrlm, glszm, shape
import random as rnd
from dicom_conversion import convertToSUV
import sys
sys.path.insert(1, './Dicom-To-CNN')
from dicom_to_cnn.tools.pyradiomics.pyradiomic import  calcul_distance_max, get_center_of_mass_modified


    
def classical_features_extraction(nifti_path, mask_path, list_patients, settings, verbosity=40, save_extraction=True):
    '''
    Parameters
    ----------
    nifti_path : string
        data path to GAINED_nifti_patient folder  
    mask_path : string
        data path to GAINED_MASK folder  
    list_patients : list (string)
        patient number for who compute radiomics

    verbosity : int, optional
        pyradiomics verbosity. The default is 40.
    save_extraction: bool
        save extraction in "./fikles_to_load"

    Returns
    -------
    classical_features: array (n_patients, n_features)
        array of features: SUV_max, TMTV, TLG, Dmax

    '''
 
    n_patients = len(list_patients)
    radiomics.setVerbosity(verbosity)
    

    # extractor relative to DLBCL
    classical_features_names = ["TMTV", "SUV_max", "TLG", "Dmax"]
    n_classical_features = len(classical_features_names)
    classical_features = np.zeros((n_patients, n_classical_features))
    
    # classical features extractor
    classical_feature_extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    classical_feature_extractor.disableAllFeatures()
    classical_feature_extractor.enableImageTypes(Original={})
    classical_feature_extractor.enableFeaturesByName(firstorder=['Maximum', 'Mean'], shape=['MeshVolume'])


    for i in range(n_patients):
        
        patient = list_patients[i]
        print("patient: ", patient)
        imagePath = nifti_path + "/" + patient + "/" + "PET0/patient.nii"
        petPath = nifti_path + "/" + patient + "/" + "PET0/dicom_slice"
        label_dir = mask_path + "/" + patient + "/pet0/"
        mask_name = os.listdir(label_dir)[0]
        labelPath = os.path.join(label_dir, mask_name)
        image = sitk.ReadImage(imagePath)
        
        # SUV conversion
        image, suv_factor = convertToSUV(petPath, image) # conversion to SUV
            
        mask = sitk.ReadImage(labelPath) 
        
        # Segmentation
        image_arr = sitk.GetArrayFromImage(image) 
        mask_arr = sitk.GetArrayFromImage(mask)
        
        #plot_slices_im_mask(image_arr, mask_arr)  # plot lesion and mask coherence
        
        m4=      mask_suv4(image_arr, mask_arr)
        m2p5=    mask_suv2p5(image_arr, mask_arr)
        m40=     mask_40_percent_suvmax(image_arr, mask_arr)
        m_km =   mask_kmeans(image_arr,mask_arr)
        mask_arr = majority_vote(m4,m2p5,m40,m_km) 
        
        # mask_arr= mask_kmeans(image_arr,mask_arr)
        mask = sitk.GetImageFromArray(mask_arr) 
        mask.SetSpacing(image.GetSpacing())
        mask.SetOrigin(image.GetOrigin())
           
        
        # classical_features extraction: Dmax
        
        liste_center = get_center_of_mass_modified(mask)
        classical_features[i, -1] = calcul_distance_max(liste_center)
        
        # Define extraction zone and classical features extraction
        dim_mask = mask_arr.ndim
        if dim_mask==4:  # fourth dimension => there is more than one ROI
             
            # tmtv mask (3D)
            mask_arr_tmtv = np.sum(mask_arr, axis=-1)>0 # in case that ROI are overlapped
            mask_arr_tmtv = np.array(mask_arr_tmtv, dtype='int') # sum 4th dimension  
            
            mask_tmtv = sitk.GetImageFromArray(mask_arr_tmtv) 
            mask_tmtv.SetSpacing(image.GetSpacing())
            mask_tmtv.SetOrigin(image.GetOrigin())
        
            # classical_features extraction: TMTV, SUV_max, TLG
            result = classical_feature_extractor.execute(image, mask_tmtv, label=1) # feature extraction
            classical_features[i, :-1]  = [value for key, value in result.items() if key.startswith("original")]
            classical_features[i, 2] = classical_features[i, 0]*classical_features[i, 2] # replace SUV_mean by TLG=TMTV*SUV_mean
     
                
        else: # only 1 ROI => No 4th dimension, the mask is already in tmtv/hottest/largest segmentation 3D
        
            #classical_features extraction: TMTV and SUV_max
            result = classical_feature_extractor.execute(image, mask, label=1) # feature extraction
            classical_features[i, :-1] = np.array([value for key, value in result.items() if key.startswith("original")])
            classical_features[i, 2] = classical_features[i, 0]*classical_features[i, 2] # replace SUV_mean by TLG=TMTV*SUV_mean
        
    if save_extraction:   
       np.save("./files_to_load/classical_features.npy", classical_features)
       np.save("./files_to_load/classical_features_names.npy", classical_features_names)
       np.save("./files_to_load/list_patients.npy", list_patients)
       
       
    return(classical_features, classical_features_names)     
    
    
    
def radiomics_extraction(nifti_path, mask_path, list_patients, settings, extraction_zone="tmtv", radiomics_log_and_wlt = False, verbosity=40, loop_textural_analysis=0, save_extraction=True, save_name="_all"):
    '''


    Parameters
    ----------
    nifti_path : string
        data path to GAINED_nifti_patient folder  
    mask_path : string
        data path to GAINED_MASK folder  
    list_patients : list (string)
        patient number for who compute radiomics
    settings : dict
        settings for the radiomics extractor of pyradiomics
    extraction_zone: str
        zone for the radiomics extraction: "tmtv", "hottest" or "largest" lesions
    verbosity : int, optional
        pyradiomics verbosity. The default is 40.
    loop_textural_analysis : int, optional
        iterations number of random feature (i.e pixels shuffling before computation) to calculate, if 0, no random feature computation. The default is 0.
    save_extraction: bool
        save extraction in "./files_to_load"
    save_name: str
        name to add after './files_to_load/rad_features......"
        
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
    BC = 64 # Hard coded !!
    BW = 0.3 # Hard coded !!
    
    
        # features list 
    if extraction_zone=="tmtv": 
        # we don't consider maximum and MeshVolume that are equal to SUV_max and TMTV in this case
        first_order = ['Energy']
        shape = ['SurfaceArea', 'SurfaceVolumeRatio', 'Sphericity', 'Elongation', 'Flatness', 'Maximum3DDiameter', 'MajorAxisLength', 'MinorAxisLength']
    elif extraction_zone=="largest": 
        # maximum and mesh volume different of SUV_max and TMTV
        first_order = ['Maximum', 'Energy']
        shape = ['MeshVolume', 'SurfaceArea', 'SurfaceVolumeRatio', 'Sphericity', 'Elongation', 'Flatness', 'Maximum3DDiameter', 'MajorAxisLength', 'MinorAxisLength']

    elif extraction_zone=="hottest": 
        # mesh volume different of SUV_max and TMTV, but max=suvmax
        first_order = ['Energy']
        shape = ['MeshVolume', 'SurfaceArea', 'SurfaceVolumeRatio', 'Sphericity', 'Elongation', 'Flatness', 'Maximum3DDiameter', 'MajorAxisLength', 'MinorAxisLength']

    glcm = ['JointEntropy', 'Id', 'DifferenceAverage', 'JointAverage', 'DifferenceEntropy', 'ClusterShade']
    glrlm = ['HighGrayLevelRunEmphasis', 'LongRunHighGrayLevelEmphasis', 'RunLengthNonUniformity', 'ShortRunHighGrayLevelEmphasis', 'ShortRunLowGrayLevelEmphasis', 'LowGrayLevelRunEmphasis', 'GrayLevelNonUniformity']
    glszm = ['SizeZoneNonUniformity', 'SmallAreaEmphasis', 'LargeAreaEmphasis', 'ZonePercentage']
    gldm = ['DependenceNonUniformity','SmallDependenceHighGrayLevelEmphasis', 'LargeDependenceLowGrayLevelEmphasis', 'SmallDependenceEmphasis', 'LowGrayLevelEmphasis', 'LargeDependenceHighGrayLevelEmphasis', 'HighGrayLevelEmphasis', 'GrayLevelNonUniformity']
    ngtdm = ['Coarseness', 'Complexity', 'Strength']
    
    len_shape_features = len(shape) 
    len_textural = len(glcm) + len(glrlm) + len(glszm) + len(gldm) + len(ngtdm) 
    len_textural_and_first_order = len(first_order) + len_textural 

    # Radiomics features_extractor
    settings['binCount'] = BC
    settings['binWidth'] = None
    feature_extractor_BC = featureextractor.RadiomicsFeatureExtractor(**settings)
    settings['binWidth'] = BW
    settings['binCount'] = None
    feature_extractor_BW = featureextractor.RadiomicsFeatureExtractor(**settings)
    feature_extractor_BC.disableAllFeatures()
    feature_extractor_BW.disableAllFeatures()
    
    # compute radiomics only from original image or from also log and wlt images
    if radiomics_log_and_wlt:
        feature_extractor_BW.enableImageTypes(Original={}, LoG={}, Wavelet={})
        feature_extractor_BC.enableImageTypes(Original={}, LoG={}, Wavelet={})
        n_derived_images = 11 # original image + 2*log images + 8*wavelet images
        list_image_type = ("original", "log", "wavelet")
        
    else:
        feature_extractor_BC.enableImageTypes(Original={})
        feature_extractor_BW.enableImageTypes(Original={})
        n_derived_images = 1 # only original image 
        list_image_type = ("original")
        
    
    # BC features    
    feature_extractor_BC.enableFeaturesByName(firstorder=first_order, glcm=glcm, glrlm=glrlm, glszm=glszm, gldm=gldm, ngtdm=ngtdm, shape=shape)
    n_features_BC = len_shape_features + n_derived_images*len_textural_and_first_order
    
    # BW features: idem without shape features
    feature_extractor_BW.enableFeaturesByName(glcm=glcm, glrlm=glrlm, glszm=glszm, gldm=gldm, ngtdm=ngtdm)
    n_features_BW =  n_derived_images*len_textural
    

    rad_features = np.zeros((n_patients, n_features_BC + n_features_BW))
    rad_features_random = np.zeros((n_patients, loop_textural_analysis, n_features_BC + n_features_BW)) # needed only if textural analysis(loop_textural_analysios>0)

    for i in range(n_patients):
        
        patient = list_patients[i]
        print("patient: ", patient)
        imagePath = nifti_path + "/" + patient + "/" + "PET0/patient.nii"
        petPath = nifti_path + "/" + patient + "/" + "PET0/dicom_slice"
        label_dir = mask_path + "/" + patient + "/pet0/"
        mask_name = os.listdir(label_dir)[0]
        labelPath = os.path.join(label_dir, mask_name)
        image = sitk.ReadImage(imagePath)
        
        # SUV conversion
        image, suv_factor = convertToSUV(petPath, image) # conversion to SUV
            
        mask = sitk.ReadImage(labelPath) 
        
        # Segmentation
        image_arr = sitk.GetArrayFromImage(image) 
        mask_arr = sitk.GetArrayFromImage(mask)
        mask_arr = mask_suv4(image_arr, mask_arr) #new mask with a suv4 segmentaiton inside initial mask
        mask = sitk.GetImageFromArray(mask_arr) 
        mask.SetSpacing(image.GetSpacing())
        mask.SetOrigin(image.GetOrigin())
        
        
        # Define extraction zone and classical features extraction
        dim_mask = mask_arr.ndim
        if dim_mask==4:  # fourth dimension => there is more than one ROI
             
            # tmtv mask (3D)
            mask_arr_tmtv = np.sum(mask_arr, axis=-1)>0 # in case that ROI are overlapped
            mask_arr_tmtv = np.array(mask_arr_tmtv, dtype='int') # sum 4th dimension  
            
            mask_tmtv = sitk.GetImageFromArray(mask_arr_tmtv) 
            mask_tmtv.SetSpacing(image.GetSpacing())
            mask_tmtv.SetOrigin(image.GetOrigin())
        
            if extraction_zone=="tmtv":  # sum channel
                  
                # tmtv mask (3D)
                newMask = mask_tmtv
                    
            elif extraction_zone=="largest": # find largest volume
            
                largest_ROI = find_largest_ROI(mask_arr)
                mask_arr = mask_arr[:, :, :, largest_ROI] # largest ROI       
                
                # largest mask 3D
                newMask = sitk.GetImageFromArray(mask_arr)
                newMask.SetSpacing(image.GetSpacing())
                newMask.SetOrigin(image.GetOrigin())
            
            elif extraction_zone=="hottest": # find ROI with the highest max value
            
                image_arr = sitk.GetArrayFromImage(image)
                hottest_ROI = find_hottest_ROI(mask_arr, image_arr)    
                mask_arr = mask_arr[:, :, :, hottest_ROI] #hottest ROI       
                
                # hottest mask 3D
                newMask = sitk.GetImageFromArray(mask_arr)
                newMask.SetSpacing(image.GetSpacing())
                newMask.SetOrigin(image.GetOrigin())
                
        else: # only 1 ROI => No 4th dimension, the mask is already in tmtv/hottest/largest segmentation 3D
        
            newMask = mask
          
          
        # #radiomics computation BC resampling BC
        settings['binCount'] = BC
        settings['binWidth'] = None
        image_BC, newMask_BC = imageoperations.resampleImage(image, newMask, **settings)
        origin =  image_BC.GetOrigin()
        spacing = image_BC.GetSpacing()        
        image_arr_BC = sitk.GetArrayFromImage(image_BC) 
     
        result_BC = feature_extractor_BC.execute(image_BC, newMask_BC, label=1) #feature extraction
        rad_features[i, :n_features_BC] = np.array([value for key, value in result_BC.items() if key.startswith(list_image_type)])
        

        #random features computation    
        if loop_textural_analysis != 0:
            for k in range(loop_textural_analysis):
                image_arr_BC = sitk.GetArrayFromImage(image_BC)
                label_arr_BC = sitk.GetArrayFromImage(newMask_BC)
                pixels = image_arr_BC[label_arr_BC==1]
                rnd.shuffle(pixels)  #pixel shuffled before radiomics computation
                image_arr_BC[label_arr_BC==1] = pixels
                
                image_BC = sitk.GetImageFromArray(image_arr_BC)
                image_BC.SetSpacing(spacing)
                image_BC.SetOrigin(origin)       
                   
                result = feature_extractor_BC.execute(image_BC, newMask_BC, label=1)            
                rad_features_random[i, k, :n_features_BC] = np.array([value for key, value in result.items() if key.startswith(list_image_type)]) 
     
        #radiomics computation BW
        settings['binWidth'] = BW
        settings['binCount'] = None
        image_BW, newMask_BW = imageoperations.resampleImage(image, newMask, **settings)
        origin =  image_BW.GetOrigin()
        spacing = image_BW.GetSpacing()        
        image_arr_BW = sitk.GetArrayFromImage(image_BW) 
     
        result_BW= feature_extractor_BW.execute(image_BW, newMask_BW, label=1) #feature extraction
        rad_features[i, n_features_BC:] = np.array([value for key, value in result_BW.items() if key.startswith(list_image_type)])
        

        #random features computation    
        if loop_textural_analysis != 0:
            for k in range(loop_textural_analysis):
                image_arr_BW = sitk.GetArrayFromImage(image_BW)
                label_arr_BW = sitk.GetArrayFromImage(newMask_BW)
                pixels = image_arr_BW[label_arr_BW==1]
                rnd.shuffle(pixels)  #pixel shuffled before radiomics computation
                image_arr_BW[label_arr_BW==1] = pixels
                
                image_BW = sitk.GetImageFromArray(image_arr_BW)
                image_BW.SetSpacing(spacing)
                image_BW.SetOrigin(origin)       
                   
                result = feature_extractor_BW.execute(image_BW, newMask_BW, label=1)            
                rad_features_random[i, k, n_features_BC:] = np.array([value for key, value in result.items() if key.startswith(list_image_type)]) 

    features_names_BC = [key + "_BC"  for key, value in result_BC.items() if key.startswith(list_image_type)]
    features_names_BW = [key + "_BW" for key, value in result_BW.items() if key.startswith(list_image_type)]
    features_names = features_names_BC + features_names_BW

    if save_extraction:
        np.save("./files_to_load/list_patients_" + extraction_zone  + ".npy", list_patients)
        np.save("./files_to_load/rad_features" + save_name + "_" + extraction_zone  + ".npy", rad_features)
        np.save("./files_to_load/rad_features_random" + save_name + "_" + extraction_zone  + ".npy", rad_features_random)
        np.save("./files_to_load/features_name" + save_name  + "_" + extraction_zone  +  ".npy", features_names)
  
    return(rad_features, rad_features_random, features_names) 


def find_largest_ROI(mask):
    
    n_ROI = np.shape(mask)[-1]
    largest_ROI = 0
    v_largest_ROI = 0

    for c in range(n_ROI):
        
        v_ROI = np.sum(mask[:, :, :, c])
        
        if v_ROI >= v_largest_ROI:
            largest_ROI = c
            v_largest_ROI = v_ROI
    
    return(largest_ROI)    
    


def find_hottest_ROI(mask, im):
    
    n_ROI = np.shape(mask)[-1]
    hottest_ROI = 0
    max_hottest_ROI = 0
    max_ROI = 0

    for c in range(n_ROI):
        
        mask_ROI = mask[:, :, :, c]
        ROI = im[mask_ROI>0]
        if len(ROI)>0: # it can happen that the ROI is empty !
            max_ROI = np.max(ROI)
        
            if max_ROI >= max_hottest_ROI:
                hottest_ROI = c
                max_hottest_ROI = max_ROI
    
    return(hottest_ROI)    
    


def mask_suv4(image_arr, mask_arr):
    """
    apply a suv4 segmentaiton to the image inside initial mask in order to create a new suv 4 mask

    Parameters
    ----------
    image_arr : arr
        suv image array 
    mask_arr : arr
        mask where to apply the segmentation

    Returns
    -------
    m4 : int arr
        mask of the suv4segmentation inside inital mask

    """
    suv_threshold = 4
    m4=np.zeros_like(mask_arr)
    dim_array = len(mask_arr.shape)
    
    if dim_array == 3 : 
        #3D mask
        image_lesion = np.copy(image_arr)
        image_lesion[np.where(mask_arr==0)] = 0
        m4[:,:,:] = np.array(image_lesion > suv_threshold, dtype='int')
     

    else :
        #4D mask
        for i in range(mask_arr.shape[3]):
       
            mask_3d = mask_arr[:,:,:,i].astype('int8')

            if len(np.unique(mask_3d)) > 1 : 
                image_lesion = np.copy(image_arr)
                image_lesion[np.where(mask_3d==0)] = 0
                m4[:,:,:,i] = np.array(image_lesion > suv_threshold, dtype='int')
    m4=m4.astype(np.uint8)
    
    return m4




def mask_suv2p5(image_arr, mask_arr):
    
    suv_threshold = 2.5
    m2p5=np.zeros_like(mask_arr)
    dim_array = len(mask_arr.shape)
    
    if dim_array == 3 : 
        #3D mask
        image_lesion = np.copy(image_arr)
        image_lesion[np.where(mask_arr==0)] = 0
        m2p5[:,:,:] = np.array(image_lesion > suv_threshold, dtype='int')
    else :
        #4D mask
        for i in range(mask_arr.shape[3]):
            mask_3d = mask_arr[:,:,:,i].astype('int8')

            if len(np.unique(mask_3d)) > 1 : 
                image_lesion = np.copy(image_arr)
                image_lesion[np.where(mask_3d==0)] = 0
                m2p5[:,:,:,i] = np.array(image_lesion > suv_threshold, dtype='int')
    m2p5=m2p5.astype(np.uint8)
    
    return m2p5



def mask_40_percent_suvmax(image_arr, mask_arr):
    
    m40=np.zeros_like(mask_arr)
    dim_array = len(mask_arr.shape)
    
    if dim_array == 3 : 
        #3D mask
        image_lesion = np.copy(image_arr)
        image_lesion[np.where(mask_arr==0)] = 0
        max_im = np.max(image_lesion)
        suv_threshold = 0.4* max_im
        m40[:,:,:] = np.array(image_lesion > suv_threshold, dtype='int')
     
    else :
        #4D mask
        for i in range(mask_arr.shape[3]):
            mask_3d = mask_arr[:,:,:,i].astype('int8')

            if len(np.unique(mask_3d)) > 1 : 
                image_lesion = np.copy(image_arr)
                image_lesion[np.where(mask_3d==0)] = 0
                max_im = np.max(image_lesion)
                suv_threshold = 0.4*max_im
                m40[:,:,:,i] = np.array(image_lesion > suv_threshold, dtype='int')
                
    m40=m40.astype(np.uint8)
    
    return m40



def mask_kmeans(image_arr,mask_arr):
    
    m_km=np.zeros_like(mask_arr)   
    dim_array = len(mask_arr.shape)
    
    if dim_array==3:
        
        image_lesion = np.copy(image_arr)
        image_lesion[np.where(mask_arr==0)] = 0
        Z= image_lesion.reshape((-1,1))
        Z = np.float32(Z)        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1)
        ret,label,center=cv2.kmeans(Z,2,None,criteria,10,cv2.KMEANS_PP_CENTERS)
        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((image_lesion.shape))
        
        pos = np.array(res2==np.max(center), dtype='int')
        image_lesion[np.where(pos==0)] = 0
        m_km[:,:,:]= np.array(image_lesion!=0, dtype='int')
    
    else:
        
        for i in range(mask_arr.shape[3]):
            
            mask_3d = mask_arr[:,:,:,i].astype('int8')
            
            if len(np.unique(mask_3d)) > 1 : 
                
                image_lesion = np.copy(image_arr)
                image_lesion[np.where(mask_3d==0)] = 0
                Z= image_lesion.reshape((-1,1))
                Z = np.float32(Z)        
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1)
                ret,label,center=cv2.kmeans(Z,2,None,criteria,10,cv2.KMEANS_PP_CENTERS)
                # Now convert back into uint8, and make original image
                center = np.uint8(center)
                res = center[label.flatten()]
                res2 = res.reshape((image_lesion.shape))
    
                pos = np.array(res2==np.max(center), dtype='int')
                image_lesion[np.where(pos==0)] = 0
                m_km[:,:,:,i]= np.array(image_lesion!=0, dtype='int')
                   
                
    m_km=m_km.astype(np.uint8)
    
    
    return m_km
    
   
def majority_vote(m4,m2p5,m40,m_km):
    ch=np.where(m2p5!=0)
    if (ch[0].size == 0 and ch[1].size == 0 and ch[2].size == 0) or ch[0].size*ch[1].size*ch[2].size < 64: #To avoid very small delineation for low intensity lesion
        thresh = 0.66  # 0.66  # threshold value for accepting a voxel as belonging in the resulting majority vote mask
        nbMethods = 3  # Number of methods used (40%, 4.0 and kmean)
    else :
       thresh = 0.49   #0.49   #threshold value for accepting a voxel as belonging in the resulting majority vote mask
       nbMethods = 4   # Number of methods used (40%, 4.0 and kmean)

    # Initialize resulting matrix
    sum_mask = m4 + m2p5 + m40 + m_km
    sum_mask = np.divide(sum_mask, nbMethods)
    mMajority=np.array( sum_mask > thresh, dtype='int')
    mMajority= mMajority.astype(np.uint8)
    
    return mMajority

     

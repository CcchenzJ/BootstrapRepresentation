from .cfg_class import *

# -------------------------------------------------------------------------------------
""" M&Ms """

mnm_dataset = Config({
    'name': 'MnM_Volume',
    # choose the wrapped dataset.
    'type': 'MnMVolumetricSet',
    # labels in oder 0-3. 
    'class_names': ('bg', 'lv', 'myo', 'rv'),
    # sepecifiy the vendors using. {PHASE: List[VENDOR]} 
    'vendors': {'train': ['A', 'B']},
    # preprocessing configs.
    'target_resolution': (1.25, 1.25),
    
    'train': Config({
        'data': 'dataset/MnM/Training/',
    }),
    'valid': Config({
        'data': 'dataset/MnM/Validation/',
    }),
    'test': Config({
        'data': 'dataset/MnM/Testing/',
    })
})

mnm_img_dataset = mnm_dataset.copy({
    'name': 'MnM_Image',
    'type': 'MnMImageSet', 
    
    'vendors': {'train': ['A', 'B'], 'valid':['A','B','C','D'], 'test':['A','B','C','D']}
})

# ----------------------------------------------------------------------------------------
""" ACDC """

acdc_dataset = Config({
    'name': 'ACDC_Volume',
    'type': 'VolumetricSet', 
    'class_names': ('bg', 'rv', 'myo', 'lv'),

    'target_resolution': (1.36719, 1.36719),
    
    'train': Config({
        'data': 'dataset/ACDC/train/*_gt.nii.gz',
    }),
    'valid': Config({
        'data': 'dataset/ACDC/valid/*_gt.nii.gz',
    }),
    'test': Config({
        'data': 'dataset/ACDC/test/*_gt.nii.gz',
    })
})

acdc_img_dataset = acdc_dataset.copy({
    'name': 'ACDC_Image',
    'type': 'ImageSet', 
})

acdc_cgl_dataset = acdc_dataset.copy({
    'type': 'CGLACDCPreTrainSet',
})

# ---------------------------------------------------------------------------------------------
""" Prostate """

prostate_dataset = Config({
    'name': 'Prostate_Volume',
    'type': 'VolumetricSet', 
    'class_names': ('bg', 'pz', 'tz'),
  
    'target_resolution': (0.625, 0.625),
    
    'train': Config({
        'data': 'dataset/Prostate/train/*_gt.nii.gz',
    }),
    'valid': Config({
        'data': 'dataset/Prostate/valid/*_gt.nii.gz',
    }),
    'test': Config({
        'data': 'dataset/Prostate/test/*_gt.nii.gz',
    })
})

prostate_img_dataset = prostate_dataset.copy({
    'name': 'Prostate_Image',
    'type': 'ImageSet', 
})

prostate_cgl_dataset = prostate_dataset.copy({
    'type': 'CGLACDCPreTrainSet',
})

# -----------------------------------------------------------------------------
""" CAMUS """

camus_a2c_dataset = Config({
    'name': 'CAMUS_A2C_Sequence',
    'type': 'CAMUSSequenceSet', 

    # sepecifiy the view using. List[views] 
    'use_view': ['A2C'], 

    'class_names': ('bg', 'cv', 'myo', 'ca'),
      
    'train': Config({
        'data': 'dataset/CAMUS/All_train.npy',
    }),
    'valid': Config({
        'data': 'dataset/CAMUS/All_valid.npy',
    }),
    'test': Config({
        'data': 'dataset/CAMUS/All_test.npy',
    })
})

camus_a2c_img_dataset = camus_a2c_dataset.copy({
    'name': 'CAMUS_A2C_Image',
    'type': 'CAMUSImageSet', 
})

camus_a2c_cgl_dataset = camus_a2c_dataset.copy({
    'type': 'CGLCAMUSPreTrainSet',
})

camus_a4c_dataset = camus_a2c_dataset.copy({
    'name': 'CAMUS_A4C_Sequence',
    'use_view': ['A4C'], 
})

camus_a4c_img_dataset = camus_a4c_dataset.copy({
    'name': 'CAMUS_A4C_Image',
    'type': 'CAMUSImageSet', 
})

camus_a4c_cgl_dataset = camus_a4c_dataset.copy({
    'type': 'CGLCAMUSPreTrainSet',
})

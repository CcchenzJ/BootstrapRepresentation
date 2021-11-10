from .cfg_class import *

seg_loss = Config({
    'type': 'SegLoss',
    'labels': ['D', 'C'],

    'use_ce_loss' : False,
    'ce_loss_alpha': 1.0,

    'use_smooth_label_loss': False,
    'smooth_epsilon': 0.1,

    'use_dice_loss': True,
    'dice_loss_alpha': 1.0,
})

bootstrap_loss = Config({
    'type': 'BootstrapLoss',    
    'labels': ['S', 'C', 'P'],
    
    'tau': 0.1,

    'use_seg_loss': True,
    'use_dice_loss': True,
    'use_ce_loss': False,
    'seg_loss_alpha': 2.0,

    'use_pred_loss': True,
    'pred_loss_alpha': 1.0,

    'use_fbc_loss': True,
    'fbc_loss_alpha': 0.5,
})

contrast_loss = Config({
    'type': 'ContrastiveLoss',
    'labels': ['C'],

    'use_cossim_loss': True,
    'cossim_loss_alpha': 1.0,
})

global_loss = Config({
    'type': 'CGLContrastiveLoss',
    'labels': ['G'],

    'tau': 0.1,
    'use_global_loss': True, 
    'global_loss_alpha': 1.0,

    'use_local_loss': False,
})

local_loss = Config({
    'type': 'CGLContrastiveLoss',
    'labels': ['L'],

    'region_size': 3, 
    'sqrt_num_regions': 3,

    'tau': 0.1,
    'use_local_loss': True,
    'local_loss_alpha': 1.0,

    'use_global_loss': False, 
})


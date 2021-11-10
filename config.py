
''' Config file. '''
import math

from configs.cfg_class import Config

from configs.loss_cfg import *
from configs.model_cfg import * 
from configs.dataset_cfg import *
from configs.optimizer_cfg import *
from configs.transformer_cfg import *

# These are in BGR and are for ImageNet
MEANS = (103.94, 116.78, 123.68)
STD   = (57.38, 57.12, 58.40)
# Project to Gray scale.
MEANS = (0.29900 * MEANS[0] + 0.58700 * MEANS[1] + 0.11400 * MEANS[2])
STD   = (0.29900 * STD[0]   + 0.58700 * STD[1]   + 0.11400 * STD[2])
# Normal distribution.
MEANS = 0
STD   = 1

# maybe useful
COLORS = ((244, 67, 54),(158,158,158),(156, 39,176),(112,131, 35),
          (103, 58,183),(63,  81,181),(33, 150,243),(3,  169,244),
          (0,  188,212),(0,  150,136),(76, 175, 80),(139,195, 74),
          (205,220, 57),(255,235, 59),(255,193,  7),(255,152,  0),
          (255, 87, 34),(121, 85, 72),(233, 30, 99),(96, 125,139))

# ---------------------------------------------------------- #

baseline_acdc_all_config = Config({
    'name': 'baseline_acdc_trall',
    # including the background.
    'num_classes': 3+1,
    # select the training mode, 'normal' means supervised training.
    'scheme': 'normal',
    # disable if set to 0.
    'mixup_alpha': 0,

    'dataset': acdc_img_dataset,
    'img_size': (192,192),
    'max_iter': 20000,
    # set the learning rate schedule, 'cos' or 'step'.
    'lr_schedule': 'cos',
    'gamma': 0.1, # the decay rate for step lr schedule.
    'lr_steps': (10000, 15000),
    'lr_warmup_until': 0,

    # select the metric to collate the best checkpoint.
    'best_metric': ('dice', True),
    # select the metrics to plot the curve during training. 
    'tovis_metrics' : ('dice', ),
    # define the metrics to evaluation. {METRIC: (Type of Metric, {'mission': MISSION})}
    'toevl_metrics' : {'dice': ('VolDiceCalc', {'mission': 'seg'}),
                       'dist': ('HanDistCalc', {'mission': 'seg'}),
                       'assd': ('AssDistCalc', {'mission': 'seg'}) },

    'post_processes': {},
    
    'optimizer': adam.copy({
        'args': {'lr': 1e-3, 'eps': 1e-8, 'betas': (0.9, 0.99), 'weight_decay': 5e-4},
    }),

    'transformer': acdc_img_augmentor,
    'model': unet,
    'loss': seg_loss,
}) 

baseline_acdc_2p_config = baseline_acdc_all_config.copy({
    'name': 'baseline_acdc_tr2',
    'max_iter': 10000,
    'dataset': acdc_img_dataset.copy({
        # define the number of patients to use, the randomly selected patient index is
        # saved in './dataset/patient_set/'.
        'num_patients': 2,
        # set True to use the pre-selected patient set, or to re-select.
        'has_patient_set': True,
    }),
})

pt_acdc_all_config = baseline_acdc_all_config.copy({
    'name': 'acdc_trall',
    'num_classes': 4,
    'scheme': 'proposed',
    
    'mixup_alpha': 0,

    'two_dataset': {'labeled':acdc_img_dataset, 
                    'unlabeled': acdc_dataset,
                    },
    'dataset':acdc_img_dataset,
    'img_size': (192,192),

    # Max distance for selecting slices.
    'max_dist': 8,
    'num_samples': 3,
    'num_subjects': 2,

    'max_iter': 40000,
    'lr_steps': (),

    'loss_warmup': {'keys':('fbc', 'pred'), 
                    'until': 1000,},

    'best_metric': ('dice', True),  
    'tovis_metrics' : ('dice', ),
    'toevl_metrics' : {'dice': ('VolDiceCalc', {'mission': 'seg'}),
                       'dist': ('HanDistCalc', {'mission': 'seg'}),
                       'assd': ('AssDistCalc', {'mission': 'seg'}) },

    'post_processes': {},

    'optimizer': adam.copy({
        'args': {'lr': 1e-3, 'eps': 1e-8, 'betas': (0.9, 0.99), 'weight_decay': 5e-4},
    }),

    'transformer': {'labeled': acdc_img_augmentor, 'unlabeled': acdc_vol_augmentor},
    
    'model': SimSiam_DSAG,
    'loss': bootstrap_loss,
})

pt_acdc_2p_config = pt_acdc_all_config.copy({
    'name': 'acdc_tr2',
    'two_dataset': {'labeled':acdc_img_dataset.copy({
                        'num_patients': 2,
                        'has_patient_set': True,}), 
                    'unlabeled': acdc_dataset,
                },
    'max_iter': 10000,

    'loss_warmup': {'keys':('fbc', 'pred'), 
                    'until': 1000,},
})

# -------------------------------------------------------------------------------------
pt_cgl_acdc_enc_config = baseline_acdc_all_config.copy({
    'name': "acdc_CglPtrEnc",
    'scheme': 'pretrain_CGL',
    
    'num_parts': 4,

    'dataset': acdc_cgl_dataset,
    'img_size': (192, 192),

    'max_iter': 20000,

    'lr_schedule': 'cos',
    'gamma': 0.1,
    'lr_steps': (),
    'lr_warmup_until': 0,

    'best_metric': ('none',),  
    'tovis_metrics' : (),
    'toevl_metrics' : {},
    'post_processes': {},
    
    'transformer': acdc_img_augmentor,
        
    'model': cgl_encoder,
    'loss': global_loss,    
})
pt_cgl_acdc_dec_config = pt_cgl_acdc_enc_config.copy({
    'name': 'acdc_CglPtrDec',
    'max_iter': 10000,
    'optimizer': adam_freeze_enc,

    'model': cgl_decoder.copy({
        'backbone':{
            'path': 'path/to/pretrained/weights.pth',
            'arch': dict(**cgl_encoder.encoder.architecture),
        }
    }),
    'loss': local_loss,
})

# -------------------------------------------------------------------------------------
baseline_mnm_all_config = baseline_acdc_all_config.copy({
   'name': 'baseline_mnm_trall',
    'num_classes': 3+1,
    'scheme': 'normal',

    'mixup_alpha': 0,

    'dataset': mnm_img_dataset,
    'img_size': (192,192),
    'max_iter': 80000,

    'lr_schedule': 'cos',
    'gamma': 0.1,
    'lr_steps': (),
    'lr_warmup_until': 0,

    'best_metric': ('dice', True),  
    'tovis_metrics' : ('dice', ),
    'toevl_metrics' : {'dice':('VolDiceCalc', {'mission': 'seg'}),
                       'dist': ('HanDistCalc', {'mission': 'seg'}),
                       'assd': ('AssDistCalc', {'mission': 'seg'}) },

    'post_processes': {},
    
    'optimizer': adam.copy({
        'args': {'lr': 1e-3, 'eps': 1e-8, 'betas': (0.9, 0.99), 'weight_decay': 5e-4},
    }),

    'transformer': mnm_img_augmentor,
        
    'model': unet,
    'loss': seg_loss,
}) 

baseline_mnm_2p_config = baseline_mnm_all_config.copy({
    'name': 'baseline_mnm_tr2',
    'max_iter': 10000,
    'dataset': mnm_img_dataset.copy({
        'num_patients': 2,
        'has_patient_set': True,
    }),
})

pt_mnm_all_config = baseline_mnm_all_config.copy({
    'name': 'mnm_trall',
    'num_classes': 4,
    'scheme': 'proposed',
    
    'mixup_alpha': 0,

    'two_dataset': {'labeled':mnm_img_dataset, 
                    'unlabeled': mnm_dataset,
                    },
    'dataset':mnm_img_dataset,
    'img_size': (192,192),

    'max_dist': 8,
    'num_samples': 3,
    'num_subjects': 2,

    'max_iter': 10000,
    'lr_steps': (),

    'best_metric': ('dice', True),  
    'tovis_metrics' : ('dice', ),
    'toevl_metrics' : {'dice':('VolDiceCalc', {'mission': 'seg'}),
                       'dist': ('HanDistCalc', {'mission': 'seg'}),
                       'assd': ('AssDistCalc', {'mission': 'seg'}) },

    'post_processes': {},

    'optimizer': adam.copy({
        'args': {'lr': 1e-3, 'eps': 1e-8, 'betas': (0.9, 0.99), 'weight_decay': 5e-4},
    }),

    'transformer': {'labeled': mnm_img_augmentor, 'unlabeled': mnm_vol_augmentor},
    
    'model': SimSiam_DSAG,
    'loss': bootstrap_loss.copy({
        'seg_loss_alpha': 2.0,
        'pred_loss_alpha': 2.0,
        'fbc_loss_alpha': 0.25,
    }),
})

pt_mnm_2p_config = pt_mnm_all_config.copy({
    'name': 'mnm_tr2',
    'two_dataset': {'labeled':mnm_img_dataset.copy({
                        'num_patients': 2,
                        'has_patient_set': True,}), 
                    'unlabeled': mnm_dataset,
                },
    'max_iter': 10000,

    'loss_warmup': {'keys':('fbc', 'pred'), 
                    'until': 1000,},
})

# ------------------------------------------  CAMUS ----------------------------------------------- #
baseline_a2c_all_config = Config({
    'name': 'baseline_camus_a2c_trall',
    'num_classes': 3+1,
    'scheme': 'normal',

    'mixup_alpha': 0,

    'dataset': camus_a2c_img_dataset,
    'img_size': (256, 256),

    'max_iter': 20000,

    'lr_schedule': 'cos',
    'gamma': 0.1,
    'lr_steps': (),
    'lr_warmup_until': 0,

    'best_metric': ('dice', True),  
    'tovis_metrics' : ('dice', ),
    'toevl_metrics' : {'dice':('VolDiceCalc', {'mission': 'seg'}),
                       'dist': ('HanDistCalc', {'mission': 'seg'}),
                       'assd': ('AssDistCalc', {'mission': 'seg'}) },

    'post_processes': {},
    
    'optimizer': adam.copy({
        'args': {'lr': 1e-3, 'eps': 1e-8, 'betas': (0.9, 0.99), 'weight_decay': 5e-4},
    }),

    'transformer': camus_img_augmentator,
        
    'model': unet,
    'loss': seg_loss,
})

baseline_a2c_8p_config = baseline_a2c_all_config.copy({
    'name': 'baseline_camus_a2c_tr8',
    'max_iter': 10000,
    'dataset': camus_a2c_img_dataset.copy({
        'num_patients': 8,
        'has_patient_set': True,
    }),
})

pt_a2c_all_config = baseline_a2c_all_config.copy({
    'name': 'camus_a2c_trall',
    'num_classes': 4,
    'scheme': 'proposed',
    
    'mixup_alpha': 0,

    'two_dataset': {'labeled':camus_a2c_img_dataset, 
                    'unlabeled': camus_a2c_dataset,
                    },
    'dataset':camus_a2c_img_dataset,

    'max_dist': 6,
    'num_samples': 3,
    'num_subjects': 2,

    'loss_warmup': {'keys':('fbc', 'pred'), 
                    'until': 1000,},

    'img_size': (256, 256),
    'max_iter': 40000,
    'lr_steps': (),

    'best_metric': ('dice', True),  
    'tovis_metrics' : ('dice', ),
    'toevl_metrics' : {'dice': ('VolDiceCalc', {'mission': 'seg'}),
                       'dist': ('HanDistCalc', {'mission': 'seg'}),
                       'assd': ('AssDistCalc', {'mission': 'seg'}) },

    'post_processes': {},
    
    'transformer': {'labeled': camus_img_augmentator, 'unlabeled': camus_seq_augmentator},
    
    'model': SimSiam_DSAG,
    'loss': bootstrap_loss,
})

pt_a2c_8p_config = pt_a2c_all_config.copy({
    'name': 'camus_a2c_tr8',
    'scheme': 'proposed',
    
    'two_dataset': {'labeled':camus_a2c_img_dataset.copy({
                        'num_patients': 8,
                        'has_patient_set': True,}), 
                    'unlabeled': camus_a2c_dataset,
            },

    'loss_warmup': {'keys':('fbc'), 
                    'until': 800,
                    },

    'max_iter': 10000,
})

# -------------------------------------------------------------------------------------
pt_cgl_a2c_enc_config = baseline_a2c_all_config.copy({
    'name': "a2c_CglPtrEnc",
    'scheme': 'pretrain_CGL',

    'dataset': camus_a2c_cgl_dataset,
    'img_size': (256, 256),

    'max_iter': 40000,

    'lr_schedule': 'cos',
    'gamma': 0.1,
    'lr_steps': (),
    'lr_warmup_until': 0,

    'best_metric': ('none',),  
    'tovis_metrics' : (),
    'toevl_metrics' : {},
    'post_processes': {},
    
    'transformer': camus_img_augmentator,

    'num_parts': 4,
    'model': cgl_encoder,
    'loss': global_loss,    
})

pt_cgl_a2c_dec_config = pt_cgl_a2c_enc_config.copy({
    'name': 'a2c_CglPtrDec',
    'max_iter': 10000,
    'model': cgl_decoder.copy({
        # using the pretrained weights.
        'backbone':{
            'path': 'path/to/pretrained/weights.pth',
            # architecture configs
            'arch': dict(**cgl_encoder.encoder.architecture),
        }
    }),
    'optimizer': adam_freeze_enc,
    'loss': local_loss,
})

ft_cgl_a2c_8p_config = baseline_a2c_all_config.copy({
    'name': "ft_camus_a2c_cgl_tr8",
    'scheme': 'normal',

    'max_iter': 10000,
    'dataset': camus_a2c_img_dataset.copy({
        'num_patients': 8,
        'has_patient_set': True,
    }),
    'model': unet.copy({
        'backbone':{
            'path': 'path/to/pretrained/weights.pth',
            'arch': dict(**cgl_encoder.encoder.architecture, **cgl_decoder.decoder.architecture),
            },
    }),
})

# -------------------------------------- Prostate md ----------------------------------

baseline_pst_all_config = Config({
   'name': 'baseline_prostate_trall',
    'num_classes': 2+1,
    'scheme': 'normal',

    'mixup_alpha': 0,

    'dataset': prostate_img_dataset,
    'img_size': (192,192),

    'max_iter': 20000,

    'lr_schedule': 'cos',
    'gamma': 0.1,
    'lr_steps': (),
    'lr_warmup_until': 0,

    'best_metric': ('dice', True),  
    'tovis_metrics' : ('dice', ),
    'toevl_metrics' : {'dice':('VolDiceCalc', {'mission': 'seg'}),
                       'dist': ('HanDistCalc', {'mission': 'seg'}),
                       'assd': ('AssDistCalc', {'mission': 'seg'}) },

    'post_processes': {},
    
    'optimizer': adam.copy({
        'args': {'lr': 1e-3, 'eps': 1e-8, 'betas': (0.9, 0.99), 'weight_decay': 1e-5},
    }),

    'transformer': pst_img_augmentor,
        
    'model': unet,
    'loss': seg_loss,
}) 

baseline_pst_2p_config = baseline_pst_all_config.copy({
    'name': 'baseline_prostate_tr2',
    'max_iter': 10000,

    'dataset': prostate_img_dataset.copy({
        'num_patients': 2,
        'has_patient_set': True,
    }),
})

pt_pst_all_config = baseline_pst_all_config.copy({
    'name': 'prostate_trall',
    'scheme': 'proposed',
    'mixup_alpha': 0,

    'two_dataset': {'labeled':prostate_img_dataset, 
                    'unlabeled': prostate_dataset,
                    },
    'dataset':prostate_img_dataset,

    # 'lr_schedule': 'step',
    # 'lr_steps': (6000,8000),

    'max_dist': 6,
    'num_samples': 3,
    'num_subjects': 2,

    'img_size': (192,192),
    'max_iter': 40000,

    'best_metric': ('dice', True),  
    'tovis_metrics' : ('dice', ),
    'toevl_metrics' : {'dice': ('VolDiceCalc', {'mission': 'seg'}),
                       'dist': ('HanDistCalc', {'mission': 'seg'}),
                       'assd': ('AssDistCalc', {'mission': 'seg'}) },

    'post_processes': {},

    'optimizer': adam.copy({
        'args': {'lr': 1e-3, 'eps': 1e-8, 'betas': (0.9, 0.99), 'weight_decay': 1e-5},
    }),

    'transformer': {'labeled': pst_img_augmentor, 'unlabeled': pst_vol_augmentor},
    
    'model': SimSiam_DSAG,
    'loss': bootstrap_loss.copy({
        'seg_loss_alpha': 1.0,
        'use_ce_loss': True,
    }),
})

pt_pst_2p_config = pt_pst_all_config.copy({
    'name': 'prostate_tr2',
    'two_dataset': {'labeled':prostate_img_dataset.copy({
                        'num_patients': 2,
                        'has_patient_set': True,}), 
                    'unlabeled': prostate_dataset,
            },
    'loss_warmup': {'keys':('fbc', 'pred'),
                    'until': 1000,
                    },

    'max_iter': 10000,
})


# Default config
cfg = pt_acdc_2p_config.copy()

def set_cfg(config_name: str):
    """ Sets the active config. Works even if cfg is already imported! """
    global cfg

    cfg.replace(eval(config_name))

def set_dataset(dataset_name: str):
    """ Sets the dataset of the current config. """
    cfg.dataset = eval(dataset_name)

def set_max_iter(max_iter:int):
    cfg.max_iter = max_iter


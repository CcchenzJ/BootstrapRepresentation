from .cfg_class import *

net = Config({
    'type': '', 
    'backbone': None, 
    'freeze_bn': False, 
})

unet = net.copy({
    'type':'UNet', 
    # resume weights path here. 
    'backbone': None, 

    'in_channels': 1,
    'freeze_bn': False, 
    'zero_init_gamma': False,

    'encoder': Config({
        'activation_func': torch.nn.ReLU(True),
        'normalization_func': torch.nn.BatchNorm2d,
        'architecture':{ 
            # define a layer including some convolution blocks. [(OUT_DIM / LAYER OPTION, KERNEL_SIZE, {OTHER PARAMS}), ...]
            'conv0': [(16, 3, {})]*2,  
            'conv1': [('mxp', 2, {})] + [( 32, 3, {})]*2, 
            'conv2': [('mxp', 2, {})] + [( 64, 3, {})]*2, 
            'conv3': [('mxp', 2, {})] + [(128, 3, {})]*2,
            'conv4': [('mxp', 2, {})] + [(128, 3, {})]*2,
            'conv5': [('mxp', 2, {})] + [(128, 3, {})]*2,
        }
    }),

    'decoder': Config({
        'activation_func': torch.nn.ReLU(True),
        'normalization_func': torch.nn.BatchNorm2d,
        'architecture':{
            'deconv5': [(None, -2, {}), (128, 3, {})] + [('sct', ['conv4'], [128])] + [(128, 3, {})]*2,
            'deconv4': [(None, -2, {}), (128, 3, {})] + [('sct', ['conv3'], [128])] + [(128, 3, {})]*2,
            'deconv3': [(None, -2, {}), ( 64, 3, {})] + [('sct', ['conv2'], [ 64])] + [( 64, 3, {})]*2,
            'deconv2': [(None, -2, {}), ( 32, 3, {})] + [('sct', ['conv1'], [ 32])] + [( 32, 3, {})]*2,
            'deconv1': [(None, -2, {}), ( 16, 3, {})] + [('sct', ['conv0'], [ 16])] + [( 16, 3, {})]*2,
            'deconv0': [( 16, 3, {})]*1,
        }
    }),
    'to_shortcut_scopes' : ('deconv5', 'deconv4', 'deconv3', 'deconv2', 'deconv1'),
})

SimSiam_DSAG = unet.copy({
    'type': 'SimSiam', 

    'select_encoder_scope': ['conv5'],
    'select_decoder_scope': ['deconv3'],
    
    'attention_type': 'DSAGPredictor',
    'att_channels': (128, 128),
})

SimSiam_NL = SimSiam_DSAG.copy({
    'attention_type': 'NonLocal',
})
SimSiam_MLP = SimSiam_DSAG.copy({
    'attention_type': 'MLPPredictor',
    'att_channels': (128, 512, 128),
})

cgl_encoder = unet.copy({
    'type':'PretrainEncoder', 

    'featuresize': 6*6,
    'proj_channels': [1024, 128],

})

cgl_decoder = unet.copy({
    'type':'PretrainDecoder', 

    'decoder': Config({
        'activation_func': torch.nn.ReLU(True),
        'normalization_func': torch.nn.BatchNorm2d,
        'architecture':{

        }
    }),
    'proj_channels': [128, 128],
})

vgg13net = net.copy({
    'type': 'UNet',
    'in_channels': 1,
    'freeze_bn': False, 
    'zero_init_gamma': False,

    'encoder': Config({
        'activation_func': torch.nn.ReLU(True),
        'normalization_func': torch.nn.BatchNorm2d,
        'architecture':{ 
            'conv0': [(64, 3, {})]*2,  
            'conv1': [('mxp',2,{})] + [(128, 5, {})]*2, 
            'conv2': [('mxp',2,{})] + [(256, 3, {})]*2, 
            'conv3': [('mxp',2,{})] + [(512, 3, {})]*2,
            'conv4': [('mxp',2,{})] + [(512, 3, {})]*2,
            'conv5': [('mxp',2,{})],
        }
    }),

    'decoder': Config({
        'activation_func': torch.nn.ReLU(True),
        'normalization_func': torch.nn.BatchNorm2d,
        'architecture':{
            'deconv5': [(512, 3, {})]*2 + [(None, -2, {})] + [(512, 3, {})],
            'deconv4': [('sct', ['conv4'], [512], {})] + [(512, 3, {})]*2 + [(None, -2, {}), (512, 3, {})],
            'deconv3': [('sct', ['conv3'], [512], {})] + [(512, 3, {})]*2 + [(None, -2, {}), (256, 3, {})],
            'deconv2': [('sct', ['conv2'], [256], {})] + [(256, 3, {})]*2 + [(None, -2, {}), (128, 3, {})],
            'deconv1': [('sct', ['conv1'], [128], {})] + [(128, 3, {})]*2 + [(None, -2, {}), ( 64, 3, {})],
            'deconv0': [('sct', ['conv0'], [ 64], {})] + [( 64, 3, {})]*3,
        }
    }),
    'to_shortcut_scopes' : ('deconv4', 'deconv3', 'deconv2', 'deconv1', 'deconv0'),
})

cgl_encoder_vgg = vgg13net.copy({
    'type':'PretrainEncoder', 

    'featuresize': 6*6,
    'proj_channels': [1024, 512],
})
cgl_decoder_vgg = vgg13net.copy({
    'type':'PretrainDecoder', 

    'decoder': Config({
        'activation_func': torch.nn.ReLU(True),
        'normalization_func': torch.nn.BatchNorm2d,
        'architecture':{
            'deconv5': [(512, 3, {})]*2 + [(None, -2, {})] + [(512, 3, {})],
            'deconv4': [('sct', ['conv4'], [512], {})] + [(512, 3, {})]*2 + [(None, -2, {}), (512, 3, {})],
            'deconv3': [('sct', ['conv3'], [512], {})] + [(512, 3, {})]*2 + [(None, -2, {}), (256, 3, {})],
        }
    }),

    'proj_channels': [256, 256],
})

res18netv2 = net.copy({
    'type': 'UNet',
    'in_channels': 1,
    'freeze_bn': False, 
    'zero_init_gamma': False,

    'encoder': Config({
        'activation_func': torch.nn.ReLU(True),
        'normalization_func': torch.nn.BatchNorm2d,
        'architecture':{ 
            'conv0': [(64, 3, {})]*2,
            'conv1': [('mxp',3, {'stride':2, 'padding':1})] + [('res', 64, {})]*1, 
            'conv2': [('mxp',3, {'stride':2, 'padding':1})] + [('res', 64, {})]*1, 
            'conv3': [('res', 128, {'stride':2})]*1, 
            'conv4': [('res', 256, {'stride':2})]*1,
            'conv5': [('res', 512, {'stride':2})]*1,
        }
    }),

    'decoder': Config({
        'activation_func': torch.nn.ReLU(True),
        'normalization_func': torch.nn.BatchNorm2d,
        'architecture':{
            'deconv5': [(None, -2, {})] + [(512, 3, {})],
            'deconv4': [('sct', ['conv4'], [256], {})] + [(512, 3, {})]*2 + [(None, -2, {}), (512, 3, {})],
            'deconv3': [('sct', ['conv3'], [128], {})] + [(256, 3, {})]*2 + [(None, -2, {}), (128, 3, {})],
            'deconv2': [('sct', ['conv2'], [ 64], {})] + [(128, 3, {})]*2 + [(None, -2, {}), ( 64, 3, {})],
            'deconv1': [('sct', ['conv1'], [ 64], {})] + [( 64, 3, {})]*3 + [(None, -2, {}), ( 64, 3, {})],
            'deconv0': [('sct', ['conv0'], [ 64], {})] + [( 64, 3, {})]*2,
        }
    }),
    'to_shortcut_scopes' : ('deconv4', 'deconv3', 'deconv2', 'deconv1', 'deconv0'),
})

SimSiam_DSAG_vgg = vgg13net.copy({
    'type': 'SimSiam', 

    'select_encoder_scope': ['conv5'],
    'select_decoder_scope': ['deconv3'],
    
    'attention_type': 'DSAGPredictor',
    'att_channels': (512, 512),
})
SimSiam_DSAG_res = res18netv2.copy({
    'type': 'SimSiam', 

    'select_encoder_scope': ['conv5'],
    'select_decoder_scope': ['deconv3'],
    
    'attention_type': 'DSAGPredictor',
    'att_channels': (512, 512),
})

cgl_encoder_res = res18netv2.copy({
    'type':'PretrainEncoder', 

    'featuresize': 6*6,
    'proj_channels': [1024, 512],
})
cgl_decoder_res = res18netv2.copy({
    'type':'PretrainDecoder', 

    'decoder': Config({
        'activation_func': torch.nn.ReLU(True),
        'normalization_func': torch.nn.BatchNorm2d,
        'architecture':{
            'deconv5': [(None, -2, {})] + [(512, 3, {})],
            'deconv4': [('sct', ['conv4'], [256], {})] + [(512, 3, {})]*2 + [(None, -2, {}), (512, 3, {})],
            'deconv3': [('sct', ['conv3'], [128], {})] + [(256, 3, {})]*2 + [(None, -2, {}), (128, 3, {})],
        }
    }),

    'proj_channels': [128, 256],
})
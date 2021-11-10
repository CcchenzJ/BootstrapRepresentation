
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict

from backbone import *

class PretrainEncoder(ModelTemplate):
    """ Encoder to pretrain using CGL.global loss """
    def __init__(self, conf, num_classes):
        super().__init__(conf)
        
        self.encoder, out_enc_ch = make_net(conf.in_channels, 
                                            conf.encoder.architecture, 
                                            conf.encoder.activation_func,
                                            conf.encoder.normalization_func)
        
        self.projection = nn.Sequential(
            nn.Linear(conf.featuresize * out_enc_ch, conf.proj_channels[0], bias=False),
            nn.ReLU(True), 
            nn.Linear(*conf.proj_channels, bias=False)
        )

    def forward(self, x):

        x = self.layer_forward('encoder', x)
        enc = x

        x = torch.flatten(x, 1)
        x = self.projection(x)
        proj = x

        pred_outs = {}

        pred_outs['proj'] = proj
        
        if self.training:
            return pred_outs
        else:
            return pred_outs

if __name__ == '__main__':

    import sys
    from config import set_cfg, cfg
    if len(sys.argv) > 1:
        set_cfg(sys.argv[1])
    
    net = PretrainEncoder(cfg.model, cfg.num_classes)
    net.train()
    net.init_weights()
    
    net = net.cuda()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    x = torch.zeros((1,1, 192, 192))
    y = net(x)

    # for m in net.modules():
    #     print(m)
    print([p[0] for p in net.named_parameters() if p[1].requires_grad is False])
    print([p[0] for p in net.named_parameters() if p[1].requires_grad is True])
    print()
    for k, a in y.items():
        if isinstance(a, list):
            for ii in a:
                print(k + ': ', ii.size(), torch.sum(ii))
        else:
            print(k + ': ', a.size(), torch.sum(a))
    exit()
    

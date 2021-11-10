
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict

from backbone import *

class UNet(ModelTemplate):
    
    def __init__(self, conf, num_classes):
        super().__init__(conf)
        
        self.encoder, out_enc_ch = make_net(conf.in_channels, 
                                            conf.encoder.architecture, 
                                            conf.encoder.activation_func,
                                            conf.encoder.normalization_func)
        
        self.decoder, out_dec_ch = make_net(out_enc_ch, 
                                            conf.decoder.architecture, 
                                            conf.decoder.activation_func,
                                            conf.decoder.normalization_func)

        self.segmentor = nn.Conv2d(out_dec_ch, num_classes, 3, padding=1, bias=False)        

    def prepare_shortcut(self, encs):
        for scope, layer in self.decoder.items():
            if scope in self.conf.to_shortcut_scopes:
                for module in layer.modules():
                    if isinstance(module, ShortCut):
                        module.prepare(encs)

    def forward(self, x):

        encs = {}
        for scope, layer in self.encoder.items():
            x = layer(x)
            encs[scope] = x

        self.prepare_shortcut(encs)
        
        decs = {}
        for scope, layer in self.decoder.items():
            x = layer(x)
            decs[scope] = x
         
        pred_outs = {}

        pred_outs['seg'] = self.segmentor(x)
        
        if self.training:
            return pred_outs
        else:
            pred_outs['seg'] = torch.softmax(pred_outs['seg'], 1)
            pred_outs['fea'] = dict(**encs, **decs)
            return pred_outs

if __name__ == '__main__':

    import sys
    from config import set_cfg, cfg
    if len(sys.argv) > 1:
        set_cfg(sys.argv[1])
    
    net = UNet(cfg.model, cfg.num_classes)
    net.train()
    net.init_weights()
    
    net = net.cuda()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    x = torch.zeros((24,1, *cfg.img_size))
    y = net(x)

    # for m in net.modules():
    #     print(m)
    print([p[0] for p in net.named_parameters() if p[1].requires_grad is False])
    print([p[0] for p in net.named_parameters() if p[1].requires_grad is True])
    print()
    for k, a in y.items():
        if isinstance(a, list) or isinstance(a, tuple):
            for ii in a:
                print(k + ': ', ii.size(), torch.sum(ii))
        else:
            print(k + ': ', a.size(), torch.sum(a))
    exit()
    

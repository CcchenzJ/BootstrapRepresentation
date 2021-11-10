
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict

from backbone import *
from config import cfg

class BYOL(ModelTemplate):
    """ Adapted from https://github.com/facebookresearch/moco """
    def __init__(self, conf, num_classes):
        super().__init__(conf)
        
        self.encoder_q, out_enc_ch = make_net(conf.in_channels, 
                                            conf.encoder.architecture, 
                                            conf.encoder.activation_func,
                                            conf.encoder.normalization_func)
        self.encoder_k, out_enc_ch = make_net(conf.in_channels, 
                                            conf.encoder.architecture, 
                                            conf.encoder.activation_func,
                                            conf.encoder.normalization_func)

        self.m = 0.999
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False 

        self.decoder, out_dec_ch = make_net(out_enc_ch, 
                                            conf.decoder.architecture, 
                                            conf.decoder.activation_func,
                                            conf.decoder.normalization_func)

        if conf.attention_type in ('DSAGPredictor',):
            self.attention = eval(conf.attention_type)(*conf.att_channels, cfg.max_dist)
        else:
            self.attention = eval(conf.attention_type)(*conf.att_channels)
        self.segmentor = nn.Conv2d(out_dec_ch, num_classes, 3, padding=1, bias=False) 

    def prepare_shortcut(self, encs):
        for scope, layer in self.decoder.items():
            if scope in self.conf.to_shortcut_scopes:
                for module in layer.modules():
                    if isinstance(module, ShortCut):
                        module.prepare(encs)

    def forward(self, x):
        k = x

        encs = {}
        for scope, layer in self.encoder_q.items():
            x = layer(x)
            encs[scope] = x

        with torch.no_grad():
            self._momentum_update_key_encoder()

            encs_key = {}
            for scope, layer in self.encoder_k.items():
                k = layer(k)
                encs_key[scope] = k
       
        self.prepare_shortcut(encs)
        decs = {}
        for scope, layer in self.decoder_q.items():
            if scope in self.conf.to_shortcut_scopes:
                x = [x, encs]
            x = layer(x)
            decs[scope] = x

        pred_outs = {}      
        pred_outs['att'] = self.attention,

        pred_outs['enc'] = torch.cat([encs[self.conf.select_encoder_scope], \
                                      encs_key[self.conf.select_encoder_scope]], 0)
        pred_outs['dec'] = decs[self.conf.select_decoder_scope]

        pred_outs['seg'] = self.segmentor(x)

        if self.training:
            return pred_outs
        else:
            pred_outs['fea'] = dict(**encs, **decs)
            pred_outs['fea']['enc'] = pred_outs['enc']
            pred_outs['fea']['dec'] = pred_outs['dec']
            pred_outs['fea']['seg'] = pred_outs['seg']

            pred_outs['seg'] = torch.softmax(pred_outs['seg'], 1)
            return pred_outs

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """ Momentum update of the key encoder """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)


if __name__ == '__main__':

    import sys
    from config import set_cfg, cfg
    if len(sys.argv) > 1:
        set_cfg(sys.argv[1])
    
    net = BYOL(cfg.model, cfg.num_classes)
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
    

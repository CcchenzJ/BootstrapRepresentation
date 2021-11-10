
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict

from backbone import *
from config import cfg

class SimSiam(ModelTemplate):
    """ Bootstrap representation using SimSiam like architecture. 
        see https://arxiv.org/abs/2011.10566 """
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

        pred_outs['att'] = self.attention

        pred_outs['enc'] = [encs[scope].clone() for scope in self.conf.select_encoder_scope]
        pred_outs['dec'] = [decs[scope].clone() for scope in self.conf.select_decoder_scope]

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

if __name__ == '__main__':

    import sys
    from config import set_cfg, cfg
    if len(sys.argv) > 1:
        set_cfg(sys.argv[1])

    net = SimSiam(cfg.model, cfg.num_classes)
    net.train()
    net.init_weights()

    net = net.cuda()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    x = torch.zeros((24,1, *cfg.img_size))
    y = net(x)

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


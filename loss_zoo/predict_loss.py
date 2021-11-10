import itertools
import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from config import cfg

class BootstrapLoss(nn.Module):
    """Bootstrap Loss Function for representation learning.
    Compute Targets:
        1) 'S': Segmentation Loss, (defualt: Dice loss) by
        the prediction of D_L from decoder and its corresponding groundtruth.
        2) 'C': Contrastive Loss (fbc), using InfoNCE to contrast
        the foreground/background features from D_L and 
        the foreground features from D_U.
        3) 'P': Slice Prediction Loss (pred), using cosine similarity loss 
        to matching the representation from 2 slices.
    """
    def __init__(self, conf, num_classes):
        super().__init__()
        self.conf = conf
        self.num_classes = num_classes
        
    def forward(self, preds, gt, dist=None):
        num_unlabeled = cfg.num_samples * cfg.num_subjects

        seg_labeled = preds['seg'][:-num_unlabeled]
        seg_unlabeled = preds['seg'][-num_unlabeled:]

        att = preds['att']

        enc = preds['enc'][0][-num_unlabeled:]
        dec = preds['dec'][0]
 
        losses = {}

        if self.conf.use_seg_loss:
            if cfg.mixup_alpha > 0:
                losses['S'] = self.segmentation_mixup_loss(seg_labeled, gt) \
                            * self.conf.seg_loss_alpha
            else:
                losses['S'] = self.segmentation_loss(seg_labeled, gt) \
                            * self.conf.seg_loss_alpha

        if self.conf.use_fbc_loss:
            if cfg.mixup_alpha > 0: 
                gt = gt[-1]
            losses['C'] = self.FB_calibration_loss(dec, seg_unlabeled, gt, self.conf.tau) \
                        * self.conf.fbc_loss_alpha

        if self.conf.use_pred_loss:
            losses['P'] = self.slice_prediction_loss(att, enc, dist) \
                        * self.conf.pred_loss_alpha
        return losses

    def segmentation_loss(self, x, y):
        loss_s = 0.
        
        if self.conf.use_dice_loss:
            onehot_y = self._onehot(y)
            x = torch.softmax(x, 1)
            loss_s += self._dice_loss(x, onehot_y) / x.size(0)

        if self.conf.use_ce_loss:
            loss_s += F.cross_entropy(x, y.long(), reduction='mean')

        return loss_s

    def segmentation_mixup_loss(self, x, y):
        y_a, y_b, lam, _ = y

        loss_s = lam * self.segmentation_loss(x, y_a) \
               + (1. - lam) * self.segmentation_loss(x, y_b)
        return loss_s

    def slice_prediction_loss(self, att, enc, dist):
        loss_p = 0.
        
        q, k = enc, enc.clone().detach()
        for idx in range(q.size(0)):
            relative_dist = dist - dist[idx]

            cur_vdx = (idx//cfg.num_samples) * cfg.num_samples
            nxt_vdx = (idx//cfg.num_samples+1) * cfg.num_samples
            isWithin = torch.LongTensor([1]*len(relative_dist))
            isWithin[cur_vdx:nxt_vdx] = 0
            p = att(q[None, idx], relative_dist, isWithin)

            for kdx in range(k.size(0)):
                if kdx == idx: continue

                fea_q = p[kdx].permute(1,2,0).flatten(0,1)
                fea_k = k[kdx].permute(1,2,0).flatten(0,1)

                loss_p += self._negative_cosine_similarity_loss(fea_q, fea_k)

        return loss_p / q.size(0) / (k.size(0)-1)

    def slice_prediction_nodist_loss(self, att, enc, dist=None):
        """ using nonlocal as predictor. """
        loss_p = 0.
        
        q, k = att(enc), enc.clone().detach()
        for idx in range(q.size(0)):
            fea_q = q[idx].permute(1,2,0).flatten(0,1)

            for kdx in range(k.size(0)):
                if kdx == idx: continue
                fea_k = k[kdx].permute(1,2,0).flatten(0,1)
                loss_p += self._negative_cosine_similarity_loss(fea_q, fea_k)

        return loss_p / q.size(0) / (k.size(0)-1)

    def slice_prediction_within_loss(self, att, enc, dists):
        """ only within-volume slice prediction """
        loss_p = 0.
        
        for vdx in range(cfg.num_subjects):
            dist = dists[vdx*cfg.num_samples:(vdx+1)*cfg.num_samples]
            q = enc[vdx*cfg.num_samples: (vdx+1)*cfg.num_samples]
            k = enc[vdx*cfg.num_samples: (vdx+1)*cfg.num_samples].clone().detach()
            for idx in range(q.size(0)):
                relative_dist = dist - dist[idx]
                att_q = att(q[None, idx], relative_dist)

                for kdx in range(k.size(0)):
                    if kdx == idx: continue

                    fea_q = att_q[kdx].permute(1,2,0).flatten(0,1)
                    fea_k = k[kdx].permute(1,2,0).flatten(0,1)

                    loss_p += self._negative_cosine_similarity_loss(fea_q, fea_k)

        return loss_p / q.size(0) / (k.size(0)-1) / cfg.num_subjects

    def slice_prediction_within_nodist_loss(self, att, enc, dists=None):
        """ only within-volume slice prediction and using nonlocal as predictor """
        loss_p = 0.
        
        for vdx in range(cfg.num_subjects):
            q = att(enc[vdx*cfg.num_samples:(vdx+1)*cfg.num_samples])
            k = enc[vdx*cfg.num_samples:(vdx+1)*cfg.num_samples].clone().detach()
            for idx in range(q.size(0)):
                fea_q = q[idx].permute(1,2,0).flatten(0,1)
                for kdx in range(k.size(0)):
                    if kdx == idx: continue
                    fea_k = k[kdx].permute(1,2,0).flatten(0,1)

                    loss_p += self._negative_cosine_similarity_loss(fea_q, fea_k)

        return loss_p / q.size(0) / (k.size(0)-1) / cfg.num_subjects

    def slice_prediction_within_mean_loss(self, att, enc, dists=None):
        """ only within-volume slice prediction and using mean teacher as target """
        loss_p = 0.
        
        for vdx in range(cfg.num_subjects):
            q = att(enc[vdx*cfg.num_samples:(vdx+1)*cfg.num_samples])
            k = enc[vdx*cfg.num_samples:(vdx+1)*cfg.num_samples].clone().detach()
            k_mean = k.mean(0)
            for idx in range(q.size(0)):
                fea_q = q[idx].permute(1,2,0).flatten(0,1)
                fea_k = k_mean.permute(1,2,0).flatten(0,1)

                loss_p += self._negative_cosine_similarity_loss(fea_q, fea_k)

        return loss_p / q.size(0) / cfg.num_subjects

    def slice_prediction_across_loss(self, att, enc, dists):
        """ only across-volume slice prediction """
        loss_p = 0.
        
        q, k = enc, enc.clone().detach()
        for idx in range(q.size(0)):
            cur_vdx = idx // cfg.num_samples
            dist, kk = [], []
            for vdx in range(cfg.num_subjects):
                if vdx == cur_vdx: continue
                dist += [dists[vdx*cfg.num_samples:(vdx+1)*cfg.num_samples]]
                kk += [k[vdx*cfg.num_samples:(vdx+1)*cfg.num_samples]]
            kk = torch.cat(kk, 0)
            dist = torch.cat(dist, 0)

            relative_dist = dist - dists[idx]
            att_q = att(q[None, idx], relative_dist)
            
            for kdx in range(kk.size(0)):
                fea_q = att_q[kdx].permute(1,2,0).flatten(0,1)
                fea_k = kk[kdx].permute(1,2,0).flatten(0,1)

                loss_p += self._negative_cosine_similarity_loss(fea_q, fea_k)

        return loss_p / q.size(0) / kk.size(0)

    def FB_calibration_loss(self, dec, seg, gt, tau):
        def check_is_empty(vv):
            return [v for v in vv if v.size(0) != 0]

        loss_i = 0.
        
        q = dec[-cfg.num_samples*cfg.num_subjects:]
        k = dec[:-cfg.num_samples*cfg.num_subjects]

        with torch.no_grad():
            gt_resize = F.interpolate(gt.unsqueeze(1), k.size()[-2:], mode='nearest')

        topk = 0
        foreground, background = [], []
        for idx in range(k.size(0)):
            fea_k = k[idx].permute(1,2,0).flatten(0,1)
            mask  = gt_resize[idx].flatten(0)
            foreground += [fea_k[(mask != 0)].sum(0, keepdim=True)]
            background += [fea_k[(mask == 0)].mean(0, keepdim=True)]
            topk += fea_k[(mask!=0)].size(0)
        background = torch.cat(background, 0)
        
        topk = int(topk / k.size(0))
        
        with torch.no_grad():
            seg_resize = F.interpolate(torch.softmax(seg, 1), k.size()[-2:], mode='nearest')

        candidates = []
        for idx in range(q.size(0)):
            fea_q = q[idx].permute(1,2,0).flatten(0,1)
            mask_unl = seg_resize[idx][1:].permute(1,2,0).flatten(0,1)
            top_k = mask_unl.mean(1).topk(topk, largest=True, dim=0)[1]
            candidates += [fea_q[top_k].sum(0, keepdim=True)]

        for cnt2, pos_pair in enumerate(itertools.combinations(
                                        check_is_empty(candidates+foreground), 2)):
            loss_i += self._NCE(pos_pair[0], pos_pair[1], background, tau)
            loss_i += self._NCE(pos_pair[1], pos_pair[0], background, tau)

        return loss_i / 2. / (cnt2 + 1.)

    def _NCE(self, fea_q, fea_k, fea_neg, tau):
        fea_q = F.normalize(fea_q, 1)
        fea_k = F.normalize(fea_k, 1)

        l_pos = torch.einsum('nc,ck->nk', fea_q, fea_k.t())
        l_neg = torch.einsum('nc,ck->nk', fea_q, fea_neg.t())

        logit = torch.cat([l_pos, l_neg], dim=1) / tau
        label = torch.zeros(logit.size(0), dtype=torch.long)

        loss  = F.cross_entropy(logit, label, reduction='mean')
        return loss

    def _onehot(self, masks):
        y_onehot = []
        with torch.no_grad():
            for _cls in range(self.num_classes):
                onehot = torch.zeros_like(masks, requires_grad=False, device=masks.device)
                onehot[(masks==_cls)] = 1
                y_onehot += [onehot]
        return torch.stack(y_onehot, dim=1)

    def _dice_loss(self, pred, mask, smooth=1e-7):
        from torch import einsum
        intersection: torch.Tensor = einsum("bcxy, bcxy->bc", pred, mask)
        union:        torch.Tensor = einsum("bcxy->bc", pred) + einsum("bcxy->bc", mask)
        divided:      torch.Tensor = 1. - (2 * intersection + smooth) / (union + smooth)
        dc = divided.sum()
        return dc / self.num_classes

    def _negative_cosine_similarity_loss(self, fea_q, fea_k):
        norm_q = F.normalize(fea_q, 1)
        norm_k = F.normalize(fea_k, 1)

        return - (norm_k * norm_q).sum(dim=1).mean()

if __name__ == '__main__':

    import sys
    from config import set_cfg, cfg
    if len(sys.argv) > 1:
        set_cfg(sys.argv[1])
    
    from model_zoo import *
    net = eval(cfg.model.type)(cfg.model, cfg.num_classes)
    net.train()
    net.init_weights()    
    
    net = net.cuda()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    criterion = BootstrapLoss(cfg.loss, cfg.num_classes)
    
    x = torch.ones((12,1,*cfg.img_size)).cuda()
    m = torch.ones((6,*cfg.img_size)).cuda() 

    y = net(x)
    l = criterion(y, m)

    print()
    for k, a in l.items():
        print(k + ': ', a.size(), torch.sum(a))
    exit()
    

import itertools
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from config import cfg

class ContrastiveLoss(nn.Module):
    """Contrastive Loss Function
    Compute Targets:
        1) 'C': Cosine similarity lossb by matching 
        encoding features and predict features.
    """

    def __init__(self, conf, num_classes):
        super().__init__()
        self.conf = conf
        self.num_classes = num_classes
        
    def forward(self, preds):

        encs  = preds['enc']
        projs = preds['proj']

        losses = {}

        if self.conf.use_cossim_loss:
            losses['C'] = self.cosine_similarity_loss(encs, projs) \
                        * self.conf.cossim_loss_alpha
        return losses

    def cosine_similarity_loss(self, encs, projs):
        loss_c = 0.
        
        z1, z2 = encs
        p1, p2 = projs

        loss_c = self._negative_cosine_similarity_loss(p1, z2.detach()) \
               + self._negative_cosine_similarity_loss(p2, z1.detach())

        return loss_c / 2.

    def _negative_cosine_similarity_loss(self, fea_q, fea_k):
        norm_q = F.normalize(fea_q, 1)
        norm_k = F.normalize(fea_k, 1)

        return - (norm_k * norm_q).sum(dim=1).mean()

class CGLContrastiveLoss(nn.Module):
    """Contrastive of Global and Local Loss Function
    Compute Targets:
        1) 'G': Global loss, the positive set is obtained by images 
        from corresponding partitions across volumes. 
        The negative set is obtained by images only from other partitions.
        2) 'L': Local loss, the positive set is obtained by representations
        from corresponding local regions across volumes. 
        The negative set is obtained by representations from the remaining local regions.

        Args:
            tau: temperature parameters.
            num_parts: the number of partitions.
            region_size: the matching window size of the local loss.
    """

    def __init__(self, conf, num_classes):
        super().__init__()
        self.conf = conf
        self.num_classes = num_classes
        
    def forward(self, preds):
    
        proj = preds['proj']

        losses = {}

        # losess['G']
        if self.conf.use_global_loss:
            losses['G'] = self.global_loss(proj, self.conf.tau) \
                        * self.conf.global_loss_alpha
                        
        if self.conf.use_local_loss:
            losses['L'] = self.local_loss(proj, self.conf.tau) \
                        * self.conf.local_loss_alpha
        return losses

    def global_loss(self, proj, tau):
        loss_c = 0.

        batch_size = proj.size(0) // 3
        rawx = proj[:batch_size]
        aug1 = proj[batch_size:-batch_size]        
        aug2 = proj[-batch_size:]

        for idx in range(batch_size):
            pos_idx, neg_idx = [], list(range(batch_size))

            for cur_pos_idx in range(idx%cfg.num_parts, batch_size, cfg.num_parts):
                neg_idx.remove(cur_pos_idx) 
                pos_idx.append(cur_pos_idx)                        

            cur_z_pos = [F.normalize(rawx[pos_idx], dim=1)] \
                      + [F.normalize(aug1[pos_idx], dim=1)] \
                      + [F.normalize(aug2[pos_idx], dim=1)]
            cur_z_neg = torch.cat([F.normalize(rawx[neg_idx], dim=1)] \
                                + [F.normalize(aug1[neg_idx], dim=1)] \
                                + [F.normalize(aug2[neg_idx], dim=1)], dim=0)

            for pos_pair in itertools.combinations(cur_z_pos, 2):
                loss_c += self._NCE(pos_pair[0], pos_pair[1], cur_z_neg, tau)
                loss_c += self._NCE(pos_pair[1], pos_pair[0], cur_z_neg, tau)

        return loss_c / 2. / batch_size

    def local_loss(self, proj, tau):
        loss_c = 0.

        batch_size = proj.size(0) // 3
        rawx = proj[:batch_size]
        aug1 = proj[batch_size:-batch_size]        
        aug2 = proj[-batch_size:]

        for idx in range(batch_size):
            cur_pos_idx = [idx]

            cur_patch_idx = self._sample_region(rawx[None, idx])[1]
            cur_z_regions = [self._sample_region(aug1[cur_pos_idx], cur_patch_idx)[0]] \
                          + [self._sample_region(aug2[cur_pos_idx], cur_patch_idx)[0]]
                        
            for rdx in range(self.conf.sqrt_num_regions**2):
                neg_idx = list(range(self.conf.sqrt_num_regions**2))
                neg_idx.remove(rdx)

                cur_r_pos = [z[None,rdx].flatten(0,1) for z in cur_z_regions]
                cur_r_neg = torch.cat([z[neg_idx].flatten(0,1) \
                                        for z in cur_z_regions], dim=0)

                for pos_pair in itertools.combinations(cur_r_pos, 2):
                    loss_c += self._NCE(pos_pair[0], pos_pair[1], cur_r_neg, tau)
                    loss_c += self._NCE(pos_pair[1], pos_pair[0], cur_r_neg, tau)

        return loss_c / 2. / self.conf.sqrt_num_regions**2 / batch_size

    def _NCE(self, fea_q, fea_k, fea_neg, tau):
        l_pos = torch.einsum('nc,nc->n', [fea_q, fea_k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [fea_q, fea_neg.t()])

        logit = torch.cat([l_pos, l_neg], dim=1) / tau
        label = torch.zeros(logit.size(0), dtype=torch.long)

        loss  = F.cross_entropy(logit, label, reduction='mean')
        return loss

    def _sample_region(self, feat, patch_idx=None):
        gap = (self.conf.region_size-1) // 2

        if patch_idx is None:
            step = feat.size(-1) // (self.conf.sqrt_num_regions+1)
            patch_centre = list(range(0, feat.size(-1)+1, step))
            patch_centre.pop(0)
            patch_centre.pop(-1)
        else:
            patch_centre = patch_idx

        sample_regions = []
        for (centre_x, centre_y) in itertools.product(patch_centre, patch_centre):
            if gap != 0:
                region_index_x = list(range(centre_x-gap,centre_x+gap))
                region_index_y = list(range(centre_y-gap,centre_y+gap))
                sample_region  = feat[...,region_index_y][...,region_index_x,:].flatten(1)
            else:
                sample_region = feat[...,centre_x,centre_y]
            sample_regions.append(F.normalize(sample_region, dim=1))
        sample_regions = torch.stack(sample_regions) # Rx?xCxKxK  Rx?xC
        return sample_regions, patch_centre

if __name__ == '__main__':

    import sys, cv2
    from config import set_cfg, cfg
    if len(sys.argv) > 1:
        set_cfg(sys.argv[1])
    
    from model_zoo import *
    net = eval(cfg.model.type)(cfg.model, cfg.num_classes)
    net.train()
    net.init_weights()    
    
    net = net.cuda()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    criterion = ContrastiveLoss(cfg.loss, cfg.num_classes)
    
    x = torch.ones((12,1,*cfg.img_size)).cuda()
    m = torch.ones((6,*cfg.img_size)).cuda() 

    y = net(x)
    l = criterion(y, m)

    print()
    for k, a in l.items():
        print(k + ': ', a.size(), torch.sum(a))
    exit()
    

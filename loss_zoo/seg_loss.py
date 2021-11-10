
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from config import cfg

class SegLoss(nn.Module):

    def __init__(self, conf, num_classes):
        super().__init__()
        self.conf = conf
        self.num_classes = num_classes
        
    def forward(self, pred, gt):
        
        mask_p = pred['seg']
        mask_t = gt

        losses = {}

        # losess['C', 'D']
        if self.conf.use_ce_loss:
            losses['C'] = self.cross_entropy_loss(mask_p, mask_t) \
                        * self.conf.ce_loss_alpha
        if self.conf.use_dice_loss:
            if cfg.mixup_alpha > 0:
                losses['D'] = self.category_dice_loss_mixup(mask_p, mask_t) \
                            * self.conf.dice_loss_alpha
            else:
                losses['D'] = self.category_dice_loss(mask_p, mask_t) \
                            * self.conf.dice_loss_alpha
        return losses

    def category_dice_loss(self, mask_p, mask_t):
        mask_t = self._onehot(mask_t)
        mask_p = torch.softmax(mask_p, 1)
        loss = self._dice_loss(mask_p, mask_t)
        return loss / mask_p.size(0)

    def category_dice_loss_mixup(self, x, y):
        y_a, y_b, lam, _ = y

        loss_s = lam * self.category_dice_loss(x, y_a) \
               + (1. - lam) * self.category_dice_loss(x, y_b)
        return loss_s

    def cross_entropy_loss(self, mask_p, mask_t):
        if self.conf.use_smooth_label_loss:
            loss_c = self._label_smoothing_ce(mask_p, mask_t.long(), 1e-3)
        else:
            if self.num_classes > 2:
                loss_c = F.cross_entropy(mask_p, mask_t.long(), reduction='mean')
            else:
                loss_c = F.binary_cross_entropy(torch.sigmoid(mask_p), mask_t.unsqueeze(1), reduction='mean')
        return loss_c

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

    def _dice_loss_v2(self, pred, mask, smooth=1e-10):
        intersection = torch.mul(pred, mask).sum([-1,-2])
        union = pred.sum([-1,-2]) + mask.sum([-1,-2])
        divided = 2 * intersection / (union + smooth)
        dc = 1. - divided.mean()
        return dc

    def _label_smoothing_ce(self, pred, mask, epsilon):
        log_probs = F.log_softmax(pred, dim=1)
        nll_loss  = - log_probs.gather(dim=1, index=mask.unsqueeze(1)).squeeze(1)
        
        smooth_loss = - log_probs.mean(dim=1)
        loss = (1-epsilon) * nll_loss + epsilon * smooth_loss
        return loss.mean()

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

    criterion = SegLoss(cfg.loss, cfg.num_classes)
    
    x = torch.ones((2,1,256,256))
    y = net(x)
    
    m = torch.zeros((2,256,256)).cuda()

    l = criterion(y, m)

    print()
    for k, a in l.items():
        print(k + ': ', a.size(), torch.sum(a))
    exit()
    

import os

import numpy as np
import pandas as pd

from abc import abstractclassmethod

# ----------------------------- Template Class ------------------------------------ #

class Metric:
    ''' class for some metrics calculation (pred, mask) '''
    def __init__(self, num_classes, mission, metric:str):
        self.num_classes = num_classes
        self.mission     = mission
        self.metric      = metric
        self.name_cache  = 0
        self.__setattr__(metric+'Dict', {})

    @abstractclassmethod
    def __call__(self, pred, mask, name=None):
        ''' default cl0 is background. '''

        if name is None:
            name = str(self.name_cache)
            self.name_cache += 1
        
        #TODO: Calculation here.

    def ResDict(self):
        return self.__getattribute__(self.metric+'Dict')
    
    def gather(self, metric=''):
        Dict = self.ResDict() if self.metric in metric else {}

        resdict = {k:[] for k in range(1, self.num_classes)}
        for key in Dict.keys():
            for _cls in range(1, self.num_classes):
                res = Dict[key][str(_cls)]
                if res < 0: continue
                resdict[_cls].append(res)
        return resdict

    def save_to_json(self, json_path=''):
        ''' json_path include two parts: {}/{}
            path to save jsonfile ;
            exp_name of the results. '''

        exp_name = os.path.basename(json_path)
        pd.DataFrame({exp_name:self.ResDict()}).to_json(json_path+f'_{self.metric}.json', indent=2)

# ----------------------------------------------------------------------------------------------

class DiceCalc(Metric):
    ''' class for the dice score calculation (pred, mask) '''
    def __init__(self, num_classes, mission, is_modified=True):
        super().__init__(num_classes, mission, 'dice')
        self.is_modified = is_modified

    def __call__(self, pred, mask, name=None):
        ''' default cl0 is background. '''
        if name is None:
            name = str(self.name_cache)
            self.name_cache += 1

        self.diceDict.update({name:{}})
        for cl in range(1, self.num_classes):
            p = (pred == (cl)).astype(np.float).reshape(-1)
            t = (mask == (cl)).astype(np.float).reshape(-1)
            dice = self._dice_calc_v2(p, t)
            self.diceDict[name][str(cl)] = dice

    def _dice_calc(self, pred, mask):
        eps   = 1e-5
        inter = pred @ mask.T
        union = np.sum(pred) + np.sum(mask)
        if self.is_modified:
            t = (2. * inter.astype(np.float) + eps) / (union.astype(np.float)+eps)
        else:
            t = (2. * inter.astype(np.float)) / (union.astype(np.float)+eps)
        return t

    def _dice_calc_v2(self, pred, mask):
        ''' calc version from 
        https://github.com/vicmancr/mnms_example/blob/master/segmentation_model/cardiac-segmentation/metrics_acdc.py'''
        pred = np.atleast_1d(pred.astype(np.bool))
        mask = np.atleast_1d(mask.astype(np.bool))

        inter = np.count_nonzero(pred & mask)
        union = np.count_nonzero(pred) + np.count_nonzero(mask)
        try:
            t  = 2.* inter / float(union)
        except ZeroDivisionError:
            t = 1.0 if self.is_modified else 0.0
        return t
        
class VolDiceCalc(Metric):
    ''' class for the volume dice score calculation (pred, mask) '''
    def __init__(self, num_classes, mission='seg'):
        super().__init__(num_classes, mission, 'dice')

    def __call__(self, pred, mask, name=None):
        ''' default cl0 is background. '''

        if name is None:
            name = str(self.name_cache)
            self.name_cache += 1

        self.diceDict.update({name:{}})
        for cl in range(1, self.num_classes):
            p = (pred == (cl)).astype(np.float).reshape(-1)
            t = (mask == (cl)).astype(np.float).reshape(-1)
            dice = self._dice_calc_v2(p, t)
            self.diceDict[name][str(cl)] = dice

    def _dice_calc(self, pred, mask):
        inter = pred @ mask.T
        union = np.sum(pred) + np.sum(mask)
        t = (2. * inter.astype(np.float)) / (union.astype(np.float))
        return t

    def _dice_calc_v2(self, pred, mask):
        ''' calc version from 
        https://github.com/vicmancr/mnms_example/blob/master/segmentation_model/cardiac-segmentation/metrics_acdc.py'''
        pred = np.atleast_1d(pred.astype(np.bool))
        mask = np.atleast_1d(mask.astype(np.bool))

        inter = np.count_nonzero(pred & mask)
        union = np.count_nonzero(pred) + np.count_nonzero(mask)
        # t = 2.* inter / float(union)
        try:
            t = 2.* inter / float(union)
        except ZeroDivisionError:
            t = 1.0

        return t

class VolMIoUCalc(Metric):
    ''' class for the mask IoU calculation. ''' 
    def __init__(self, num_classes, mission):
        super().__init__(num_classes, mission, 'miou')

    def __call__(self, pred, mask, name=None):
        ''' default cl0 is background. '''
        if name is None:
            name = str(self.name_cache)
            self.name_cache += 1

        self.miouDict.update({name:{}})
        for _cls in range(1, self.num_classes):
            p = (pred == (_cls)).astype(np.float).reshape(-1)
            t = (mask == (_cls)).astype(np.float).reshape(-1)
            miou = self._mask_iou_v2(p, t)
            self.miouDict[name][str(_cls)] = miou

    def _mask_iou(self, mask1, mask2):
        ''' calculation in (class_map union instance_map) / area_instance) style '''
        intersection = mask1 @ mask2.T
        area1 = np.sum(mask1, axis=1).reshape(1, -1)
        area2 = np.sum(mask2, axis=1).reshape(1, -1)
        union = (area1.T + area2) - intersection
        ret = intersection.astype(np.float) / union.astype(np.float)
        return ret

    def _mask_iou_v2(self, mask1, mask2):
        mask1 = np.atleast_1d(mask1.astype(np.bool))
        mask2 = np.atleast_1d(mask2.astype(np.bool))
        
        intersection = np.count_nonzero(mask1 & mask2)
        union = np.count_nonzero(mask1 | mask2)
        
        ret = float(intersection) / float(union)
        return ret

class HanDistCalc(Metric):
    ''' class for the Hausdorff Distance.
    Computes the (symmetric) Hausdorff Distance (HD) between the binary objects in two
    images. It is defined as the maximum surface distance between the objects. '''

    def __init__(self, num_classes, mission, ismodified=True):
        super().__init__(num_classes, mission, 'dist')
        self.ismodified = ismodified

    def __call__(self, pred, mask, name=None):
        ''' default cl0 is background. '''

        if name is None:
            name = str(self.name_cache)
            self.name_cache += 1

        self.distDict.update({name:{}})
        for cl in range(1, self.num_classes):
            p = (pred == (cl)).astype(np.float).squeeze()
            t = (mask == (cl)).astype(np.float)
            dist = self._hd_calc(p, t)
            self.distDict[name][str(cl)] = dist

    def _hd_calc(self, pred, mask):
        hd_pred = self._surface_dist_calc(pred, mask).max()
        hd_mask = self._surface_dist_calc(mask, pred).max()
        hd = max(hd_pred, hd_mask)
        return hd

    def _hd95_calc(self, pred, mask):
        hd_pred = self._surface_dist_calc(pred, mask)
        hd_mask = self._surface_dist_calc(mask, pred)
        hd95 = np.percentile(np.hstack((hd_pred, hd_mask)), 95)
        return hd95
    
    def _surface_dist_calc(self, pred, mask):
        pred = np.atleast_1d(pred.astype(np.bool))
        mask = np.atleast_1d(mask.astype(np.bool))
                        
        # test for emptiness
        if 0 == np.count_nonzero(pred) or 0 == np.count_nonzero(mask): 
            return np.array([0.0]) if self.ismodified else np.array([10.0])
                
        # compute average surface distance        
        # Note: scipys distance transform is calculated only inside the borders of the
        #       foreground objects, therefore the input has to be reversed
        from scipy.ndimage.morphology import distance_transform_edt
        dt = distance_transform_edt(~mask)
        sd = dt[pred]
        return sd

class AssDistCalc(Metric):
    ''' class for the Hausdorff Distance.
    Computes the (symmetric) Hausdorff Distance (HD) between the binary objects in two
    images. It is defined as the maximum surface distance between the objects. '''

    def __init__(self, num_classes, mission, ismodified=True):
        super().__init__(num_classes, mission, 'assd')
        self.ismodified = ismodified

    def __call__(self, pred, mask, name=None):
        ''' default cl0 is background. '''

        if name is None:
            name = str(self.name_cache)
            self.name_cache += 1

        self.assdDict.update({name:{}})
        for cl in range(1, self.num_classes):
            p = (pred == (cl)).astype(np.float).squeeze()
            t = (mask == (cl)).astype(np.float)
            assd = self._assd_calc(p, t)
            self.assdDict[name][str(cl)] = assd

    def _assd_calc(self, pred, mask):
        return np.mean((self._asd_calc(pred, mask), self._asd_calc(mask, pred)))

    def _asd_calc(self, pred, mask):
        sd = self._surface_dist_calc(pred, mask)
        return sd.mean()

    def _surface_dist_calc(self, pred, mask):
        pred = np.atleast_1d(pred.astype(np.bool))
        mask = np.atleast_1d(mask.astype(np.bool))
                        
        # test for emptiness
        if 0 == np.count_nonzero(pred) or 0 == np.count_nonzero(mask): 
            return np.array([0.0]) if self.ismodified else np.array([10.0])
                
        # compute average surface distance        
        # Note: scipys distance transform is calculated only inside the borders of the
        #       foreground objects, therefore the input has to be reversed
        from scipy.ndimage.morphology import distance_transform_edt
        dt = distance_transform_edt(~mask)
        sd = dt[pred]
        return sd

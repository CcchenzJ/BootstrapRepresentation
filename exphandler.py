
import os
import json
import torch
import random
import datetime 
import setproctitle
import cv2 as cv2
import numpy as np
import pandas as pd

from config import cfg
from utils.tools import SavePath

_name = ''
_checkpoint = ''
_root_save_path = ''

def seed_everything(seed):
    # random.seed(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True

def set_root_save_path(exp_folder, ext_kwargs, is_global=True):
    """
    setting the root save path

    args:
        exp_folder: str, experiments folder to save. define in train.py.
        ext_kwargs: dict, some extension args to save in folder's name, {'seed': SEED}
    """
    global _root_save_path, _name

    ext_name = ''.join([f'_{k}{v}' for k,v in ext_kwargs.items()])
    _name = cfg.name + ext_name

    if not is_global:
        return f'./{exp_folder}/{cfg.name}/{ext_name}'
    
    setproctitle.setproctitle(_name) 
    _root_save_path = f'./{exp_folder}/{cfg.name}/{ext_name}'

def set_checkpoint(save_folder):
    """setting the checkpoint save path."""
    global _checkpoint
    
    _checkpoint = f'{_root_save_path}/{save_folder}'
    os.makedirs(_checkpoint, exist_ok=True)

def set_logger(Log, log_folder):
    """setting the logger save path and instantiate a log."""
    return Log(_name, f'{_root_save_path}/{log_folder}')
    
def set_model(model_zoo):
    """Instantiate a model according to config."""
    return getattr(model_zoo, cfg.model.type)(cfg.model, cfg.num_classes)

def set_criterion(loss_zoo):
    """Instantiate a loss according to config."""
    return getattr(loss_zoo, cfg.loss.type)(cfg.loss, cfg.num_classes)

def set_dataset(data, augmentation, phase):
    """Instantiate a dataset according to config."""
    check_transformer = lambda conf: None if conf.type is None \
                            else getattr(augmentation, conf.type)(conf)
    return getattr(data, cfg.dataset.type)(
                        conf=cfg.dataset, phase=phase,
                        transform=check_transformer(cfg.transformer))

def set_two_dataset(data, augmentation, phase):
    """Instantiate two datasets for labeled/unlabeled data according to config."""
    check_transformer = lambda conf: None if conf.type is None \
                            else getattr(augmentation, conf.type)(conf)
    return [getattr(data, cfg.two_dataset[key].type)(
                        conf=cfg.two_dataset[key], phase=phase,
                        transform=check_transformer(cfg.transformer[key]))
            for key in ('labeled', 'unlabeled')]

def set_optimizer(model):
    """Instantiate an optimizer according to config."""
    opt_params = [p for p in model.parameters() if p.requires_grad] if cfg.optimizer.tofreeze_module is None else \
        [p for n,p in model.named_parameters() if cfg.optimizer.tofreeze_module not in n]
    
    return cfg.optimizer.type(opt_params, **cfg.optimizer.args)
    
def set_save_path_func():
    return lambda epoch, iteration: SavePath(
                _name, epoch, iteration).get_path(_checkpoint)

def save_config_to_json(extra_configs=None):
    cfg_dict = cfg.to_dict()

    if extra_configs is not None:
        for _configs in extra_configs:
            cfg_dict[_configs] = extra_configs[_configs]

    pd.DataFrame({'cfg':cfg_dict}).to_json(f'{_root_save_path}/config.json', indent=2)

def get_resume_dir(resume):
    _options = {
        'interrupt': lambda ckpt, name: SavePath.get_interrupt(ckpt),
        'best': lambda ckpt, name: SavePath.get_best(ckpt),
        'latest': lambda ckpt, name: SavePath.get_latest(ckpt, name),
    }

    if resume not in _options.keys(): return None
    return _options[resume](_checkpoint, _name)    

def check_is_best(val_info, best_performance):
    if len(val_info) == 0: return best_performance, False

    judge  = lambda judge, condition: judge if condition else not judge
    check_update = lambda ref, src, k, is_ascent: \
        (ref[k]['all'], True) if judge(ref[k]['all'] >= src, is_ascent) else (src, False)
    best_performance, is_updated = check_update(val_info, best_performance, *cfg.best_metric)

    return best_performance, is_updated

def save_weights_to_ckpt(wtype, epoch, iteration, model, save_path):
    _operations = {
        'best': lambda ckpt: SavePath.remove_best(ckpt),
        'interrupt': lambda ckpt: SavePath.remove_interrupt(ckpt),
    }

    _operations[wtype](_checkpoint)
    model.save_weights(save_path(epoch, repr(iteration)+'_'+wtype))

def plot_curves(metrics, log_folder, loss_types, visualizer):
    visualizer.setup(f'{_root_save_path}{log_folder}{_name}.log')
    
    def _decode(metric):
        _cls_range = range(1, cfg.num_classes)
        datapoints = lambda metric: {cfg.dataset.class_names[_cls]: \
                                        f"x.data.{metric}['{cfg.dataset.class_names[_cls]}']" \
                                        for _cls in _cls_range}
        return datapoints

    if len(metrics) != 0:
        for metric in metrics:
            datapoints = _decode(metric)
            visualizer.plot(f'{_root_save_path}/{metric}_curve', 'valid', \
                            'x.data.epoch', datapoints(metric))

    visualizer.plot(f'{_root_save_path}/loss_curve', 'train', 'x.data.epoch', \
                        {type_: f"x.data.loss.{type_}" for type_ in loss_types}, 
                        smoothness=100)

    del visualizer

def makedirs_for_save_results(*save_folders):
    for save_folder in save_folders:
        os.makedirs(f'{_root_save_path}/{save_folder}', exist_ok=True)

def save_visual_results(image, save_folder, name):
    cv2.imwrite(f'{_root_save_path}/{save_folder}/{name}.png', image)

def save_metric_results(obj, save_folder):
    obj.save_to_json(f'{_root_save_path}/{save_folder}/{_name}')

def save_total_results(info):
    out = json.dumps(info) + '\n'
    with open(f'{_root_save_path}/total_results.txt', 'w') as f:
        f.write(out)

def save_postprocess_results(obj, save_folder):
    obj.save_to_folder(f'{_root_save_path}/{save_folder}')

def save_nii_results(pred, mask, affine, save_folder, name):
    import nibabel as nib
    pred_nii = nib.Nifti1Image(pred, affine)
    mask_nii = nib.Nifti1Image(mask, affine)
    nib.save(pred_nii, f'{_root_save_path}/{save_folder}/{name}_pred.nii.gz')
    nib.save(mask_nii, f'{_root_save_path}/{save_folder}/{name}_mask.nii.gz')

def save_overlap_results(img, pred, mask, save_folder, name, alpha=100):
    
    COLORDICT = {
        'mask': {k:v for k,v in zip(range(4), [164, 90,255,alpha])},
        'pred': {k:v for k,v in zip(range(4), [147,255,125,alpha])},
    }
    
    def _overlap(img, mask, pred, name):
        overlap = np.zeros((4,*cfg.img_size), dtype=np.uint8)
        for ch in range(4):
            overlap[ch][(mask==1)] = COLORDICT['mask'][ch]
            overlap[ch][(pred==1)] = COLORDICT['pred'][ch]
        overlap = overlap.transpose(1,2,0)
    
        from PIL import Image
        overlap = Image.fromarray(overlap)
        img = Image.fromarray(img[...,np.newaxis].repeat(3,-1))
        alpha = overlap.split()[-1]
        img.paste(overlap, mask=alpha)
        img.save(f'{_root_save_path}/{save_folder}/{name}.png')

    if len(img.shape) == 3:
        for s, (i, m, p) in enumerate(zip(img, mask, pred)):
            _overlap(i, m, p, f'{name}_s{s}')
    else:
        _overlap(img, mask, pred, name)
    
def resample(preds_numpy, gts):
    '''
    To reshape image into the target resolution and then compute the f1 score w.r.t ground truth mask
    input params:
        predicted_img_arr: predicted segmentation mask that is computed over the re-sampled and cropped input image
        gt_mask: ground truth mask in native image resolution
        pixel_size: native image resolution
    returns:
        predictions_mask: predictions mask in native resolution (re-sampled and cropped/zeros append as per size requirements)
        f1_val: f1 score over predicted segmentation masks vs ground truth
    '''
    from skimage import transform
    target_resolution = gts['target_resolution']
    pixel_size = gts['pixdim']
    gt_numpy = gts['data']

    new_x, new_y = cfg.img_size

    scale_vector = (pixel_size[0] / target_resolution[0], 
                    pixel_size[1] / target_resolution[1])

    mask_rescaled = transform.rescale(gt_numpy[0, ...], scale_vector, 
                                      order=0, preserve_range=True, mode='constant')

    x, y = mask_rescaled.shape
    x_s = (x - new_x) // 2
    y_s = (y - new_y) // 2
    x_c = (new_x - x) // 2
    y_c = (new_y - y) // 2

    total_slices = preds_numpy.shape[0]
    volume_resample = np.zeros((total_slices, gt_numpy.shape[1], gt_numpy.shape[2]))

    for slice_no in range(total_slices):
        # ASSEMBLE BACK THE SLICES
        slice_resample  = np.zeros((cfg.num_classes, x, y))
        predicted_slice = preds_numpy[slice_no,...]
        # insert cropped region into original image again
        if x > new_x and y > new_y:
            slice_resample[:, x_s:x_s+new_x, y_s:y_s+new_y] = predicted_slice
        else:
            if x <= new_x and y > new_y:
                slice_resample[:, :, y_s:y_s+new_y] = predicted_slice[:, x_c:x_c+x, :]
            elif x > new_x and y <= new_y:
                slice_resample[:, x_s:x_s+new_x, :] = predicted_slice[:, :, y_c:y_c+y]
            else:
                slice_resample[:, :, :] = predicted_slice[:, x_c:x_c+x, y_c:y_c+y]

        # RESCALING ON THE LOGITS
        prediction = transform.resize(slice_resample,
                                     (cfg.num_classes, gt_numpy.shape[1], gt_numpy.shape[2]),
                                     order=1,
                                     preserve_range=True,
                                     mode='constant')
        #print("b",prediction.shape)
        prediction_volume = np.uint16(np.argmax(prediction, axis=0))
        
        volume_resample[slice_no,...] = prediction_volume

    return volume_resample, gt_numpy

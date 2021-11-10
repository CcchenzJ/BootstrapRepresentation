
import os
import sys
import json
import random
import argparse
import setproctitle
from pathlib import Path
from collections import OrderedDict

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import measure

import data as _data
import model_zoo as _model_zoo
import augmentations as _augmentations
import exphandler as _exp_handler

from config import cfg, set_cfg, set_dataset
from config import COLORS, STD, MEANS

import utils.timer as _timer
import utils.tools as _tools
import utils.postprocess as _postprocess
import utils.metrics as _metrics

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Evaluation Script')
    parser.add_argument('--config',        default=None, help='')
    parser.add_argument('--trained_model', default=None, type=str, help='')
    parser.add_argument('--dataset',       default=None, type=str, help='')
    parser.add_argument('--seed',          default=111,  type=int, help='')
    parser.add_argument('--cuda',          default=True, type=str2bool, help='')

    parser.add_argument('--exp_folder',     default='/exps/',            type=str, help='')
    parser.add_argument('--save_folder',    default='/weights/',         type=str, help='')

    parser.add_argument('--metrics_folder', default='/results/metrics/', type=str, help='')
    parser.add_argument('--segmaps_folder', default='/results/segmaps/', type=str, help='')
    parser.add_argument('--postprocess_folder', default='/results/postprocess/', type=str, help='')

    parser.add_argument('--no_bar',  dest='no_bar',  action='store_true', help='')
    parser.add_argument('--no_sort', dest='no_sort', action='store_true', help='')
    parser.add_argument('--is_test', dest='is_test', action='store_true', help='')
    parser.add_argument('--display', dest='display', action='store_true', help='')
    parser.add_argument('--no_postprocess', dest='no_postprocess', action='store_true', help='')
    parser.set_defaults(no_bar=False, display=False, no_sort=False, no_postprocess=False, is_test=False)

    global args
    args = parser.parse_args(argv)
    
    if args.seed is not None:
        random.seed(args.seed)

def prep_metrics(metric_dict, postprocess_dict, predictions, gt, name=None):
    if len(metric_dict) == 0: return

    if predictions is not None and gt is not None:
        _timer.start(f'Post Process')
        for _, postprocess_obj in postprocess_dict.items():
            predictions = postprocess_obj.__call__(predictions, name)
        _timer.stop(f'Post Process')

    if predictions is not None and gt is not None:
        _timer.start(f'Prepare Metrics')
        for _, metric_obj in metric_dict.items():
            metric_obj.__call__(predictions, gt, name)
        _timer.stop(f'Prepare Metrics')

def evaluate(model, dataset, train_mode=False):

    frame_times  = _tools.MovingAverage()
    dataset_size = len(dataset)
    progress_bar = _tools.ProgressBar(30, dataset_size)

    print()

    metric_dict = {}
    for metric, obj_cfg in cfg.toevl_metrics.items():
        metric_dict.update({metric: getattr(_metrics, obj_cfg[0])(cfg.num_classes, **obj_cfg[1])})

    postprocess_dict = {}
    if not args.no_postprocess:
        for postprocess, obj_cfg in cfg.post_processes.items():
            postprocess_dict.update({postprocess: getattr(_postprocess, obj_cfg[0])(args.display, **obj_cfg[1])}) 
        
    dataset_indices = list(range(len(dataset)))

    if len(metric_dict) == 0:
        return {}

    try:
        for it, image_idx in enumerate(dataset_indices):
            _timer.reset()

            with _timer.env('Load Data'):
                img, gts, img_name = dataset.pull_named_item(image_idx)
                            
                batch = torch.autograd.Variable(img.unsqueeze(1))
                if args.cuda:
                    batch = batch.cuda()
                
            with _timer.env('Network Inference'):
                predictions = model(batch)

            with _timer.env('ReSample Prediction'):
                if not train_mode:
                    preds_numpy = predictions['seg'].cpu().numpy()
                else:
                    preds_numpy = predictions['seg'].detach().cpu().numpy()
                    
                if 'camus' in cfg.name:
                    preds_numpy = np.argmax(preds_numpy, 1)
                    gt_numpy = gts.cpu().numpy()
                else:
                    preds_numpy, gt_numpy = _exp_handler.resample(preds_numpy, gts)

            if not train_mode and args.display:
                _timer.start(f'Prepare Display')
                _exp_handler.save_nii_results(preds_numpy, gt_numpy, gts['affine'], args.segmaps_folder, img_name)
                # for i in range(preds_numpy.shape[0]):
                    # _exp_handler.save_visual_results(preds_numpy[i]*255//3, args.segmaps_folder, img_name+f'_f{i}')
                # _exp_handler.save_overlap_results(img.numpy(), preds_numpy, gt_numpy, args.segmaps_folder, img_name)
                _timer.stop(f'Prepare Display')

            with _timer.env('Prepare Metrics'):
                prep_metrics(metric_dict, postprocess_dict, preds_numpy, gt_numpy, img_name)

            if it > 1:
                frame_times.add(_timer.total_time())

            if not args.no_bar:
                fps = 1 / frame_times.get_avg() if it > 1 else 0
                progress = (it + 1) / dataset_size * 100
                progress_bar.set_val(it+1)
                
                print('\rProcessing Images  %s %6d / %6d (%5.2f%%)  %5.2f fps   ' \
                    % (repr(progress_bar), it+1, dataset_size, progress, fps), end='')
        
        print()
        val_info = calc_metric(metric_dict)
        if not train_mode:
            print('Saving data ... ', end='', flush=True)
            for metric in metric_dict:
                _exp_handler.save_metric_results(metric_dict[metric], args.metrics_folder)
            for postprocess in postprocess_dict:
                _exp_handler.save_postprocess_results(postprocess_dict[postprocess], args.postprocess_folder)

            _exp_handler.save_total_results(val_info)
            print('done.\n')
        return val_info 

    except KeyboardInterrupt:
        print('Stopping ... ')

def calc_metric(metric_dict):
    print('Calculating metrics ... ')

    all_metric = {}
    for metric, metric_obj in metric_dict.items():
        all_metric.update({metric: OrderedDict()})
        all_metric[metric]['all'] = 0
        resDict = metric_obj.gather(metric)

        for _cls in range(1, metric_obj.num_classes):
            mean = np.mean(np.array(resDict[_cls]))
            all_metric[metric][cfg.dataset.class_names[_cls]] = mean
        all_metric[metric]['all'] = sum(all_metric[metric].values()) / (len(all_metric[metric].values())-1)

    print_metrics(all_metric)

    all_metric = {k: {j:round(u, 2) for j, u in v.items()} for k, v in all_metric.items()}
    return all_metric 

def print_metrics(all_metric):
    default_metric = list(all_metric.keys())[0]

    make_row = lambda vals: (' %5s |' * len(vals)) % tuple(vals)
    make_sep = lambda n: ('-------+' * n)

    print()
    print(make_row([''] + [(' '+ x + ' ') for x in all_metric[default_metric].keys()]))
    print(make_sep(len(all_metric[default_metric])+1))
    for metric in all_metric.keys():
        print(make_row([metric] + ['%.2f' % x if x<100 else '%.1f' % x for x in all_metric[metric].values()]))
    print(make_sep(len(all_metric[default_metric])+1))
    print()

if __name__ == '__main__':
    parse_args()

    if args.config is not None:
        set_cfg(args.config)

    if args.config is None:
        print('Config not specified. Parsed %s from the file name.\n' % args.config)
        set_cfg(args.config)
    
    _exp_handler.set_root_save_path(args.exp_folder, {'seed':args.seed})
    _exp_handler.set_checkpoint(args.save_folder)

    args.trained_model = _exp_handler.get_resume_dir(args.trained_model)
    print('From :', args.trained_model)

    if args.dataset is not None:
        set_dataset(args.dataset)

    with torch.no_grad():
        _exp_handler.makedirs_for_save_results(args.metrics_folder, 
                                               args.segmaps_folder, 
                                               args.postprocess_folder)
        if args.cuda:
            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        phase = 'test' if args.is_test else 'valid'
        if cfg.scheme in ('proposed', 'meanteacher'):
            dataset = _exp_handler.set_two_dataset(_data, _augmentations, phase)[0]
        else:
            dataset = _exp_handler.set_dataset(_data, _augmentations, phase)

        print('Loading model ... ', end='')
        model = _exp_handler.set_model(_model_zoo)
        model.load_weights(args.trained_model)
        model.eval()
        print('Done.')

        if args.cuda:
            model = model.cuda()

        evaluate(model, dataset)












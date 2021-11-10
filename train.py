
import os
import sys
import math
import time
import datetime
import json
import argparse
import setproctitle

import torch
import torch.nn as nn
import numpy as np

import data as _data
import loss_zoo as _loss_zoo
import model_zoo as _model_zoo
import exphandler as _exp_handler
import augmentations as _augmentations

from config import cfg, set_cfg, set_dataset, set_max_iter

import utils.timer as _timer
import utils.tools as _tools
import utils.logger as _logger

import eval as eval_script

# os.environ['CUDA_LAUNCH_BLOCKING']='1'

def set_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    
    global cur_lr
    cur_lr = new_lr

parser = argparse.ArgumentParser(description='Training Script.')
parser.add_argument('--batch_size',  default=8,    type=int,      help='')
parser.add_argument('--start_iter',  default=-1,   type=int,      help='')
parser.add_argument('--max_iter',    default=-1,   type=int,      help='')
parser.add_argument('--seed',        default=111,  type=int,      help='')
parser.add_argument('--resume',      default=None, type=str,      help='')

parser.add_argument('--lr',       default=None, type=float, help='')
parser.add_argument('--gamma',    default=None, type=float, help='')

parser.add_argument('--save_folder', default='/weights/', type=str, help='')
parser.add_argument('--exp_folder',  default='/exps/',    type=str, help='')
parser.add_argument('--log_folder',  default='/logs/',    type=str, help='')

parser.add_argument('--config',  default=None, help='')
parser.add_argument('--dataset', default=None, help='')

parser.add_argument('--save_interval',    default=5000, type=int, help='')
parser.add_argument('--validation_epoch', default=2,   type=int, help='')

parser.add_argument('--keep_latest',  dest='keep_latest', action='store_true',  help='')
parser.add_argument('--no_log',       dest='log',         action='store_false', help='')
parser.add_argument('--no_interrupt', dest='interrupt',   action='store_false', help='')
parser.set_defaults(keep_latest=False, log=True, interrupt=True)

args = parser.parse_args()

if args.config is not None:
    set_cfg(args.config)

if args.dataset is not None:
    set_dataset(args.dataset)

if args.max_iter != -1:
    set_max_iter(args.max_iter)

if torch.cuda.device_count == 0:
    print('No GPUs detected. Exiting ...')
    sys.exit()

if getattr(args, 'lr') == None: 
    setattr(args, 'lr', cfg.optimizer.args['lr'])
if getattr(args, 'gamma') == None: 
    setattr(args, 'gamma', getattr(cfg, 'gamma'))

cur_lr = args.lr

loss_types = cfg.loss.labels

torch.set_default_tensor_type('torch.cuda.FloatTensor')
_exp_handler.seed_everything(args.seed)
setattr(_data, 'SEED', args.seed)

class NetLoss(nn.Module):
    """ Model Traning Wrapper """
    def __init__(self, model, criterion):
        super().__init__()
        self.model = model
        self.criterion = criterion
    
        self.forward_dict = {
            'normal': self.normal_forward,
            'proposed': self.proposed_forward,
            'pretrain_CGL': self.pretrain_CGL_forward,
            'pretrain_contrast': self.pretrain_contrast_forward,
        }

    def forward(self, *args, **kwargs):
        return self.forward_dict[cfg.scheme](*args, **kwargs)

    def normal_forward(self, inputs, gt):
        inputs = torch.autograd.Variable(inputs.cuda())
        gt = torch.autograd.Variable(gt.cuda())

        if cfg.mixup_alpha > 0:
            inputs, gt = self.mixup(inputs, gt, cfg.mixup_alpha)

        losses = self.criterion(self.model(inputs), gt)
        return losses

    def pretrain_contrast_forward(self, inputs1, inputs2):
        inputs1 = torch.autograd.Variable(inputs1.cuda())
        inputs2 = torch.autograd.Variable(inputs2.cuda())

        inputs = [inputs1, inputs2]
        losses = self.criterion(self.model(inputs))
        return losses

    def pretrain_CGL_forward(self, inputs0, inputs1, inputs2):
        inputs0 = torch.autograd.Variable(inputs0.flatten(0,1).cuda())
        inputs1 = torch.autograd.Variable(inputs1.flatten(0,1).cuda())
        inputs2 = torch.autograd.Variable(inputs2.flatten(0,1).cuda())

        inputs = torch.cat([inputs0, inputs1, inputs2])
        losses = self.criterion(self.model(inputs))
        return losses

    def proposed_forward(self, labeled_inputs, gt, unlabeled_inputs, dist=None):
        labeled_inputs = torch.autograd.Variable(labeled_inputs.cuda())
        gt = torch.autograd.Variable(gt.cuda())

        if cfg.mixup_alpha > 0:
            labeled_inputs, gt = self.mixup(labeled_inputs, gt, cfg.mixup_alpha)

        unlabeled_inputs = torch.autograd.Variable(unlabeled_inputs.cuda())
        if dist is not None:
            dist = torch.autograd.Variable(dist.cuda())

        inputs = torch.cat([labeled_inputs, unlabeled_inputs], 0)
        losses = self.criterion(self.model(inputs), gt, dist)
        return losses

    def mixup(self, x, y, alpha=0.1):
        if not np.random.randint(2):
            return x, (y, y, 1., y)

        lam = np.random.beta(alpha, alpha)
        index = torch.randperm(x.size(0)).cuda()

        mixed_x  = lam * x + (1. - lam) * x[index]
        mixed_y  = lam * y + (1. - lam) * y[index]
        y_a, y_b = y, y[index]

        return mixed_x, (y_a, y_b, lam, mixed_y)

def train():

    _exp_handler.set_root_save_path(args.exp_folder, {'seed':args.seed})
    _exp_handler.set_checkpoint(args.save_folder)
    _exp_handler.save_config_to_json({'train_script_args':dict(args._get_kwargs())})

    if cfg.scheme in ('proposed', 'meanteacher'):
        dataset, unlabeled_dataset = _exp_handler.set_two_dataset(_data, _augmentations, 'train')
        unlabeled_dataset.flow_mask = dataset.flow_mask
    else:
        dataset = _exp_handler.set_dataset(_data, _augmentations, 'train')

    if args.validation_epoch > 0:
        setup_eval()
        if cfg.scheme in ('proposed', 'meanteacher'):
            val_dataset = _exp_handler.set_two_dataset(_data, _augmentations, 'valid')[0]
        else:
            val_dataset = _exp_handler.set_dataset(_data, _augmentations, 'valid')

    model = _exp_handler.set_model(_model_zoo)
    criterion = _exp_handler.set_criterion(_loss_zoo)
    model.train()

    if args.log:
        log = _exp_handler.set_logger(_logger.Log, args.log_folder)

    _timer.disable_all()
    args.resume = _exp_handler.get_resume_dir(args.resume)

    if args.resume is not None:
        print('Resuming training, loading {} ...'.format(args.resume))
        model.load_weights(args.resume)
        if args.start_iter == -1:
            args.start_iter = _tools.SavePath.from_str(args.resume).iteration
        
    else:
        print('Initializing weights ...')
        model.init_weights(args.seed)

    optimizer = _exp_handler.set_optimizer(model)

    net = NetLoss(model, criterion).cuda()
    
    if not cfg.model.freeze_bn: model.freeze_bn()
    model(torch.zeros(1, cfg.model.in_channels, *cfg.img_size).cuda())
    if not cfg.model.freeze_bn: model.freeze_bn(True)

    iteration = max(args.start_iter, 0)
    last_time = time.time()

    epoch_size = len(dataset) // args.batch_size
    num_epochs = math.ceil(cfg.max_iter / epoch_size)

    step_index = 0
    
    data_loader = torch.utils.data.DataLoader(
                            dataset, args.batch_size,
                            num_workers=4, shuffle=True, pin_memory=True)

    if cfg.scheme in ('proposed', 'meanteacher'):
        unlabeled_indexes = list(range(len(unlabeled_dataset)))

    save_path = _exp_handler.set_save_path_func()
    time_avgs = _tools.MovingAverage()

    global loss_types
    loss_avgs = {k: _tools.MovingAverage(100) for k in loss_types}
    best_performance = -1

    print('Begin training! --', cfg.name)
    print()

    try:
        for epoch in range(num_epochs):

            if (epoch+1) * epoch_size < iteration:
                continue

            if cfg.lr_schedule == 'cos':
                set_lr(optimizer, args.lr * 0.5 * (1. + math.cos(math.pi * epoch / num_epochs)))

            for it, datum in enumerate(data_loader):
                if 'proposed' in cfg.scheme:
                    unlabeled_index = int(iteration % (len(unlabeled_dataset) // cfg.num_subjects))
                    unlabeled_epoch = int(iteration / (len(unlabeled_dataset) // cfg.num_subjects))
                    if unlabeled_index == 0:
                        np.random.shuffle(unlabeled_indexes)

                    for k in range(cfg.num_subjects):
                        unlabeled_dataset.pull_train_item(unlabeled_indexes[unlabeled_index+k], k)
                    unlabeled_datum = unlabeled_dataset.get_train_items()
                    datum = [*datum, *unlabeled_datum]

                if iteration == (epoch+1) * epoch_size:
                    break
            
                if iteration == args.max_iter:
                    break

                if hasattr(cfg, 'loss_warmup'):
                    if iteration == 0:
                        for key in cfg.loss_warmup['keys']:
                            setattr(cfg.loss, f'use_{key}_loss', False)
                    if iteration == cfg.loss_warmup['until']:
                        for key in cfg.loss_warmup['keys']:
                            setattr(cfg.loss, f'use_{key}_loss', True) 

                if cfg.lr_schedule == 'step':
                    if cfg.lr_warmup_until > 0 and iteration <= cfg.lr_warmup_until:
                        set_lr(optimizer, 
                                (args.lr - cfg.lr_warmup_until) \
                            * (iteration / cfg.lr_warmup_until) \
                            + cfg.lr_warmup_until)

                    while step_index < len(cfg.lr_steps) and iteration >= cfg.lr_steps[step_index]:
                        step_index += 1
                        set_lr(optimizer, args.lr * (args.gamma**step_index))

                optimizer.zero_grad()

                losses = net(*datum)

                losses = {k: v for k,v in losses.items()}
                loss = sum([losses[k] for k in losses])
                
                loss.backward()

                optimizer.step()

                for k in losses:
                    loss_avgs[k].add(losses[k].item())

                cur_time = time.time()
                elapsed = cur_time - last_time
                last_time = cur_time

                if iteration != args.start_iter:
                    time_avgs.add(elapsed)

                if iteration % 10 == 0:
                    eta_str = str(datetime.timedelta(seconds=(cfg.max_iter-iteration) \
                                                            * time_avgs.get_avg())).split('.')[0]
                    
                    total = sum([loss_avgs[k].get_avg() for k in losses])
                    loss_labels = sum([[k, loss_avgs[k].get_avg()] for k in loss_types if k in losses], [])
                    
                    if cfg.scheme in ('proposed',):
                        print(('[%3d|%3d] %7d ||' + (' %s: %.3f |' * len(losses)) + ' T: %.3f || ETA: %s || timer: %.3f')
                                % tuple([epoch, unlabeled_epoch, iteration] + loss_labels + [total, eta_str, elapsed]), flush=True)
                    else:
                        print(('[%3d] %7d ||' + (' %s: %.3f |' * len(losses)) + ' T: %.3f || ETA: %s || timer: %.3f')
                                % tuple([epoch, iteration] + loss_labels + [total, eta_str, elapsed]), flush=True)

                if args.log:
                    precision = 3
                    loss_info = {k: round(float(losses[k]), precision) for k in losses}
                    loss_info['T'] = round(total, precision)
                        
                    log.log('train', loss=loss_info, 
                                epoch=epoch, iter=iteration,
                                lr=round(cur_lr, 10), elapsed=elapsed)  

                iteration += 1

                if iteration % args.save_interval == 0 and iteration != args.start_iter:
                    if args.keep_latest:
                        latest = _exp_handler.get_resume_dir('latest')

                    print('Saving state, iter:', iteration)
                    model.save_weights(save_path(epoch, iteration))

                    if args.keep_latest and latest is not None:
                        print('Deleting old save ...')
                        os.remove(latest)           

            if args.validation_epoch > 0:
                if epoch % args.validation_epoch == 0 and epoch > 0:
                    val_info = compute_validation_metric(epoch, iteration, model, val_dataset, log)
                    
                    best_performance, is_updated = _exp_handler.check_is_best(val_info, best_performance)
                    if is_updated:
                        print('Best results. Saving network ...')
                        _exp_handler.save_weights_to_ckpt('best', epoch, iteration, model, save_path)
                    
                    _exp_handler.plot_curves(cfg.tovis_metrics, args.log_folder, loss_types, _logger.LogVisualizer())
                    print('Best %s is: %s' % (cfg.best_metric[0], best_performance))

        val_info = compute_validation_metric(epoch, iteration, model, val_dataset, log)
        
        best_performance, is_updated = _exp_handler.check_is_best(val_info, best_performance)
        if is_updated:
            print('Best results. Saving network ...')
            _exp_handler.save_weights_to_ckpt('best', epoch, iteration, model, save_path)

        print('Best %s is: %s' % (cfg.best_metric[0], best_performance))

    except KeyboardInterrupt:
        if args.interrupt:
            print('Stopping early. Saving network ...')
            _exp_handler.save_weights_to_ckpt('interrupt', epoch, iteration, model, save_path)
        sys.exit()

    model.save_weights(save_path(epoch, iteration))

def compute_validation_metric(epoch, iteration, model, dataset, log=None):    
    with torch.no_grad():
        model.eval()

        start = time.time()
        print()
        print('Computing validation metric ...', flush=True)
        val_info = eval_script.evaluate(model, dataset, train_mode=True)
        end = time.time()

        if log is not None:
            log.log('valid', val_info, 
                    elapsed=(end-start), epoch=epoch, iter=iteration)

        model.train()
    return val_info

def set_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    
    global cur_lr
    cur_lr = new_lr

def setup_eval():
    eval_script.parse_args(['--seed', str(args.seed), 
                            '--no_bar', 
                            '--no_sort', 
                            '--no_postprocess'])

if __name__ == '__main__':
    train()

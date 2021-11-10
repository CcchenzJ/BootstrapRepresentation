
import os
import glob
import json
import argparse

import numpy as np

import config

class Evaluator(object):
    ''' Evaluation class for setting a evaluation file. '''
    def __init__(self, cfg, root):
        self.name = getattr(config, cfg+'_config').name
        self.root = root
        self.cache = {}
        self.seed_cache = {}

    def eval(self, metric, seeds):
        print('Eval Seeds: ', seeds)
        self.cache[metric] = []
        self.seed_cache[metric] = {}
        for seed in seeds:
            self.cache[metric] += [self._gather(metric, seed)]
            self.seed_cache[metric][seed] = np.mean(np.sum(self.cache[metric][-1], axis=1) \
                                                / self.cache[metric][-1].shape[1])
        self.cache[metric] = np.concatenate(self.cache[metric])

    def _gather(self, metric, seed):
        path = f'{self.root}/{self.name}/_seed{seed}/results/metrics/'
        file = glob.glob(path+f'*_{metric}.json')[-1]
        with open(file, 'r') as f:
            metric_dict = json.load(f)[self.name+'_seed'+seed]

        all_metric = []
        for _, data in metric_dict.items():
            all_metric += [[data[idx] for idx in data]]
        all_metric = np.stack(all_metric)

        return all_metric

    def print(self, max_len=10):
        fmt_str = self.name + ' in ' + self.root + '\n'
        for k, v in self.cache.items():
            fmt_str += f'metric- {k}:\n'
            fmt_str += 'Seed:'
            for s, vv in self.seed_cache[k].items():
                fmt_str += f'{s:4}: {vv:.4f} | '
            fmt_str += '\n'
            for idx in range(v.shape[1]):
                fmt_str += f'    {np.mean(v[:,idx]):.4f} +- {np.std(v[:,idx]):.4f}\n'
            all_v = np.mean(np.sum(v, axis=1) / v.shape[1])
            std_v = np.std(np.array(list(self.seed_cache[k].values())))
            fmt_str += f'all: {all_v:.4f} +- {std_v:.4f}\n'
            # fmt_str += f'{np.mean(v):.4f}'
        print(fmt_str)

def str2list(v):
    return v.rsplit(':')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluation Script.')
    parser.add_argument('--config', default=None, type=str, help='')
    parser.add_argument('--root', default='exps', type=str, help='')
    parser.add_argument('--metric', default='dice:assd', type=str2list, help='')
    parser.add_argument('--seeds', default='111:333:555:777:999', type=str2list, help='')
    args  = parser.parse_args()

    evaluator = Evaluator(args.config, args.root)

    for metric in args.metric:
        evaluator.eval(metric, args.seeds)

    evaluator.print()
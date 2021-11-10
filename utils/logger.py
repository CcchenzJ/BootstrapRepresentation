
import os
import sys
import json
import time
import datetime

from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

import utils.tools as _tools
# import tools as _tools

class Log:
    """Adapted from https://github.com/dbolya/yolact/blob/master/utils/logger.py. 
    A class to log information during training per information and save it out.
    It also can include extra debug information like GPU usage / temp automatically.
    """

    def __init__(self, log_name, log_dir, session_data={}):
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, f'{log_name}.log')
        
        self._log_header(session_data)
    
    def _log_header(self, session_data={}):

        info = {}
        info['type'] = 'header'
        info['data'] = session_data
        info['time'] = time.time()

        out = json.dumps(info) + '\n'
        
        with open(self.log_path, 'w') as f:
            f.write(out)
        
    def log(self, type, data={}, **kwargs):

        info = {}
        info['type'] = type
        
        kwargs.update(data)
        info['data'] = kwargs
        info['time'] = time.time()
    
        out = json.dumps(info) + '\n'

        with open(self.log_path, 'a') as f:
            f.write(out)

class LogEntry:
    """ A class that allows you to navigate a dictonary using x.a.b[2].c, etc. """
    def __init__(self, entry):
        self._ = entry

    def __getattr__(self, name):
        if name == '_':
            return self.__dict__['_']

        res = self.__dict__['_'][name]

        if type(res) == dict or type(res) == list:
            return LogEntry(res)
        else:
            return res
    
    def __getitem__(self, name):
        return self.__getattr__(name)

    def __len__(self):
        return len(self.__dict__['_'])

class LogVisualizer:
    def __init__(self):
        self.COLORS = [
            'xkcd:azure', 'xkcd:coral', 'xkcd:turquoise', 'xkcd:orchid', 'xkcd:orange',
            'xkcd:blue','xkcd:red','xkcd:teal','xkcd:magenta','xkcd:orangered' 
            ]

    def _decode(self, query):
        path, select = (query.split(';') + [''])[:2]
        
        if select.strip() == '':
            select = lambda x: True
        else:
            select = eval('lambda x: ' + select)

        if path.strip() == '':
            path = lambda x: x
        else:
            path = eval('lambda x: ' + path)
        
        return path, select

    def _follow(self, entry, query):
        path, select = query

        try:
            if select(entry):
                res = path(entry)

                if type(res) == LogEntry:
                    return res.__dict__['_']
                else:
                    return res
            else:
                return None
        except (KeyError, IndexError):
            return None

    def _color(self, idx:int):
        return self.COLORS[idx % len(self.COLORS)]

    def setup(self, log_path):
        self.log = defaultdict(lambda: [])

        if not os.path.exists(log_path):
            print(f'{log_path} doesn\'t exist!')
            return
        
        with open(log_path, 'r') as f:
            for line in f:
                line =line.strip()
                if not len(line) > 0:
                    continue

                js = json.loads(line)

                _type = js['type']
                ljs = LogEntry(js)

                self.log[_type].append(ljs)

    def plot(self, path, entry_type, x, y_dict, smoothness=0):
        log = self.log[entry_type]
        query_x = self._decode(x)

        plt.figure()
        for idx, (label, y) in enumerate(y_dict.items()):
            query_y = self._decode(y)

            if smoothness > 1:
                avg = _tools.MovingAverage()

            _x, _y = [], []
            for datum in log:
                val_x = self._follow(datum, query_x)
                val_y = self._follow(datum, query_y)

                if val_x is not None and val_y is not None:
                    if smoothness > 1:
                        avg.append(val_y)
                        val_y = avg.get_avg()
                    
                        if len(avg) < smoothness // 10:
                            continue
                    _x.append(val_x)
                    _y.append(val_y)
            
            plt.plot(_x, _y, color=self._color(idx), label=label)
        
        plt.title(os.path.basename(path))
        plt.legend()
        plt.grid(linestyle=':', linewidth=0.5)
        # plt.show()
        plt.savefig(path+'_plot.png')
        plt.close()

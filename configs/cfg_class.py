
import torch

class Config:
    ''' Config class for setting a config file. '''
    def __init__(self, config_dict):
        for key, val in config_dict.items():
            self.__setattr__(key, val)

    def copy(self, new_config_dict={}):
        ret = Config(vars(self))
        for key, val in new_config_dict.items():
            ret.__setattr__(key, val)
        return ret

    def replace(self, new_config_dict):
        if isinstance(new_config_dict, Config):
            new_config_dict = vars(new_config_dict)

        for key, val in new_config_dict.items():
            self.__setattr__(key, val)

    def print(self):
        for k, v in vars(self).items():
            print(k, ' = ', v if not isinstance(v, Config) else v.print())

    def to_dict(self):
        config_dict = {}
        for key, val in vars(self).items():
            config_dict[key] = str(val) if not isinstance(val, Config) \
                                        else val.to_dict()
        return config_dict
        

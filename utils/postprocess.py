''' PostProcess objects. '''

import numpy as np
import math
import cv2
import os
from skimage import measure, transform
from abc import abstractclassmethod

# ----------------------------- Template Class ------------------------------------ #

class PostProcess:

    def __init__(self, display, pptype):
        self.display = display
        self.pptype  = pptype
        self.name_cache = 0
        self.__setattr__(pptype+'Dict', {})

    @abstractclassmethod
    def __call__(self, preds):
        pass

    def ResDict(self):
        return self.__getattribute__(self.pptype+'Dict')
    
    @abstractclassmethod
    def save_to_folder(self, folder_path=''):
        for name, imgs in self.ResDict().items():
            cv2.imwrite(folder_path+f'/{self.pptype}_{name}_src.png', imgs['src'])
            cv2.imwrite(folder_path+f'/{self.pptype}_{name}_dst.png', imgs['dst'])

# ----------------------------------------------------------------------------------------------

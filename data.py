import os
import cv2
import sys
import glob
import h5py
import json

import torch
import random
import numpy as np
import nibabel as nib

from abc import abstractclassmethod

from config import cfg

SEED = 111

class DataSetTemplate(torch.utils.data.Dataset):
    """ Dataset Template to build the dataset according to the config. 
        :param conf: config.
        :param phase: [train, valid, test].
        :transform: transform object for training data.
    """
    def __init__(self, conf, phase, transform=None):
        self.conf = conf
        self.phase = phase

        self.transform = transform

        # Initialize data according to the phase.
        self.data = self._init_data(phase)

    @abstractclassmethod
    def _init_data(self, phase):
        return []
    
    @abstractclassmethod
    def pull_item(self, i):
        return None, None
    
    def pull_named_item(self, i):
        data, gt = self.pull_item(i)
        return data, gt, self.data_name[i]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.pull_item(i)

    def __repr__(self):
        _conf   = getattr(self.conf, self.phase)
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    cfg.dataset.name: {}\n'.format(self.conf.name)
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: \n' 
        fmt_str += '       data: {}\n'.format(_conf.data)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

class VolumetricSet(DataSetTemplate):
    """ Dataset for the volumetric data from ACDC & Prostate. """
    def __init__(self, conf, phase, transform=None):
        super().__init__(conf, phase, transform)
        self.sample_cache = []
        self.select_cache = []
        
    def _init_data(self, phase):
        conf = getattr(self.conf, phase)
        filelist = glob.glob(conf.data)

        # select the labeled data according to CGL git repo.
        self.labeled_id_list = ["001","002","003","004","005","006","017","018","019","020","012",\
                                "021","022","023","024","025","026","037","038","039","040", \
                                "041","042","043","044","045","046","057","058","059","060",\
                                "061","062","063","064","065","066","077","078","079","080","072",\
                                "081","082","083","084","085","086","097","098","099","100"]

        self.data_name, data = [], []
        for file in filelist:
            if os.path.basename(file).split('_gt')[0][-3:] not in self.labeled_id_list \
                and 'ACDC' in self.conf.name and phase == 'train':
                continue
            data.append((file.replace('_gt', ''), file))
            self.data_name.append(os.path.basename(file).split('_gt')[0])
        return data

    def pull_item(self, i):
        if self.phase == 'train':
            return self.pull_train_item(i)
        elif self.phase in ('valid', 'test'):
            return self.pull_valid_item(i)
    
    def pull_train_item(self, i, cnt=0):
        """ Pull the training item to cache
            :param cnt: volume count. """
        data = np.transpose(nib.load(self.data[i][0]).get_fdata(), [2,0,1])

        # select N(num_samples) slices within the max distance(max_dist).
        if cnt == 0:
            select = [0, 100]
            while max(select) - min(select) + 1 > cfg.max_dist:
                select = random.sample(range(data.shape[0]), cfg.num_samples)
            self.min_select = max(max(select) - cfg.max_dist,  0)
            self.max_select = self.min_select + cfg.max_dist
        else:
            while data.shape[0] - self.min_select < cfg.num_samples:
                idx = random.sample(range(len(self.data)), 1)[0]
                data = np.transpose(nib.load(self.data[idx][0]).get_fdata(), [2,0,1])
            select = random.sample(range(self.min_select, min(self.max_select, data.shape[0])), cfg.num_samples)

        data = data.astype(np.float32)
        sample = data[select]
        self.sample_cache += [sample]
        self.select_cache += [select]

    def get_train_items(self):
        sample = np.concatenate(self.sample_cache, axis=0)
        select = np.concatenate(self.select_cache, axis=0)

        if self.transform is not None:
            sample = self.transform(sample, None, self.flow_mask)[0]

        self.sample_cache = []
        self.select_cache = []

        return torch.FloatTensor(sample.copy()).unsqueeze(1), \
               torch.LongTensor(select)

    def pull_valid_item(self, i):
        data = np.transpose(nib.load(self.data[i][0]).get_fdata(), [2,0,1])
        gt = np.transpose(nib.load(self.data[i][1]).get_fdata(), [2,0,1])

        data = data.astype(np.float32)
        gt = data.astype(np.float32)
                
        return torch.FloatTensor(data), gt
 
class ImageSet(DataSetTemplate):
    def __init__(self, conf, phase, transform=None):
        super().__init__(conf, phase, transform)

    def _init_data(self, phase):
        conf = getattr(self.conf, phase)
        filelist = glob.glob(conf.data)

        patient_set = set([os.path.basename(file).split('_')[0] for file in filelist])
        # randomly select patients to construct the labeled dataset(D_L)
        # and save to the file($DATASET_$NUM_patient_set.txt).
        if self.phase == 'train' and hasattr(self.conf, 'num_patients'):
            if not self.conf.has_patient_set: 
                patient_set = random.sample(patient_set, self.conf.num_patients)
                os.makedirs(f'dataset/patient_set/{self.conf.name}', exist_ok=True)
                with open(f'dataset/patient_set/{self.conf.name}/'
                    + f'{self.conf.name}_{self.conf.num_patients}p_{SEED}_patient_set.txt', 'w') as f:
                    for patient in patient_set:
                        f.write(patient+'\n')
            else:
                with open(f'dataset/patient_set/{self.conf.name}/'
                    + f'{self.conf.name}_{self.conf.num_patients}p_{SEED}_patient_set.txt', 'r') as f:
                    patient_set = set([line.replace('\n','') for line in f])

        self.data_name, data = [], []
        for file in filelist:
            if phase == 'train':
                if os.path.basename(file).split('_')[0] not in patient_set:
                    continue
                n_slice = nib.load(file).get_fdata().shape[-1]
                for sdx in range(n_slice):
                    data.append((file.replace('_gt', ''), file, sdx))
                    self.data_name.append(os.path.basename(file).split('_gt')[0]+f'_s{sdx}')
            else:
                data.append((file.replace('_gt', ''), file))
                self.data_name.append(os.path.basename(file).split('_gt')[0])
        
        if phase == 'train': self._get_occurence_map(data)
        return data

    def _get_occurence_map(self, data):
        load_gt = lambda d,i: nib.load(d[i][1]).get_fdata()[...,d[i][-1]]

        gt_mask = np.zeros((192,192))
        for i in range(len(data)):
            gt_mask += np.greater(load_gt(data, i), 0)
       
        self.gt_mask = gt_mask / len(data)
        self.flow_mask = gaussian(1. - self.gt_mask)

    def pull_item(self, i):
        if self.phase == 'train':
            return self.pull_train_item(i)
        elif self.phase in ('valid', 'test'):
            return self.pull_valid_item(i)

    def pull_train_item(self, i):
        data = nib.load(self.data[i][0]).get_fdata()[..., self.data[i][-1]]
        gt = nib.load(self.data[i][1]).get_fdata()[..., self.data[i][-1]]

        data = data.astype(np.float32)
        gt = gt.astype(np.float32)
        
        if self.transform is not None:
            data, gt = self.transform(data, gt, self.flow_mask)

        data = torch.FloatTensor(data).unsqueeze(0)
        gt = torch.FloatTensor(gt)
        return data, gt 

    def pull_valid_item(self, i):
        data = np.transpose(nib.load(self.data[i][0]).get_fdata(), [2,0,1])
        
        gt_load = nib.load(self.data[i][1])
        gt = np.transpose(gt_load.get_fdata(), [2,0,1])
        pixel_size = gt_load.header['pixdim'][1:4]
        affine_ret = gt_load.affine

        data = data.astype(np.float32)
        
        return torch.FloatTensor(data), \
               {'data': gt, 'affine': affine_ret, 'pixdim': pixel_size, 
                'target_resolution': self.conf.target_resolution }

class MnMImageSet(ImageSet):
    def __init__(self, conf, phase, transform=None):
        super().__init__(conf, phase, transform)

    def _init_data(self, phase):
        conf = getattr(self.conf, phase)
        filelist = sum([glob.glob(f'{conf.data}/{vendor}/*_ED_gt.nii.gz') for vendor in self.conf.vendors[phase]], [])

        # randomly select patients to construct the labeled dataset(D_L)
        # and save to the file($DATASET_$NUM_patient_set.txt).
        if self.phase == 'train' and hasattr(self.conf, 'num_patients'):
            if not self.conf.has_patient_set: 
                if not self.conf.num_patients % len(self.conf.vendors['train']) == 0:
                    raise ValueError('Not matched.')
                
                os.makedirs(f'dataset/patient_set/{self.conf.name}', exist_ok=True)
                with open(f'dataset/patient_set/{self.conf.name}/'
                    + f'{self.conf.name}_{self.conf.num_patients}p_{SEED}_patient_set.txt', 'w') as f:

                    # sample the same number of patient for each vendors.
                    patient_set = []
                    for vendor in self.conf.vendors['train']:
                        vendor_set = set([os.path.basename(file).split('_')[0] \
                                            for file in glob.glob(f'{conf.data}/{vendor}/*_ED_gt.nii.gz')])
                        vendor_set = random.sample(vendor_set, self.conf.num_patients//len(self.conf.vendors['train']))
                
                        for patient in vendor_set:
                            f.write(vendor+'/'+patient+'\n')
                        patient_set += vendor_set
                    patient_set = set(patient_set)

            else:
                with open(f'dataset/patient_set/{self.conf.name}/'
                    + f'{self.conf.name}_{self.conf.num_patients}p_{SEED}_patient_set.txt', 'r') as f:
                    patient_set = set([line.replace('\n','')[2:] for line in f])

        self.data_name, data = [], []
        for file in filelist:
            if phase == 'train':
                if os.path.basename(file).split('_')[0] not in patient_set:
                    continue
                n_slice = nib.load(file).get_fdata().shape[-1]
                for sdx in range(n_slice):
                    data.append((file.replace('_gt', ''), file, sdx))
                    self.data_name.append(os.path.basename(file).split('_gt')[0]+f'_s{sdx}')
            else:
                data.append((file.replace('_gt', ''), file))
                self.data_name.append(os.path.basename(file).split('_gt')[0])

        if phase == 'train': self._get_occurence_map(data)
        return data

class MnMVolumetricSet(VolumetricSet):
    def __init__(self, conf, phase, transform=None):
        super().__init__(conf, phase, transform)
        
    def _init_data(self, phase):
        conf = getattr(self.conf, 'train')
        filelist = sum([glob.glob(f'{conf.data}/{vendor}/*ED.nii.gz') for vendor in self.conf.vendors['train']], [])

        self.data_name, data = [], []
        for file in filelist:
            if '_gt' in file: continue
            data.append((file, ''))
            self.data_name.append(os.path.basename(file).split('ED.nii')[0])
        return data

class CAMUSSequenceSet(DataSetTemplate):
    def __init__(self, conf, phase, transform=None):
        super().__init__(conf, phase, transform)
        self.sample_cache = []
        self.select_cache = []

    def _init_data(self, phase):
        conf = getattr(self.conf, phase)
        meta = np.load(conf.data)

        self.meta = {'A2C': meta[0,...], 'A4C': meta[1,...]}
 
        self.data_name, data = [], []
        for idx in range(self.meta['A2C'].shape[0]):
            for view in self.conf.use_view:
                data += [(view, idx)]
                self.data_name.append(f'{view}_pat{idx}')
        return data
       
    def pull_item(self, i):
        return self.pull_train_item(i)

    def pull_train_item(self, i, cnt=0):
        view, idx = self.data[i]
        data = self.meta[view][idx]

        # select N(num_samples) slices within the max distance(max_dist).
        if cnt == 0:
            select = [0, 100]
            while max(select) - min(select) + 1 > cfg.max_dist:
                select = random.sample(range(data.shape[0]), cfg.num_samples)
            self.min_select = max(max(select) - cfg.max_dist,  0)
            self.max_select = self.min_select + cfg.max_dist
        else:
            while data.shape[0] - self.min_select < cfg.num_samples:
                idx = random.sample(range(len(self.data)), 1)[0]
                data = np.transpose(nib.load(self.data[idx][0]).get_fdata(), [2,0,1])
            select = random.sample(range(self.min_select, min(self.max_select, data.shape[0])), cfg.num_samples)

        sample = data[select]
        self.sample_cache += [sample]
        self.select_cache += [select]

    def get_train_items(self):
        sample = np.concatenate(self.sample_cache, axis=0)
        select = np.concatenate(self.select_cache, axis=0)

        if self.transform is not None:
            sample = self.transform(sample, None, self.flow_mask)[0]

        self.sample_cache = []
        self.select_cache = []

        return torch.FloatTensor(sample.copy()).unsqueeze(1), \
               torch.LongTensor(select)

class CAMUSImageSet(DataSetTemplate):
    def __init__(self, conf, phase, transform=None, target_transform=None):
        super().__init__(conf, phase, transform, target_transform)

    def _init_data(self, phase):
        conf = getattr(self.conf, phase)
        meta, gt = np.load(conf.data), np.load(conf.data.replace('.npy', '_gt.npy'))
        self.meta = {'A2C': (meta[0,...], gt[0,...]), 'A4C': (meta[1,...], gt[1,...])}

        # randomly select patients to construct the labeled dataset(D_L)
        # and save to the file($DATASET_$NUM_patient_set.txt).
        patient_set = list(range(self.meta['A2C'][0].shape[0]))
        if self.phase == 'train' and hasattr(self.conf, 'num_patients'):
            if not self.conf.has_patient_set: 
                patient_set = random.sample(patient_set, self.conf.num_patients)
                os.makedirs(f'dataset/patient_set/{self.conf.name}', exist_ok=True)
                with open(f'dataset/patient_set/{self.conf.name}/'
                    + f'{self.conf.name}_{self.conf.num_patients}p_{SEED}_patient_set.txt', 'w') as f:
                    for patient in patient_set:
                        f.write(str(patient)+'\n')
            else:
                with open(f'dataset/patient_set/{self.conf.name}/'
                    + f'{self.conf.name}_{self.conf.num_patients}p_{SEED}_patient_set.txt', 'r') as f:
                    patient_set = set([line.replace('\n','') for line in f])
        
        self.data_name, data = [], []
        for idx in patient_set:
            for view in self.conf.use_view:
                for key in ('ED', 'ES'):
                    data += [(view, int(idx), key)]
                    self.data_name.append(f'{view}_pat{idx}_{key}')   

        if phase == 'train': self._get_occurence_map(data)
        return data

    def _get_occurence_map(self, data):
        _dict = {'ED':0, 'ES':-1}

        gt_mask = np.zeros((256,256))
        for i in range(len(data)):
            view, idx, key = data[i]
            gt = self.meta[view][1][idx][_dict[key]]
            gt_mask += np.greater(gt, 0)

        self.gt_mask = gt_mask / len(data)
        self.flow_mask = gaussian(1. - self.gt_mask)
       
    def pull_item(self, i):
        if self.phase == 'train':
            return self.pull_train_item(i)
        elif self.phase in ('valid', 'test'):
            return self.pull_valid_item(i)

    def pull_train_item(self, i):
        _dict = {'ED':0, 'ES':-1}

        view, idx, key = self.data[i]
        data, gt = self.meta[view][0][idx][_dict[key]], \
                   self.meta[view][1][idx][_dict[key]]
        
        if self.transform is not None:
            data, gt = self.transform(data, gt, self.flow_mask)

        data = torch.FloatTensor(data).unsqueeze(0)
        gt = torch.FloatTensor(gt)
        return data, gt

    def pull_valid_item(self, i):
        _dict = {'ED':0, 'ES':-1}

        view, idx, key = self.data[i]
        data, gt = self.meta[view][0][idx][_dict[key]], \
                   self.meta[view][1][idx][_dict[key]]
        
        data = torch.FloatTensor(data).unsqueeze(0)
        gt = torch.FloatTensor(gt)
        return data, gt

# ----------------------------------------------------------------------------------------------------------------

class CGLCAMUSPreTrainSet(DataSetTemplate):
    def __init__(self, conf, phase, transform=None):
        super().__init__(conf, phase, transform)

    def _init_data(self, phase):
        conf = getattr(self.conf, phase)
        meta = np.load(conf.data)
        self.meta = {'A2C': meta[0,...], 'A4C': meta[1,...]}
 
        self.data_name, data = [], []
        for idx in range(self.meta['A2C'].shape[0]):
            for view in self.conf.use_view:
                data += [(view, idx)]
                self.data_name.append(f'{view}_pat{idx}')
        return data

    def pull_item(self, i):
        return self.pull_train_item(i)

    def pull_train_item(self, i):
        view, idx = self.data[i]
        data = self.meta[view][idx]

        part_idx = [0] + [(p+1) * data.shape[0] // cfg.num_parts \
                                            for p in range(cfg.num_parts)]

        # randomly select slices from each partitions.
        sample = []
        raw_samples, aug_sample1, aug_sample2 = [], [], []
        for pdx in range(len(part_idx)-1):
            select = random.sample(range(part_idx[pdx], part_idx[pdx+1]), 1)
            sample = data[select]

            if self.transform is not None:
                aug1 = self.transform(sample[0].copy())[0]
                aug2 = self.transform(sample[0].copy())[0]

            raw_samples += [sample]
            aug_sample1 += [aug1[None,...]]
            aug_sample2 += [aug2[None,...]]

        raw_samples = np.concatenate(raw_samples, 0)
        aug_sample1 = np.concatenate(aug_sample1, 0)
        aug_sample2 = np.concatenate(aug_sample2, 0)

        return torch.FloatTensor(raw_samples.copy()).unsqueeze(1), \
               torch.FloatTensor(aug_sample1.copy()).unsqueeze(1), \
               torch.FloatTensor(aug_sample2.copy()).unsqueeze(1)

class CGLACDCPreTrainSet(DataSetTemplate):
    def __init__(self, conf, phase, transform=None):
        super().__init__(conf, phase, transform)

    def _init_data(self, phase):
        conf = getattr(self.conf, phase)
        filelist = glob.glob(conf.data)

        self.data_name, data = [], []
        for file in filelist:
            data.append((file.replace('_gt', ''), file))
            self.data_name.append(os.path.basename(file).split('_gt')[0])
        return data

    def pull_item(self, i):
        return self.pull_train_item(i)

    def pull_train_item(self, i):
        data = np.transpose(nib.load(self.data[i][0]).get_fdata(), [2,0,1])
        data = data.astype(np.float32)

        part_idx = [0] + [(p+1) * data.shape[0] // cfg.num_parts \
                                            for p in range(cfg.num_parts)]

        # randomly select slices from each partitions.
        raw_samples, aug_sample1, aug_sample2 = [], [], []
        for pdx in range(len(part_idx)-1):
            select = random.sample(range(part_idx[pdx], part_idx[pdx+1]), 1)
            sample = data[select]

            if self.transform is not None:
                aug1 = self.transform(sample[0].copy())[0]
                aug2 = self.transform(sample[0].copy())[0]

            raw_samples += [sample]
            aug_sample1 += [aug1[None,...]]
            aug_sample2 += [aug2[None,...]]

        raw_samples = np.concatenate(raw_samples, 0)
        aug_sample1 = np.concatenate(aug_sample1, 0)
        aug_sample2 = np.concatenate(aug_sample2, 0)

        return torch.FloatTensor(raw_samples.copy()).unsqueeze(1), \
               torch.FloatTensor(aug_sample1.copy()).unsqueeze(1), \
               torch.FloatTensor(aug_sample2.copy()).unsqueeze(1)

class CGLMnMPreTrainSet(CGLACDCPreTrainSet):
    def __init__(self, conf, phase, transform=None):
        super().__init__(conf, phase, transform)

    def _init_data(self, phase):
        conf = getattr(self.conf, 'train')
        filelist = sum([glob.glob(f'{conf.data}/{vendor}/*ED.nii.gz') for vendor in self.conf.vendors['train']], [])

        self.data_name, data = [], []
        for file in filelist:
            if '_gt' in file: continue
            data.append((file, ''))
            self.data_name.append(os.path.basename(file).split('ED.nii')[0])
        return data

# -------------------------------------------------------------------------------------------------------------

def gaussian(x, mu=0., sigma=0.7):
    sigma = np.power(sigma, 2)
    alpha = 1. / np.power(2*np.pi*sigma, 0.5) 
    return alpha * np.exp(- np.power((x-mu),2) / (2*sigma)) * 2 

# -------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    from config import set_cfg, cfg
    from augmentations import ACDCImageAugmentator
    import sys
    if len(sys.argv) > 1:
        set_cfg(sys.argv[1])

    set_range = lambda x: (x-np.min(x)) / (np.max(x)-np.min(x)+1e-8) * 255

    def plot_and_save_fig(arr, filename):
        import matplotlib.pyplot as plt
        print(np.max(arr), np.min(arr))
        plt.imshow(arr, cmap='jet')
        plt.colorbar()
        plt.savefig(f'{filename}.png')
        plt.close()

    from exphandler import seed_everything
    seed_everything(111)

    dataset = ImageSet(cfg.dataset, 'train', transform=ACDCImageAugmentator(cfg.transformer))
    loader  = torch.utils.data.DataLoader(dataset, 
                batch_size=4, shuffle=False, num_workers=4)

    print(len(loader))
    for datum in loader:
        data, gt = datum
        print(data.size(), gt.size())
        os.makedirs('$VISUAL_PATH', exist_ok=True)

        for i, (im, t) in enumerate(zip(data, gt)):
            cv2.imwrite(f'$VISUAL_PATH/raw_im{i}.png', set_range(im.squeeze().numpy()))
            cv2.imwrite(f'$VISUAL_PATH/raw_gt{i}.png', set_range(t.squeeze().numpy()))
        break
    
    exit()
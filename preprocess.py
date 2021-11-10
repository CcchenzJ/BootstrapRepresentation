
import numpy as np
import nibabel as nib
import skimage.transform as skt

class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, masks=None):
        for t in self.transforms:
            image, masks = t(image, masks)
        return image, masks

class MinMaxNormalize(object):
    def __init__(self, min_val=1, max_val=99):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, image, masks=None):
        min_val_pix = np.percentile(image, self.min_val)
        max_val_pix = np.percentile(image, self.max_val)

        # min-max norm on total 3D volume
        image = (image - min_val_pix) / (max_val_pix - min_val_pix)
        return image, masks

class ReSample(object):
    def __init__(self, resolution, affine_matrix):
        self.resolution = resolution
        self.affine_matrix = affine_matrix
    
    def __call__(self, image, masks=None):
        # for idx, res in enumerate(self.resolution):
        #     self.affine_matrix[idx,idx] = -res
        
        new_affine = self.rescale_affine()

        image = nib.Nifti1Image(image, new_affine)
        if masks is not None:
            masks = nib.Nifti1Image(
                np.round(masks).astype(np.int16), new_affine)
        return image, masks

    def rescale_affine(self):
        ret = np.array(self.affine_matrix, copy=True)
        zoom_len = len(self.resolution)
        RZS = self.affine_matrix[:zoom_len, :zoom_len]
        zooms = np.sqrt(np.sum(RZS ** 2, axis=0))
        scale = np.divide(self.resolution, zooms)
        ret[:zoom_len, :zoom_len] = RZS * np.diag(scale)
        return ret

class ReScale(object):
    def __init__(self, pixel_size, resolution):
        self.scale_vector = [ps / rs for ps,rs in \
                                zip(pixel_size, resolution)]
    def __call__(self, image, masks=None):
        
        if len(self.scale_vector) < len(image.shape):
            new_image, new_masks = [], []
            for slice_idx in range(image.shape[-1]):
                slice_image = image[...,slice_idx]
                slice_image = skt.rescale(slice_image, 
                                          self.scale_vector,
                                          order=1,
                                          preserve_range=True,
                                          mode='constant')
                new_image += [slice_image]

                if masks is not None:
                    slice_masks = masks[...,slice_idx]
                    slice_masks = skt.rescale(slice_masks,
                                              self.scale_vector,
                                              order=0,
                                              preserve_range=True,
                                              mode='constant')
                    new_masks += [slice_masks]
            new_image = np.stack(new_image, -1)
            new_masks = np.stack(new_masks, -1) if masks is not None else None
        else:
            new_image = skt.rescale(image,
                                    self.scale_vector,
                                    order=1,
                                    preserve_range=True,
                                    mode='constant')
            if masks is not None:
                new_masks = skt.rescale(masks,
                                        self.scale_vector,
                                        order=0,
                                        preserve_range=True,
                                        mode='constant')

        return new_image, new_masks

class CropOrPad(object):
    def __init__(self, new_x, new_y):
        self.new_x = new_x
        self.new_y = new_y
    
    def __call__(self, image, masks=None):
        image = self._crop(image)
        if masks is not None:
            masks = self._crop(masks)

        return image, masks
        
    def _crop(self, image):
        x, y, z = image.shape
        x_s = (x - self.new_x) // 2
        y_s = (y - self.new_y) // 2
        x_c = (self.new_x - x) // 2
        y_c = (self.new_y - y) // 2

        cropped = np.zeros((self.new_x, self.new_y, z))
        if x > self.new_x and y > self.new_y:
            cropped = image[x_s: x_s+self.new_x, y_s: y_s+self.new_y, :]
        else:
            if x <= self.new_x and y > self.new_y:
                cropped[x_c: x_c+x, :, :] = image[:, y_s: y_s+self.new_y, :]
            elif x > self.new_x and y <= self.new_y:
                cropped[:, y_c: y_c+y, :] = image[x_s: x_s+self.new_x, :, :]
            else:
                cropped[x_c: x_c+x, y_c: y_c+y, :] = image[:, :, :]
        return cropped

class VolACDCPreProcess(object):
    def __init__(self, resolution, pixel_size, affine_matrix):
        self.augment = Compose([
            MinMaxNormalize(1, 99),
            ReScale(pixel_size, resolution),
            CropOrPad(192, 192),
            ReSample(resolution, affine_matrix)
        ])

    def __call__(self, image, masks):
        return self.augment(image, masks)

class VolProstatePreProcess(object):
    def __init__(self, resolution, pixel_size, affine_matrix):
        self.augment = Compose([
            MinMaxNormalize(1, 99),
            ReScale(pixel_size, resolution),
            CropOrPad(192, 192),
            ReSample(resolution, affine_matrix)
        ])

    def __call__(self, image, masks):
        return self.augment(image, masks)

class VolMnMPreProcess(object):
    def __init__(self, resolution, pixel_size, affine_matrix):
        self.augment = Compose([
            MinMaxNormalize(1, 99),
            ReScale(pixel_size, resolution),
            CropOrPad(192, 192),
            ReSample(resolution, affine_matrix)
        ])

    def __call__(self, image, masks):
        return self.augment(image, masks)

def main_prostate():
    datadir = '../Prostate/Task05_Prostate'
    savedir = '../Prostate/PreProcessed'

    import os, glob
    os.makedirs(savedir, exist_ok=True)

    data = []
    for file in glob.glob(datadir+'/imagesTr/*.nii.gz'):
        data.append((file, file.replace('imagesTr', 'labelsTr')))
    print(len(data))

    for (volume, gt) in data:
        vol_name = os.path.basename(volume)
        gt_name  = os.path.basename(gt)

        tagger = nib.load(volume)
        pixel_size = tagger.header['pixdim'][1:4]
        affine_matrix = tagger.affine

        volume = tagger.get_fdata()[...,0]
        gt = nib.load(gt).get_fdata()

        preprocessor = VolProstatePreProcess(
            (0.625, 0.625), pixel_size, affine_matrix)

        new_volume, new_gt = preprocessor(volume, gt)
        nib.save(new_volume, savedir+'/'+vol_name)
        if new_gt is not None:
            nib.save(new_gt, savedir+'/'+gt_name.replace('.nii.gz', '_gt.nii.gz'))
        
        del preprocessor
        print('\r Processing', vol_name)

def main_acdc():
    datadir = '../ACDC/data/*_gt.nii.gz'
    savedir = '../ACDC/new_data'

    import os, glob
    os.makedirs(savedir, exist_ok=True)

    data = []
    for file in glob.glob(datadir):
        data.append((file.replace('_gt', ''), file))
    print(len(data))

    judge_phase = lambda idx: ('ED' if int(idx) % 2 == 0 else 'ES', int(idx) // 2)
    for (volume, gt) in data:
        vol_name = os.path.basename(volume)
        gt_name  = os.path.basename(gt)

        pat_index = vol_name.split('_')[-1].split('.')[0]
        phase, pat_idx = judge_phase(pat_index)
        vol_name = vol_name.replace(pat_index, f'{pat_idx}_{phase}')
        gt_name = gt_name.replace(pat_index, f'{pat_idx}_{phase}')

        tagger = nib.load(volume)
        pixel_size = tagger.header['pixdim'][1:4]
        affine_matrix = tagger.affine

        volume = tagger.get_fdata()
        gt = nib.load(gt).get_fdata()

        preprocessor = VolACDCPreProcess(
            (1.36719,1.36719), pixel_size, affine_matrix)

        new_volume, new_gt = preprocessor(volume, gt)
        nib.save(new_volume, savedir+'/'+vol_name.replace('pat_', 'pat'))
        if new_gt is not None:
            nib.save(new_gt, savedir+'/'+gt_name.replace('pat_', 'pat'))
        
        del preprocessor
        print('\r Processing', vol_name)

def main_mnm():
    datadir = '../MnM/'    
    savedir = '../MnM/Processed_MnM'
    infodir = '../MnM/*.csv'
    
    import os, glob
    import pandas as pd
    info = pd.read_csv(glob.glob(infodir)[-1]).values
    info_dict = {}
    for subject in info:
        name, vendor_name, vendor, centre, ed, es = subject
        info_dict[name] = (vendor_name, vendor, centre, ed, es)

    # data = {}       
    # for root in ('Training', 'Validation','Testing'): 
    #     os.makedirs(savedir+'/'+root, exist_ok=True)
    #     data[root] = []
    #     for rdir, _, files in os.walk(datadir+root):
    #         if len(files) == 0: continue
        
    #         for file in files:
    #             if '_gt' not in file: continue
    #             file = rdir +'/'+ file
    #             data[root].append((file.replace('_gt', ''), file))
    # print([len(data[key]) for key in data])
    
    data = {}       
    for root in ('Unlabeled', ):
        os.makedirs(savedir+'/'+root, exist_ok=True)
        data[root] = []
        for rdir, _, files in os.walk(datadir+root):
            if len(files) == 0: continue
        
            for file in files:
                file = rdir +'/'+ file
                data[root].append((file, None))
    print([len(data[key]) for key in data])

    for root in data:
        for (volume, gt) in data[root]:
            vol_name = os.path.basename(volume)
            if gt is not None:
                gt_name  = os.path.basename(gt)

            ed, es = info_dict[vol_name.split('_')[0]][-2:]

            tagger = nib.load(volume)
            pixel_size = tagger.header['pixdim'][1:4]
            affine_matrix = tagger.affine

            preprocessor = VolMnMPreProcess(
                (1.25,1.25), pixel_size, affine_matrix)
            
            for time, idx in zip(('ED', 'ES'), (ed, es)):
                _volume = tagger.get_fdata()[...,idx]
                _gt = gt if gt is None else \
                        nib.load(gt).get_fdata()[...,idx]

                new_volume, new_gt = preprocessor(_volume, _gt)
                nib.save(new_volume, savedir+'/'+root+'/' \
                        + vol_name.replace('.nii', '_'+time+'.nii'))
                if new_gt is not None:
                    nib.save(new_gt, savedir+'/'+root+'/' \
                        + gt_name.replace('_gt.', '_'+time+'_gt.'))
            
            del preprocessor
            print('\r Processing', vol_name)

if __name__ == '__main__':
    main_acdc()
    
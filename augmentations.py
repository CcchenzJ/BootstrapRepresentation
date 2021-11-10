
import torch
import torch.nn.functional as F

import cv2 as cv2
import numpy as np

import types
from numpy import random
from math import sqrt

from scipy.ndimage import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from skimage.transform import rescale, resize
from skimage.exposure import match_histograms
from preprocess import CropOrPad

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

    def __call__(self, image, masks=None, fm=None, rf=None):
        for t in self.transforms:
            if isinstance(t, RandomWarp) or isinstance(t, RandomVolWarp):
                image, masks = t(image, masks, fm)
            else:
                image, masks = t(image, masks)    
        return image, masks

class Choice(object):
    """Choose one augmentation from several augmentations. """

    def __init__(self, transforms):
        self.transforms = transforms + [do_nothing]

    def __call__(self, image, masks=None):
        kk = random.randint(len(self.transforms))
        image, masks = self.transforms[kk](image, masks)
        return image, masks

# -------------------------------------------------------------------------------------------

class RandomWarp(object):
    def __init__(self, mu=0., sigma=5., num_classes=4):
        self.mu = mu
        self.sigma = sigma
        self.num_classes = num_classes

    def __call__(self, image, masks=None, flow_mask=None):
        img_h, img_w = image.shape
        if random.randint(2):
            # if flow_mask is None:
            flow = self._generate_flow(img_h, img_w)
            # else:
            #     flow = self._generate_masked_flow(img_h, img_w, flow_mask)
            self.flow = flow

            image = self._dense_image_warp(
                np.expand_dims(image, axis=2), flow, img_h, img_w, 1)
            image = np.reshape(image, [img_h, img_w])

            if masks is not None:
                masks = self._dense_image_warp(
                        self._onehot(masks), flow, img_h, img_w, self.num_classes)
                masks = np.argmax(masks, axis=2)

        return image, masks 

    def _dense_image_warp(self, image, flow, h, w, c):
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        stacked_grid = np.stack([grid_y, grid_x], axis=2).astype(flow.dtype)

        query_points_on_grid = stacked_grid - flow
        query_points_flattened = np.reshape(query_points_on_grid, [h * w, 2])

        interpolated = self._interpolate_bilinear(image, query_points_flattened)
        interpolated = np.reshape(interpolated, [h, w, c])

        return interpolated

    def _generate_flow(self, h, w):
        flow_vec = np.zeros((h, w, 2))

        dx = np.random.normal(self.mu, self.sigma, 9)
        dx_mat = np.reshape(dx, (3,3))
        dx_img = resize(dx_mat, output_shape=(h, w), order=3, mode='reflect')

        dy = np.random.normal(self.mu, self.sigma, 9)
        dy_mat = np.reshape(dy, (3,3))
        dy_img = resize(dy_mat, output_shape=(h, w), order=3, mode='reflect')

        flow_vec[:,:,0] = dx_img
        flow_vec[:,:,1] = dy_img

        return flow_vec

    def _generate_masked_flow(self, h, w, mask):
        flow_vec = np.zeros((h, w, 2))

        dx = np.random.normal(self.mu, self.sigma, 9)
        dx_mat = np.reshape(dx, (3,3))
        dx_img = resize(dx_mat, output_shape=(h, w), order=3, mode='reflect')

        dy = np.random.normal(self.mu, self.sigma, 9)
        dy_mat = np.reshape(dy, (3,3))
        dy_img = resize(dy_mat, output_shape=(h, w), order=3, mode='reflect')

        flow_vec[:,:,0] = dx_img * mask
        flow_vec[:,:,1] = dy_img * mask

        return flow_vec

    def _onehot(self, masks):
        y_onehot = []
        for _cls in range(self.num_classes):
            onehot = np.zeros_like(masks)
            onehot[(masks==_cls)] = 1
            y_onehot += [onehot]
        return np.stack(y_onehot, axis=2)

    def _interpolate_bilinear(self, grid, query_points, indexing='ij'):
        if indexing != 'ij' and indexing != 'xy':
            raise ValueError('Indexing mode must be \'ij\' or \'xy\'')

        height, width, channels = grid.shape
        grid_type = grid.dtype
        num_queries = query_points.shape[0]

        alphas = []
        floors = []
        ceils  = []
        index_order = [0, 1] if indexing == 'ij' else [1, 0]
        unstacked_query_points = np.split(query_points, 2, axis=1)
        
        for dim in index_order:
            queries = unstacked_query_points[dim]
            size_in_indexing_dimension = grid.shape[dim]

            # max_floor is size_in_indexing_dimension - 2 so that max_floor + 1
            # is still a valid index into the grid.
            max_floor = (size_in_indexing_dimension - 2.)
            min_floor = 0.
            floor = np.minimum(np.maximum(min_floor, np.floor(queries)), max_floor)

            int_floor = floor.astype(np.int32)            
            floors.append(int_floor)
            ceil = int_floor + 1
            ceils.append(ceil)

            # alpha has the same type as the grid, as we will directly use alpha
            # when taking linear combinations of pixel values from the image.
            alpha = (queries - floor).astype(grid.dtype)
            min_alpha = 0.
            max_alpha = 1.
            alpha = np.minimum(np.maximum(min_alpha, alpha), max_alpha)

            # Expand alpha to [b, n, 1] so we can use broadcasting
            # (since the alpha values don't depend on the channel).
            # alpha = np.expand_dims(alpha, 2)
            alphas.append(alpha)

        flattened_grid = np.reshape(grid, [height * width, channels])
        # batch_offsets = np.reshape(height * width, [batch_size, 1])

        # This wraps array_ops.gather. We reshape the image data such that the
        # batch, y, and x coordinates are pulled into the first dimension.
        # Then we gather. Finally, we reshape the output back. It's possible this
        # code would be made simpler by using array_ops.gather_nd.
        def gather(y_coords, x_coords):
            linear_coordinates = y_coords * width + x_coords
            gathered_values = flattened_grid[linear_coordinates]
            return np.reshape(gathered_values, [num_queries, channels])

        # grab the pixel values in the 4 corners around each query point
        top_left = gather(floors[0], floors[1])
        top_right = gather(floors[0], ceils[1])
        bottom_left = gather(ceils[0], floors[1])
        bottom_right = gather(ceils[0], ceils[1])

        # now, do the actual interpolation
        interp_top = alphas[1] * (top_right - top_left) + top_left
        interp_bottom = alphas[1] * (bottom_right - bottom_left) + bottom_left
        interp = alphas[0] * (interp_bottom - interp_top) + interp_top

        return interp

class RandomVolWarp(RandomWarp):
    def __init__(self, mu=0., sigma=5., num_classes=4):
        super().__init__(mu, sigma, num_classes)

    def __call__(self, image, masks=None, flow_mask=None):
        img_h, img_w = image.shape[1:]
        flow = self._generate_masked_flow(img_h, img_w, flow_mask)

        if random.randint(2):
            new_image = []
            for idx in range(len(image)):
                new = self._dense_image_warp(
                    np.expand_dims(image[idx], axis=2), flow, img_h, img_w, 1)
                new = np.reshape(new, [img_h, img_w])
                new_image += [new]
            image = np.stack(new_image)    
        return image, masks

class RandomCrop(object):
    def __init__(self, lower=10, upper=70, box_size=100):
        self.lower = lower
        self.upper = upper
        self.box_size = box_size
    
    def __call__(self, image, masks=None):
        img_h, img_w = image.shape
        if random.randint(2):
            x = random.randint(low=self.lower, high=self.upper)
            y = random.randint(low=self.lower, high=self.upper)
        
            image = image[x: x+self.box_size, y:y+self.box_size]
            image = resize(image, (img_h, img_w), order=1)

            if masks is not None:
                masks = masks[x: x+self.box_size, y:y+self.box_size]
                masks = resize(masks, (img_h, img_w), order=0)
        return image, masks

class RandomScale(object):
    def __init__(self, lower=0.9, upper=1.1):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "scale upper must be >= lower."
        assert self.lower >= 0, "scale lower must be non-negative."

    def __call__(self, image, masks=None):
        img_h, img_w = image.shape
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image = rescale(image, alpha, 
                        order=1, preserve_range=True, mode='constant')
            image = self._crop(image, img_h, img_w)

            if masks is not None:
                masks = rescale(masks, alpha, 
                        order=0, preserve_range=True, mode='constant')
                masks = self._crop(masks, img_h, img_w)

        return image, masks

    def _crop(self, image, new_x, new_y):
        x, y = image.shape
        x_s = (x - new_x) // 2
        y_s = (y - new_y) // 2
        x_c = (new_x - x) // 2
        y_c = (new_y - y) // 2

        cropped = np.zeros((new_x, new_y))
        if x > new_x and y > new_y:
            cropped = image[x_s: x_s+new_x, y_s: y_s+new_y]
        else:
            if x <= new_x and y > new_y:
                cropped[x_c: x_c+x, :] = image[:, y_s: y_s+new_y]
            elif x > new_x and y <= new_y:
                cropped[:, y_c: y_c+y] = image[x_s: x_s+new_x, :]
            else:
                cropped[x_c: x_c+x, y_c: y_c+y] = image[:, :]
        return cropped

class RandomContrast(object):
    def __init__(self, lower=0.7, upper=1.3):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, masks=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
            image = np.clip(image, 0, 1.5)
        return image, masks

class RandomBrightness(object):
    def __init__(self, delta=0.3):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, masks=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, masks

class RandomGaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, image, masks=None):
        if random.randint(2):
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            image = gaussian_filter(image, sigma)
        return image, masks

class RandomMirror(object):
    def __call__(self, image, masks=None):
        if random.randint(2):
            image = image[:, ::-1].copy()
            if masks is not None:
                masks = masks[:, ::-1].copy()
        return image, masks

class RandomFlip(object):
    def __call__(self, image, masks=None):
        if random.randint(2):
            image = image[::-1, :].copy()
            if masks is not None:
                masks = masks[::-1, :].copy()
        return image, masks

class RandomRot90(object):
    def __call__(self, image, masks=None):
        k = random.randint(4)
        image = np.rot90(image,k, axes=(0,1)).copy()
        if masks is not None:
            masks = np.rot90(masks,k, axes=(0,1)).copy()
        return image, masks

# --------------------------------------------------------------------------------

class RandomVolContrast(object):
    def __init__(self, lower=0.7, upper=1.3):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, masks=None):
        if random.randint(2):
            for idx in range(len(image)):
                alpha = random.uniform(self.lower, self.upper)
                image[idx,...] = image[idx,...] * alpha
                image[idx,...] = np.clip(image[idx,...], 0, 1.5)
        return image, masks

class RandomVolBrightness(object):
    def __init__(self, delta=0.3):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, masks=None):
        if random.randint(2):
            for idx in range(len(image)):
                delta = random.uniform(-self.delta, self.delta)
                image[idx,...] += delta
        return image, masks

class RandomVolGaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, image, masks=None):
        if random.randint(2):
            for idx in range(len(image)):
                sigma = random.uniform(self.sigma[0], self.sigma[1])
                image[idx,...] = gaussian_filter(image[idx,...], sigma)
        return image, masks

class RandomVolMirror(object):
    def __call__(self, image, masks=None):
        if random.randint(2):
            image = image[:, :, ::-1]
            if masks is not None:
                masks = masks[:, :, ::-1]
        return image, masks

class RandomVolFlip(object):
    def __call__(self, image, masks=None):
        if random.randint(2):
            image = image[:, ::-1, :]
            if masks is not None:
                masks = masks[:, ::-1, :]
        return image, masks

class RandomVolRot90(object):
    def __call__(self, image, masks=None):
        k = random.randint(4)
        image = np.rot90(image,k, axes=(1,2))
        if masks is not None:
            masks = np.rot90(masks,k, axes=(1,2))
        return image, masks

# ----------------------------------------------------------------------------

def do_nothing(image=None, masks=None, fm=None):
    return image, masks

def enable_if(condition, obj):
    return obj if condition else do_nothing

class ACDCVolumeAugmentator(object):
    """ Transform to be used when training. """

    def __init__(self, conf):
        self.augment = Compose([
            enable_if(conf.is_RandomWarp, RandomVolWarp(0., 10., 4)),
            enable_if(conf.is_RandomFlip, RandomVolFlip()),
            enable_if(conf.is_RandomMirror, RandomVolMirror()),
            enable_if(conf.is_RandomRot90, RandomVolRot90()),
            enable_if(conf.is_RandomGaussianBlur, RandomVolGaussianBlur([.1, 2.])),
            enable_if(conf.is_RandomBrightness, RandomVolBrightness(0.4)),
            enable_if(conf.is_RandomContrast, RandomVolContrast(0.4, 1.6)),
        ])

    def __call__(self, image, masks=None, fm=None):
        return self.augment(image, masks, fm)

class ACDCImageAugmentator(object):
    """ Transform to be used when training. """

    def __init__(self, conf):
        self.augment = Compose([
            enable_if(conf.is_RandomWarp, RandomWarp(0., 10., 4)),
            enable_if(conf.is_RandomFlip, RandomFlip()),
            enable_if(conf.is_RandomMirror, RandomMirror()),
            enable_if(conf.is_RandomRot90, RandomRot90()),
            enable_if(conf.is_RandomGaussianBlur, RandomGaussianBlur([.1, 2.])),
            enable_if(conf.is_RandomBrightness, RandomBrightness(0.1)),
            enable_if(conf.is_RandomContrast, RandomContrast(0.2, 1.8)),
        ])

    def __call__(self, image, masks=None, fm=None):
        return self.augment(image, masks, fm)
       
class MnMVolumeAugmentator(object):
    """ Transform to be used when training. """

    def __init__(self, conf):
        self.augment = Compose([
            enable_if(conf.is_RandomWarp, RandomVolWarp(0., 10., 4)),
            enable_if(conf.is_RandomFlip, RandomVolFlip()),
            enable_if(conf.is_RandomMirror, RandomVolMirror()),
            enable_if(conf.is_RandomRot90, RandomVolRot90()),
            enable_if(conf.is_RandomGaussianBlur, RandomVolGaussianBlur([.1, 2.])),
            enable_if(conf.is_RandomBrightness, RandomVolBrightness(0.3)),
            enable_if(conf.is_RandomContrast, RandomVolContrast(0.2, 1.8)),
        ])

    def __call__(self, image, masks=None, fm=None):
        return self.augment(image, masks, fm)

class MnMImageAugmentator(object):
    """ Transform to be used when training. """

    def __init__(self, conf):
        self.augment = Compose([
            # RandomHistMatching(),
            enable_if(conf.is_RandomWarp, RandomWarp(0., 10., 4)),
            enable_if(conf.is_RandomFlip, RandomFlip()),
            enable_if(conf.is_RandomMirror, RandomMirror()),
            enable_if(conf.is_RandomRot90, RandomRot90()),
            enable_if(conf.is_RandomGaussianBlur, RandomGaussianBlur([.1, 2.])),
            enable_if(conf.is_RandomBrightness, RandomBrightness(0.1)),
            enable_if(conf.is_RandomContrast, RandomContrast(0.3, 1.7)),
        ])

    def __call__(self, image, masks=None, fm=None, rf=None):
        return self.augment(image, masks, fm, rf)

class ProstateVolumeAugmentator(object):
    """ Transform to be used when training. """

    def __init__(self, conf):
        self.augment = Compose([
            enable_if(conf.is_RandomWarp, RandomVolWarp(0., 10., 3)),
            enable_if(conf.is_RandomFlip, RandomVolFlip()),
            enable_if(conf.is_RandomMirror, RandomVolMirror()),
            enable_if(conf.is_RandomRot90, RandomVolRot90()),
            enable_if(conf.is_RandomGaussianBlur, RandomVolGaussianBlur([.1, 2.])),
            enable_if(conf.is_RandomBrightness, RandomVolBrightness(0.3)),
            enable_if(conf.is_RandomContrast, RandomVolContrast(0.6, 1.4)),
        ])

    def __call__(self, image, masks=None, fm=None):
        return self.augment(image, masks, fm)

class ProstateImageAugmentator(object):
    """ Transform to be used when training. """

    def __init__(self, conf):
        self.augment = Compose([
            enable_if(conf.is_RandomWarp, RandomWarp(0., 10., 3)),
            enable_if(conf.is_RandomFlip, RandomFlip()),
            enable_if(conf.is_RandomMirror, RandomMirror()),
            enable_if(conf.is_RandomRot90, RandomRot90()),
            enable_if(conf.is_RandomGaussianBlur, RandomGaussianBlur([.1, 2.])),
            enable_if(conf.is_RandomBrightness, RandomBrightness(0.2)),
            enable_if(conf.is_RandomContrast, RandomContrast(0.8, 1.2)),
        ])

    def __call__(self, image, masks=None, fm=None):
        return self.augment(image, masks, fm)

class CGLAugmentator(object):

    def __init__(self, conf):
        self.augment1 = Choice([
            enable_if(conf.is_RandomFlip, RandomFlip()),
            enable_if(conf.is_RandomMirror, RandomMirror()),
            enable_if(conf.is_RandomRot90, RandomRot90()),
            enable_if(conf.is_RandomScale, RandomScale(0.95, 1.05)),
        ])
        self.augment2 = Compose([
            enable_if(conf.is_RandomBrightness, RandomBrightness(0.1)),
            enable_if(conf.is_RandomContrast, RandomContrast(0.2, 1.8)),
            enable_if(conf.is_RandomGaussianBlur, RandomGaussianBlur([.1, 2.])),
        ])
    
    def __call__(self, image, masks=None):
        return self.augment2(*self.augment1(image, masks))

class CAMUSSequenceAugmentator(object):
    """ Transform to be used when training. """

    def __init__(self, conf):
        self.augment = Compose([
            enable_if(conf.is_RandomWarp, RandomVolWarp(0., 10., 4)),
            enable_if(conf.is_RandomFlip, RandomVolFlip()),
            enable_if(conf.is_RandomMirror, RandomVolMirror()),
            enable_if(conf.is_RandomRot90, RandomVolRot90()),
            enable_if(conf.is_RandomBrightness, RandomVolBrightness(0.3)),
            enable_if(conf.is_RandomContrast, RandomVolContrast(0.7, 1.3)),
            enable_if(conf.is_RandomGaussianBlur, RandomVolGaussianBlur([.1, 2.])),
        ])

    def __call__(self, image, masks=None, fm=None):
        return self.augment(image, masks, fm)

class CAMUSImageAugmentator(object):
    """ Transform to be used when training. """

    def __init__(self, conf):
        self.augment = Compose([
            enable_if(conf.is_RandomWarp, RandomWarp(0., 15., 4)),
            enable_if(conf.is_RandomFlip, RandomFlip()),
            enable_if(conf.is_RandomMirror, RandomMirror()),
            enable_if(conf.is_RandomRot90, RandomRot90()),
            enable_if(conf.is_RandomBrightness, RandomBrightness(0.2)),
            enable_if(conf.is_RandomContrast, RandomContrast(0.8, 1.2)),
            enable_if(conf.is_RandomGaussianBlur, RandomGaussianBlur([.1, 2.])),
        ])

    def __call__(self, image, masks=None, fm=None):
        return self.augment(image, masks, fm)

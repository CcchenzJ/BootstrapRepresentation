from .cfg_class import *

null_augmentor = Config({
    'type': None,
})

augmentor = Config({
    'type': None,
    'is_RandomBrightness': False,
    'is_RandomContrast': False,
    'is_RandomGaussianBlur': False,    
    'is_RandomWarp': False,

    'is_RandomFlip': False,
    'is_RandomMirror': False,
    'is_RandomRot90': False,

    'is_RandomCrop': False,
    'is_RandomScale': False,
})

acdc_img_augmentor = augmentor.copy({
    'type': 'ACDCImageAugmentator',
    'is_RandomBrightness': True,
    'is_RandomContrast': True,
    'is_RandomGaussianBlur': True,    
    'is_RandomWarp': True,

    'is_RandomFlip': True,
    'is_RandomMirror': True,
    'is_RandomRot90': True,

})
acdc_vol_augmentor = augmentor.copy({
    'type': 'ACDCVolumeAugmentator',
    'is_RandomBrightness': True,
    'is_RandomContrast': True,
    'is_RandomGaussianBlur': True,    
    'is_RandomWarp': True,

    'is_RandomFlip': True,
    'is_RandomMirror': True,
    'is_RandomRot90': True,
})

pst_img_augmentor = augmentor.copy({
    'type': 'ProstateImageAugmentator',
    'is_RandomBrightness': True,
    'is_RandomContrast': True,
    'is_RandomGaussianBlur': False,    
    'is_RandomWarp': True,

    'is_RandomFlip': True,
    'is_RandomMirror': True,
    'is_RandomRot90': True,

})
pst_vol_augmentor = augmentor.copy({
    'type': 'ProstateVolumeAugmentator',
    'is_RandomBrightness': True,
    'is_RandomContrast': True,
    'is_RandomGaussianBlur': False,   
    'is_RandomWarp': True,

    'is_RandomFlip': True,
    'is_RandomMirror': True,
    'is_RandomRot90': True,

})

mnm_img_augmentor = augmentor.copy({
    'type': 'MnMImageAugmentator',
    'is_RandomBrightness': True,
    'is_RandomContrast': True,
    'is_RandomGaussianBlur': True,   
    'is_RandomWarp': True， 

    'is_RandomFlip': True,
    'is_RandomMirror': True,
    'is_RandomRot90': True,

})
mnm_vol_augmentor = augmentor.copy({
    'type': 'MnMVolumeAugmentator',
    'is_RandomBrightness': True,
    'is_RandomContrast': True,
    'is_RandomGaussianBlur': True,
    'is_RandomWarp': True,  

    'is_RandomFlip': True,
    'is_RandomMirror': True,
    'is_RandomRot90': True,
})

camus_seq_augmentator = augmentor.copy({
    'type': 'CAMUSSequenceAugmentator',
    'is_RandomBrightness': True,
    'is_RandomContrast': True,
    'is_RandomGaussianBlur': False,    
    'is_RandomWarp': True,

    'is_RandomFlip': True,
    'is_RandomMirror': True,
    'is_RandomRot90': True,

})

camus_img_augmentator = augmentor.copy({
    'type': 'CAMUSImageAugmentator',
    'is_RandomBrightness': True,
    'is_RandomContrast': True,
    'is_RandomGaussianBlur': False,    
    'is_RandomWarp': True,

    'is_RandomFlip': True,
    'is_RandomMirror': True,
    'is_RandomRot90': True,
})


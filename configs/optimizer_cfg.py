from .cfg_class import *

sgd = Config({
    'type': torch.optim.SGD,
    'tofreeze_module': None,
    # dw' = momentum * dw - lr * (grad + decay * w)
    'args': {'lr': 1e-2, 'momentum': 0.9, 'weight_decay': 5e-4},
})

adam = Config({
    'type': torch.optim.Adam,
    'tofreeze_module': None,
    'args': {'lr': 1e-3, 'eps': 1e-8, 'betas': (0.9, 0.99), 'weight_decay': 5e-4},
})

adam_freeze_enc = adam.copy({
    # do not update the weights in <tofreeze_module>.
    'tofreeze_module': 'encoder',
})

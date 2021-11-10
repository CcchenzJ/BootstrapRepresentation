
import glob
from .backbone import *
from abc import abstractclassmethod

class ModelTemplate(nn.Module):

    def __init__(self, conf):
        super().__init__()
        self.conf = conf

        if self.conf.freeze_bn:
            self.freeze_bn()

        # Extra parameters for the extra losses
        
    def forward(self, x):
        return x

    def layer_forward(self, layer, x):
        """ Go forward the layer. """
        for scope, layer in getattr(self, layer).items():
            x = layer(x)
        return x

    def save_weights(self, path):
        """ Saves the model's weights. """
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        """ Loads weights. """
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)

    def init_weights(self, init_type='he2'):
        """ Initialize weights for training. """

        init_func = {
            'he1': lambda w: nn.init.kaiming_normal_(w, a=1e-2, nonlinearity='leaky_relu'),
            'he2': lambda w: nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu'),
            'xia': lambda w: nn.init.xavier_uniform_(w, gain=1),
        }

        # Initialize the weights.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_func[init_type](m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Initialize the backbone with the pretrained weights.
        self.load_backbone()

    def load_backbone(self):
        """ Load the weights for the backbone. """
        if self.conf.backbone is not None:
            path = self.conf.backbone['path']
            print('loading from {} ...'.format(path), end='')
            _src_state_dict = torch.load(path)
            _dst_state_dict = self.state_dict()

            keys = [arch for arch in self.conf.backbone['arch']]
            for key, val in _dst_state_dict.items():
                if key.split('.')[1] in keys:
                    _dst_state_dict.update({key: _src_state_dict[key]}) 

            self.load_state_dict(_dst_state_dict)
            
            del _src_state_dict
            del _dst_state_dict
            print('done.')

    def train(self, mode=True):
        super().train(mode)

        if self.conf.freeze_bn:
            self.freeze_bn()

    def freeze_bn(self, enable=False):
        """ Adapted from https://discuss.pytorch.org/t/how-to-train-with-frozen-batchnorm/12106/8 """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train() if enable else module.eval()

                module.weight.requires_grad = enable
                module.bias.requires_grad   = enable

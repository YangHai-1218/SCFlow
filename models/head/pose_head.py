import torch
from torch import nn
from torch.nn import functional as F
from mmcv.runner import BaseModule
from mmcv.cnn import ConvModule

from .builder import HEAD



@HEAD.register_module()
class SingleClassPoseHead(BaseModule):

    _conv_feat_channels = {'Basic':[128, 128, 128], 'Large':[128, 128, 128]}
    _conv_strides = {'Basic':[2, 2, 2], 'Large':[2, 2, 2]}
    _conv_paddings = {'Basic':[1, 1, 1], 'Large':[1, 1, 1]}
    _conv_kernel_sizes = {'Basic':[3, 3, 3], 'Large':[3, 3, 3]}
    _fc_feat_channels = {'Basic':[1024, 256], 'Large':[1024, 256]}


    _image_size = {'Basic':(256, 256), 'Large':(256, 256)}
    _feat_size = {'Basic':(32, 32), 'Large':(64, 64)}
    def __init__(self, 
                in_channels: int,
                net_type: str,
                norm_cfg: dict,
                act_cfg: dict,
                feat_size: tuple=None,
                rotation_mode: str='quaternion',
                init_cfg=None):
        super().__init__(init_cfg)
        assert net_type in ['Basic', 'Small', 'Large']
        conv_feat_channels = self._conv_feat_channels.get(net_type)
        conv_strides = self._conv_strides.get(net_type)
        conv_kernel_sizes = self._conv_kernel_sizes.get(net_type)
        conv_paddings = self._conv_paddings.get(net_type)
        fc_feat_channels = self._fc_feat_channels.get(net_type)
        if feat_size is None:
            feat_size = self._feat_size.get(net_type)
        else:
            assert isinstance(feat_size, (list, tuple))
            assert len(feat_size) == 2
        self.rotation_mode = rotation_mode

        conv_layers = []
        conv_out_size = feat_size[0] * feat_size[1]
        for ch, kernel_size, stride, p in zip(
            conv_feat_channels, conv_kernel_sizes, conv_strides, conv_paddings):
            conv_layers.append(  
                ConvModule(
                    in_channels=in_channels,
                    out_channels=ch,
                    kernel_size=kernel_size,
                    padding=p,
                    stride=stride,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                )
            )
            in_channels = ch
            conv_out_size = int(conv_out_size / (stride**2))
        conv_out_channels = ch

        self.conv_layers = nn.Sequential(*conv_layers)

        fc_in_channels = conv_out_channels * conv_out_size
        fc_layers = []
        for ch in fc_feat_channels:
            fc_layer = [nn.Linear(in_features=fc_in_channels, out_features=ch), nn.ReLU()]
            fc_layers.append(nn.Sequential(*fc_layer))
            fc_in_channels = ch
        fc_out_channels = ch
        self.flatten_op = nn.Flatten(start_dim=1, end_dim=-1)
        self.fc_layers = nn.Sequential(*fc_layers)
        if rotation_mode == 'quaternion':
            self.rotation_out_channels = 4
        elif rotation_mode == 'ortho6d':
            self.rotation_out_channels = 6
        else:
            raise RuntimeError(f"Not supported rotation mode:{rotation_mode}")
        self.rotation_pred = nn.Linear(in_features=fc_out_channels, out_features=self.rotation_out_channels)
        self.translation_pred = nn.Linear(in_features=fc_out_channels, out_features=3)
        self.init_weights()
        
    
    def init_weights(self):
        # zero translation
        nn.init.zeros_(self.translation_pred.weight)
        nn.init.zeros_(self.translation_pred.bias)
        # identity quarention
        nn.init.zeros_(self.rotation_pred.weight)
        with torch.no_grad():
            if self.rotation_mode == 'quaternion':
                self.rotation_pred.bias.copy_(torch.Tensor([0., 0., 0., 1.]))
            elif self.rotation_mode == 'ortho6d':
                self.rotation_pred.bias.copy_(torch.Tensor([1., 0., 0., 0., 1., 0.]))

    def forward(self, x: torch.Tensor, label:torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.flatten_op(x)
        x = self.fc_layers(x)
        pred_translation_delta = self.translation_pred(x)
        pred_rotation_delta = self.rotation_pred(x)
        return pred_rotation_delta, pred_translation_delta





@HEAD.register_module()
class MultiClassPoseHead(BaseModule):

    _conv_feat_channels = {'Basic':[128, 128, 128], 'Large':[128, 128, 128]}
    _conv_strides = {'Basic':[2, 2, 2], 'Large':[2, 2, 2]}
    _conv_paddings = {'Basic':[1, 1, 1], 'Large':[1, 1, 1]}
    _conv_kernel_sizes = {'Basic':[3, 3, 3], 'Large':[3, 3, 3]}
    _fc_feat_channels = {'Basic':[1024, 256], 'Large':[1024, 256]}


    _image_size = {'Basic':(256, 256), 'Large':(256, 256)}
    _feat_size = {'Basic':(32, 32), 'Large':(64, 64)}
    def __init__(self, 
                num_class: int,
                in_channels: int,
                net_type: str,
                norm_cfg:dict,
                act_cfg: dict,
                feat_size: tuple=None,
                rotation_mode: str='quaternion',
                init_cfg=None):
        super().__init__(init_cfg)
        self.num_class = num_class
        assert net_type in ['Basic', 'Small', 'Large']
        conv_feat_channels = self._conv_feat_channels.get(net_type)
        conv_strides = self._conv_strides.get(net_type)
        conv_kernel_sizes = self._conv_kernel_sizes.get(net_type)
        conv_paddings = self._conv_paddings.get(net_type)
        fc_feat_channels = self._fc_feat_channels.get(net_type)
        if feat_size is None:
            feat_size = self._feat_size.get(net_type)
        else:
            assert isinstance(feat_size, (list, tuple))
            assert len(feat_size) == 2
        

        conv_layers = []
        conv_out_size = feat_size[0] * feat_size[1]
        for ch, kernel_size, stride, p in zip(
            conv_feat_channels, conv_kernel_sizes, conv_strides, conv_paddings):
            conv_layers.append(  
                ConvModule(
                    in_channels=in_channels,
                    out_channels=ch,
                    kernel_size=kernel_size,
                    padding=p,
                    stride=stride,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                )
            )
            in_channels = ch
            conv_out_size = int(conv_out_size / (stride**2))
        conv_out_channels = ch

        self.conv_layers = nn.Sequential(*conv_layers)

        fc_in_channels = conv_out_channels * conv_out_size
        fc_layers = []
        for ch in fc_feat_channels:
            fc_layer = [nn.Linear(in_features=fc_in_channels, out_features=ch), nn.ReLU()]
            fc_layers.append(nn.Sequential(*fc_layer))
            fc_in_channels = ch
        fc_out_channels = ch
        self.flatten_op = nn.Flatten(start_dim=1, end_dim=-1)
        self.fc_layers = nn.Sequential(*fc_layers)
        if rotation_mode == 'quaternion':
            self.rotation_out_channels = 4
        elif rotation_mode == 'ortho6d':
            self.rotation_out_channels = 6
        else:
            raise RuntimeError(f"Not supported rotation mode:{rotation_mode}")
        self.rotation_mode = rotation_mode
        self.rotation_pred = nn.Linear(in_features=fc_out_channels, out_features=self.rotation_out_channels*num_class)
        self.translation_pred = nn.Linear(in_features=fc_out_channels, out_features=3*num_class)
        self.init_weights()
    
    def init_weights(self):
        # zero translation
        nn.init.zeros_(self.translation_pred.weight)
        nn.init.zeros_(self.translation_pred.bias)
        nn.init.zeros_(self.rotation_pred.weight)
        # identity quarention
        if self.rotation_mode == 'quaternion':
            with torch.no_grad():
                self.rotation_pred.bias.copy_(torch.Tensor([0., 0., 0., 1.]*self.num_class))
        elif self.rotation_mode == 'ortho6d':
            with torch.no_grad():
                self.rotation_pred.bias.copy_(torch.Tensor([1., 0., 0., 0., 1., 0.]*self.num_class))


    def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.flatten_op(x)
        x = self.fc_layers(x)
        pred_translation_delta = self.translation_pred(x)
        pred_rotation_delta = self.rotation_pred(x)
        pred_translation_delta = pred_translation_delta.view(-1, self.num_class, 3)
        pred_rotation_delta = pred_rotation_delta.view(-1, self.num_class, self.rotation_out_channels)
        pred_translation_delta = torch.index_select(pred_translation_delta, dim=1, index=label)[:, 0, :]
        pred_rotation_delta = torch.index_select(pred_rotation_delta, dim=1, index=label)[:, 0, :]
        return pred_rotation_delta, pred_translation_delta
    

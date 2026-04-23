import torch
import torch.nn as nn
import torch.optim as optim

import os
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt

from ..base_model import BaseModel
from ..modules import SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs, SeparateBNNecks, TemporalSpectralAdapter, AdaptiveHarmonicResonanceAdapter, ComplexHarmonicFilterBankAdapter, PeriodicTemporalStateAdapter, TemporalQualityGateAdapter, LaStGaitAdapter, LaStTemporalPooling, conv1x1, conv3x3, BasicBlock2D, BasicBlockP3D, BasicBlock3D
from utils import get_valid_args, get_attr_from

from einops import rearrange

blocks_map = {
    '2d': BasicBlock2D, 
    'p3d': BasicBlockP3D, 
    '3d': BasicBlock3D
}

class DeepGaitV2(BaseModel):

    def build_network(self, model_cfg):
        mode = model_cfg['Backbone']['mode']
        assert mode in blocks_map.keys()
        block = blocks_map[mode]

        in_channels = model_cfg['Backbone']['in_channels']
        layers      = model_cfg['Backbone']['layers']
        channels    = model_cfg['Backbone']['channels']
        self.inference_use_emb2 = model_cfg['use_emb2'] if 'use_emb2' in model_cfg else False

        if mode == '3d': 
            strides = [
                [1, 1], 
                [1, 2, 2], 
                [1, 2, 2], 
                [1, 1, 1]
            ]
        else: 
            strides = [
                [1, 1], 
                [2, 2], 
                [2, 2], 
                [1, 1]
            ]

        self.inplanes = channels[0]
        self.layer0 = SetBlockWrapper(nn.Sequential(
            conv3x3(in_channels, self.inplanes, 1), 
            nn.BatchNorm2d(self.inplanes), 
            nn.ReLU(inplace=True)
        ))
        self.layer1 = SetBlockWrapper(self.make_layer(BasicBlock2D, channels[0], strides[0], blocks_num=layers[0], mode=mode))

        self.layer2 = self.make_layer(block, channels[1], strides[1], blocks_num=layers[1], mode=mode)
        self.layer3 = self.make_layer(block, channels[2], strides[2], blocks_num=layers[2], mode=mode)
        self.layer4 = self.make_layer(block, channels[3], strides[3], blocks_num=layers[3], mode=mode)

        if mode == '2d': 
            self.layer2 = SetBlockWrapper(self.layer2)
            self.layer3 = SetBlockWrapper(self.layer3)
            self.layer4 = SetBlockWrapper(self.layer4)

        self.FCs = SeparateFCs(16, channels[3], channels[2])
        self.BNNecks = SeparateBNNecks(16, channels[2], class_num=model_cfg['SeparateBNNecks']['class_num'])

        pooling_cfg = dict(model_cfg.get('TemporalPooling', {}))
        pooling_enable = pooling_cfg.pop('enable', False)
        pooling_type = pooling_cfg.pop('type', 'max')
        if pooling_enable and pooling_type in ['last', 'last_temporal', 'frequency_stable']:
            self.TP = LaStTemporalPooling(**pooling_cfg)
        else:
            self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=[16])
        adapter_cfg = model_cfg.get('TemporalSpectralAdapter', {})
        if adapter_cfg.get('enable', False):
            adapter_cfg = dict(adapter_cfg)
            adapter_cfg.pop('enable')
            adapter_type = adapter_cfg.pop('adapter_type', 'branch_attention_residual')
            if adapter_type == 'adaptive_harmonic_resonance':
                self.temporal_adapter = AdaptiveHarmonicResonanceAdapter(
                    channels=channels[3],
                    **adapter_cfg,
                )
            elif adapter_type in ['complex_harmonic_filter_bank', 'chfb']:
                self.temporal_adapter = ComplexHarmonicFilterBankAdapter(
                    channels=channels[3],
                    **adapter_cfg,
                )
            elif adapter_type in ['periodic_temporal_state', 'ptsa']:
                self.temporal_adapter = PeriodicTemporalStateAdapter(
                    channels=channels[3],
                    **adapter_cfg,
                )
            elif adapter_type in ['temporal_quality_gate', 'tqg']:
                self.temporal_adapter = TemporalQualityGateAdapter(
                    channels=channels[3],
                    **adapter_cfg,
                )
            elif adapter_type in ['last_gait', 'last', 'lazystrike']:
                self.temporal_adapter = LaStGaitAdapter(
                    channels=channels[3],
                    **adapter_cfg,
                )
            else:
                self.temporal_adapter = TemporalSpectralAdapter(
                    channels=channels[3],
                    **adapter_cfg,
                )
        else:
            self.temporal_adapter = None
        self.finetune_cfg = model_cfg.get('Finetune', {})
        self.configure_finetune()

    def configure_finetune(self):
        mode = self.finetune_cfg.get('mode', 'full')
        extra_trainable_modules = self.finetune_cfg.get('extra_trainable_modules', [])
        if mode in ['full', 'full_finetune']:
            for param in self.parameters():
                param.requires_grad = True
        elif mode == 'adapter_head':
            for param in self.parameters():
                param.requires_grad = False
            if self.temporal_adapter is not None:
                for param in self.temporal_adapter.parameters():
                    param.requires_grad = True
            for param in self.TP.parameters():
                param.requires_grad = True
            for module in [self.FCs, self.BNNecks]:
                for param in module.parameters():
                    param.requires_grad = True
        else:
            raise ValueError("Unsupported Finetune.mode: {}".format(mode))

        for module_name in extra_trainable_modules:
            module = getattr(self, module_name, None)
            if module is None:
                raise ValueError("Unknown module in Finetune.extra_trainable_modules: {}".format(module_name))
            for param in module.parameters():
                param.requires_grad = True

    def get_optimizer(self, optimizer_cfg):
        self.msg_mgr.log_info(optimizer_cfg)
        optimizer_cls = get_attr_from([optim], optimizer_cfg['solver'])
        valid_arg = get_valid_args(optimizer_cls, optimizer_cfg, ['solver'])

        finetune_cfg = self.finetune_cfg if hasattr(self, 'finetune_cfg') else {}
        if not finetune_cfg:
            return optimizer_cls(
                filter(lambda p: p.requires_grad, self.parameters()), **valid_arg)

        backbone_lr = optimizer_cfg.get('backbone_lr', optimizer_cfg['lr'])
        head_lr = optimizer_cfg.get('head_lr', optimizer_cfg['lr'])
        adapter_lr = optimizer_cfg.get('adapter_lr', optimizer_cfg['lr'])
        weight_decay = optimizer_cfg.get('weight_decay', 0.0)
        mode = finetune_cfg.get('mode', 'full')

        param_groups = []
        backbone_modules = [self.layer0, self.layer1, self.layer2, self.layer3, self.layer4]
        backbone_params = [p for module in backbone_modules for p in module.parameters() if p.requires_grad]
        head_params = [p for module in [self.FCs, self.BNNecks] for p in module.parameters() if p.requires_grad]
        adapter_params = list(self.temporal_adapter.parameters()) if self.temporal_adapter is not None else []
        adapter_params += list(self.TP.parameters())
        adapter_params = [p for p in adapter_params if p.requires_grad]

        if mode in ['full', 'full_finetune'] and backbone_params:
            param_groups.append({'params': backbone_params, 'lr': backbone_lr, 'weight_decay': weight_decay})
        if head_params:
            param_groups.append({'params': head_params, 'lr': head_lr, 'weight_decay': weight_decay})
        if adapter_params:
            param_groups.append({'params': adapter_params, 'lr': adapter_lr, 'weight_decay': weight_decay})

        if not param_groups:
            raise ValueError("No trainable parameters found for DeepGaitV2 finetuning.")
        return optimizer_cls(param_groups, **valid_arg)

    def make_layer(self, block, planes, stride, blocks_num, mode='2d'):

        if max(stride) > 1 or self.inplanes != planes * block.expansion:
            if mode == '3d':
                downsample = nn.Sequential(nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=[1, 1, 1], stride=stride, padding=[0, 0, 0], bias=False), nn.BatchNorm3d(planes * block.expansion))
            elif mode == '2d':
                downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride=stride), nn.BatchNorm2d(planes * block.expansion))
            elif mode == 'p3d':
                downsample = nn.Sequential(nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=[1, 1, 1], stride=[1, *stride], padding=[0, 0, 0], bias=False), nn.BatchNorm3d(planes * block.expansion))
            else:
                raise TypeError('xxx')
        else:
            downsample = lambda x: x

        layers = [block(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes * block.expansion
        s = [1, 1] if mode in ['2d', 'p3d'] else [1, 1, 1]
        for i in range(1, blocks_num):
            layers.append(
                    block(self.inplanes, planes, stride=s)
            )
        return nn.Sequential(*layers)

    def forward(self, inputs):
        ipts, labs, typs, vies, seqL = inputs
        
        if len(ipts[0].size()) == 4:
            sils = ipts[0].unsqueeze(1)
        else:
            sils = ipts[0]
            sils = sils.transpose(1, 2).contiguous()
        assert sils.size(-1) in [44, 88]

        del ipts
        out0 = self.layer0(sils)
        out1 = self.layer1(out0)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3) # [n, c, s, h, w]
        if self.temporal_adapter is not None:
            out4 = self.temporal_adapter(out4)

        # Temporal Pooling, TP
        outs = self.TP(out4, seqL, options={"dim": 2})[0]  # [n, c, h, w]

        # Horizontal Pooling Matching, HPM
        feat = self.HPP(outs)  # [n, c, p]

        embed_1 = self.FCs(feat)  # [n, c, p]
        embed_2, logits = self.BNNecks(embed_1)  # [n, c, p]

        if self.inference_use_emb2:
                embed = embed_2
        else:
                embed = embed_1

        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed_1, 'labels': labs},
                'softmax': {'logits': logits, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': rearrange(sils, 'n c s h w -> (n s) c h w'),
            },
            'inference_feat': {
                'embeddings': embed
            }
        }

        return retval

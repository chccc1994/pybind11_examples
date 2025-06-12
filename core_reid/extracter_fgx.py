# feature_extractor.py

import os
import math
import yaml
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable

from model import (
    ft_net, ft_net_dense, ft_net_NAS, PCB, PCB_test,
    ft_net_swin, ft_net_swinv2, ft_net_convnext, ft_net_efficient, ft_net_hr
)
from torch.nn.utils import fuse_conv_bn_eval

def fuse_all_conv_bn(model):
    stack = []
    for name, module in model.named_children():
        if list(module.named_children()):
            fuse_all_conv_bn(module)
            
        if isinstance(module, nn.BatchNorm2d):
            if not stack:
                continue
            if isinstance(stack[-1][1], nn.Conv2d):
                setattr(model, stack[-1][0], fuse_conv_bn_eval(stack[-1][1], module))
                setattr(model, name, nn.Identity())
        else:
            stack.append((name, module))
    return model


class FeatureExtractor:
    def __init__(self, config_path, device='cuda:0', which_epoch='last', ms='1'):
        # 显式传入 torch.device
        self.device = torch.device(device if torch.cuda.is_available() or device == 'cpu' else 'cpu')

        # 加载配置文件
        with open(config_path, 'r') as stream:
            config = yaml.load(stream, Loader=yaml.FullLoader)
        print(config)
        self.opt = type('Opt', (), config)()
        self.opt.which_epoch = which_epoch
        self.opt.ms = ms

        self.ms = [math.sqrt(float(s)) for s in self.opt.ms.split(',')]

        # 构建模型并加载权重
        model = self.build_model()
        model.eval()
        model = fuse_all_conv_bn(model)
        self.model = model.to(self.device)

        # 图像尺寸和预处理器
        if getattr(self.opt, 'use_swin', False):
            self.h, self.w = 224, 224
        else:
            self.h, self.w = 256, 128

        if getattr(self.opt, 'PCB', False):
            self.h, self.w = 384, 192

        self.data_transforms = transforms.Compose([
            transforms.Resize((self.h, self.w), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def build_model(self):
        opt = self.opt
        if getattr(opt, 'use_dense', False):
            model = ft_net_dense(opt.nclasses, stride=opt.stride, linear_num=opt.linear_num)
        elif getattr(opt, 'use_NAS', False):
            model = ft_net_NAS(opt.nclasses, linear_num=opt.linear_num)
        elif getattr(opt, 'use_swin', False):
            model = ft_net_swin(opt.nclasses, linear_num=opt.linear_num)
        elif getattr(opt, 'use_swinv2', False):
            model = ft_net_swinv2(opt.nclasses, (opt.h, opt.w), linear_num=opt.linear_num)
        elif getattr(opt, 'use_convnext', False):
            model = ft_net_convnext(opt.nclasses, linear_num=opt.linear_num)
        elif getattr(opt, 'use_efficient', False):
            model = ft_net_efficient(opt.nclasses, linear_num=opt.linear_num)
        elif getattr(opt, 'use_hr', False):
            model = ft_net_hr(opt.nclasses, linear_num=opt.linear_num)
        else:
            model = ft_net(opt.nclasses, stride=opt.stride, ibn=opt.ibn, linear_num=opt.linear_num)

        if getattr(opt, 'PCB', False):
            model = PCB(opt.nclasses)

        model = self.load_network(model)

        if getattr(opt, 'PCB', False):
            model = PCB_test(model)
        else:
            model.classifier.classifier = nn.Sequential()

        model.eval()                      # 主模型设置为 eval
        for m in model.modules():        # 所有子模块也设置为 eval
            m.eval()
            
        return fuse_all_conv_bn(model)

    def load_network(self, model):
        save_path = os.path.join('./model', self.opt.name, f'net_{self.opt.which_epoch}.pth')
        state_dict = torch.load(save_path, map_location=self.device)
        model.load_state_dict(state_dict)
        return model

    def fliplr(self, img):
        return img.flip(dims=[3])  # 水平翻转

    def extract_feature(self, img_pil: Image.Image):
        opt = self.opt
        linear_dim = opt.linear_num if getattr(opt, 'linear_num', 0) > 0 else 2048
        if getattr(opt, 'PCB', False):
            linear_dim = 2048 * 6

        img = self.data_transforms(img_pil).unsqueeze(0).to(self.device)
        ff = torch.zeros((1, linear_dim), device=self.device)

        if getattr(opt, 'PCB', False):
            ff = torch.zeros((1, 2048, 6), device=self.device)

        for i in range(2):
            if i == 1:
                img = self.fliplr(img)

            for scale in self.ms:
                scaled_img = nn.functional.interpolate(img, scale_factor=scale, mode='bicubic', align_corners=False) if scale != 1 else img
                with torch.no_grad():
                    output = self.model(scaled_img)
                ff += output

        if getattr(opt, 'PCB', False):
            ff = ff.view(ff.size(0), -1)
            ff = ff / (torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6))
        else:
            ff = ff / torch.norm(ff, p=2, dim=1, keepdim=True)

        return ff.cpu().numpy().flatten()

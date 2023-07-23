import torchvision.transforms.functional as TF
import torch.nn as nn
import random
import numpy as np
import torch
from perlin2d import generate_perlin_noise_2d
random.seed(0)


class GeometricTransform:
    """ Enables Geometric transforms on img and mask simultaneously.

    We employ reflection padding, since it ensures that the roads are not cutoff, and it"""

    def __init__(self):
        # sample ranges
        self.max_angle = 180
        self.max_trans_x = 100
        self.max_trans_y = 100
        self.min_scale = 0.95
        self.max_scale = 1.05
        self.max_shear = 0
        self.hfilp_prob = 0.5
        self.vfilp_prob = 0.5
        # state
        self.angle = 0
        self.translate = [0, 0]
        self.scale = 1
        self.shear = 0
        self.hfilp = False
        self.vfilp = False
        self.output_size = 400

        self.pad = nn.ReflectionPad2d(300)

    def __call__(self, x):
        if self.vfilp:
            x = TF.vflip(x)
        if self.hfilp:
            x = TF.hflip(x)
        x = self.pad(x.to(torch.float))
        x = TF.affine(x, self.angle, self.translate, self.scale, self.shear, interpolation=TF.InterpolationMode.BILINEAR)
        x = TF.center_crop(x, self.output_size)
        return x

    def backward(self, x):
        x = TF.center_crop(x, 400)
        # invert the affine warp
        theta = np.deg2rad(self.angle)
        c, s = np.cos(theta), np.sin(theta)
        rot_mat = np.array(((c, s), (-s, c)))
        translate_inv = list(- rot_mat @ np.array(self.translate)/self.scale)

        x = TF.affine(x, -self.angle, list(translate_inv), 1./self.scale, -self.shear, interpolation=TF.InterpolationMode.BILINEAR)
        if self.hfilp:
            x = TF.hflip(x)
        if self.vfilp:
            x = TF.vflip(x)
        return x

    def sample_params(self):
        self.hfilp = random.random() < self.hfilp_prob
        self.vfilp = random.random() < self.vfilp_prob
        self.angle = random.uniform(-self.max_angle, self.max_angle)
        self.translate = [random.uniform(-self.max_trans_x, self.max_trans_x), random.uniform(-self.max_trans_y, self.max_trans_y)]
        self.scale = random.uniform(self.min_scale, self.max_scale)
        self.shear = random.random()*self.max_shear

    def zero_params(self):
        # state
        self.angle = 0
        self.translate = [0, 0]
        self.scale = 1
        self.shear = 0
        self.hfilp = False
        self.vfilp = False


import matplotlib.pyplot as plt

class AddPerlinNoise:
    def __init__(self, res=8,h=0.1,s=0.1,v=10):
        self.res = res
        self.h = h
        self.s = s
        self.v = v

    def __call__(self, tensor):
        h, w = tensor.shape[-2:]
        ph, pw = map(lambda h: int(np.ceil(h / self.res)) * self.res, [h, w])

        noise = np.array([generate_perlin_noise_2d((ph, pw),(self.res, self.res)) for i in range(3)])
        #noise += np.array([generate_perlin_noise_2d((ph, pw), (2*self.res, 2*self.res)) for i in range(3)])
        noise = noise[:, :h, :w]
        noise -= np.mean(noise)
        noise[0] = self.h * noise[0]
        noise[1] = self.s * noise[1]
        noise[2] = self.v * noise[2]

        color_noise = torch.tensor(noise, dtype=float)

        tensor = rgb2hsv_torch(tensor.unsqueeze(0).to(torch.float))
        tensor = tensor.to(float) + color_noise


        tensor = hsv2rgb_torch(tensor)
        tensor = tensor.squeeze(0)
        tensor = torch.clip(tensor, 0, 255)
        return tensor.to(torch.uint8)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)




def rgb2hsv_torch(rgb: torch.Tensor) -> torch.Tensor:
    """https://github.com/limacv/RGB_HSV_HSL/blob/master/color_torch.py"""
    cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
    cmin = torch.min(rgb, dim=1, keepdim=True)[0]
    delta = cmax - cmin
    hsv_h = torch.empty_like(rgb[:, 0:1, :, :])
    cmax_idx[delta == 0] = 3
    hsv_h[cmax_idx == 0] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6)[cmax_idx == 0]
    hsv_h[cmax_idx == 1] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2)[cmax_idx == 1]
    hsv_h[cmax_idx == 2] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)[cmax_idx == 2]
    hsv_h[cmax_idx == 3] = 0.
    hsv_h /= 6.
    hsv_s = torch.where(cmax == 0, torch.tensor(0.).type_as(rgb), delta / cmax)
    hsv_v = cmax
    return torch.cat([hsv_h, hsv_s, hsv_v], dim=1)


def hsv2rgb_torch(hsv: torch.Tensor) -> torch.Tensor:
    """https://github.com/limacv/RGB_HSV_HSL/blob/master/color_torch.py"""
    hsv_h, hsv_s, hsv_l = hsv[:, 0:1], hsv[:, 1:2], hsv[:, 2:3]
    _c = hsv_l * hsv_s
    _x = _c * (- torch.abs(hsv_h * 6. % 2. - 1) + 1.)
    _m = hsv_l - _c
    _o = torch.zeros_like(_c)
    idx = (hsv_h * 6.).type(torch.uint8)
    idx = (idx % 6).expand(-1, 3, -1, -1)
    rgb = torch.empty_like(hsv)
    rgb[idx == 0] = torch.cat([_c, _x, _o], dim=1)[idx == 0]
    rgb[idx == 1] = torch.cat([_x, _c, _o], dim=1)[idx == 1]
    rgb[idx == 2] = torch.cat([_o, _c, _x], dim=1)[idx == 2]
    rgb[idx == 3] = torch.cat([_o, _x, _c], dim=1)[idx == 3]
    rgb[idx == 4] = torch.cat([_x, _o, _c], dim=1)[idx == 4]
    rgb[idx == 5] = torch.cat([_c, _o, _x], dim=1)[idx == 5]
    rgb += _m
    return rgb

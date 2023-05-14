import torchvision.transforms.functional as TF
import torch.nn as nn
import random
import numpy as np
import torch
random.seed(0)


class GeometricTransform:
    """ Enables Geometric transforms on img and mask simultaneously.

    We employ reflection padding, since it ensures that the roads are not cutoff, and it"""

    def __init__(self):
        # sample ranges
        self.max_angle = 45
        self.max_trans_x = 100
        self.max_trans_y = 100
        self.min_scale = 0.95
        self.max_scale = 1.05
        self.max_shear = 0

        # state
        self.angle = 0
        self.translate = [0, 0]
        self.scale = 1
        self.shear = 0
        self.hfilp = False
        self.vfilp = False
        self.output_size = 400

        self.pad = nn.ReflectionPad2d(200)

    def __call__(self, x):
        if self.vfilp:
            x = TF.vflip(x)
        if self.hfilp:
            x = TF.hflip(x)
        x = self.pad(torch.tensor(x, dtype=float))
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
        self.hfilp = random.random() > 0.5
        self.vfilp = random.random() > 0.5
        self.angle = random.uniform(-self.max_angle, self.max_angle) + random.choice([0, 90])
        self.translate = [random.random() * self.max_trans_x, random.random() * self.max_trans_y]
        self.scale = random.uniform(self.min_scale, self.max_scale)
        self.shear = random.random()*self.max_shear

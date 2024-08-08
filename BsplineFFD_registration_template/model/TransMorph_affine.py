'''
TransMorph model
Junyu Chen
jchen245@jhmi.edu
Johns Hopkins University
'''

import torch
import torch.nn as nn
import torch.nn.functional as nnf
import numpy as np
import model.configs_affine as configs

class AffineTransform(nn.Module):
    """
    2-D Affine Transformer with shear
    """

    def __init__(self, config):
        super().__init__()
        self.mode = config.mode

    def apply_affine(self, src, mat):
        grid = nnf.affine_grid(mat, src.size(), align_corners=False)
        return nnf.grid_sample(src, grid, align_corners=False, mode=self.mode)

    def forward(self, src, affine, scale, translate, shear):
        theta_x = affine[:, 0]
        theta_y = affine[:, 1]
        scale_x = scale[:, 0]
        scale_y = scale[:, 1]
        trans_x = translate[:, 0]
        trans_y = translate[:, 1]
        shear_x = shear[:, 0]
        shear_y = shear[:, 1]
        '''
        rot_mat_x = torch.stack([torch.cos(theta_x), -torch.sin(theta_x), torch.zeros_like(theta_x),
                                        torch.sin(theta_x), torch.cos(theta_x), torch.zeros_like(theta_x),
                                        torch.zeros_like(theta_x), torch.zeros_like(theta_x), torch.ones_like(theta_x)], dim=1).view(-1, 3, 3)
                rot_mat_y = torch.stack([torch.cos(theta_y), torch.sin(theta_y), torch.zeros_like(theta_y),
                                        -torch.sin(theta_y), torch.cos(theta_y), torch.zeros_like(theta_y),
                                        torch.zeros_like(theta_y), torch.zeros_like(theta_y), torch.ones_like(theta_y)], dim=1).view(-1, 3, 3)
                scale_mat = torch.stack(
                    [torch.stack([scale_x, torch.zeros_like(scale_x), torch.zeros_like(scale_x)], dim=1),
                    torch.stack([torch.zeros_like(scale_y), scale_y, torch.zeros_like(scale_y)], dim=1),
                    torch.stack([torch.zeros_like(scale_y), torch.zeros_like(scale_y), torch.zeros_like(scale_y)], dim=1)], dim=1)
                shear_mat = torch.stack(
                    [torch.stack([torch.ones_like(shear_x), shear_x, torch.zeros_like(shear_x)], dim=1),
                    torch.stack([shear_y, torch.ones_like(shear_y), torch.zeros_like(shear_y)], dim=1),
                    torch.stack([torch.zeros_like(shear_y), torch.zeros_like(shear_y), torch.ones_like(shear_y)], dim=1)], dim=1)
                trans = torch.stack([trans_x, trans_y, torch.zeros_like(trans_y)], dim=1).unsqueeze(dim=2)
                print(shear_mat.shape)
        '''
        rot_mat_x = torch.stack([torch.cos(theta_x), -torch.sin(theta_x), torch.zeros_like(theta_x),
                                 torch.sin(theta_x), torch.cos(theta_x), torch.zeros_like(theta_x),
                                 torch.zeros_like(theta_x), torch.zeros_like(theta_x), torch.ones_like(theta_x)], dim=1).view(-1, 3, 3)
        rot_mat_y = torch.stack([torch.cos(theta_y), torch.sin(theta_y), torch.zeros_like(theta_y),
                                 -torch.sin(theta_y), torch.cos(theta_y), torch.zeros_like(theta_y),
                                 torch.zeros_like(theta_y), torch.zeros_like(theta_y), torch.ones_like(theta_y)], dim=1).view(-1, 3, 3)
        scale_mat = torch.stack(
            [torch.stack([scale_x, torch.zeros_like(scale_x), torch.zeros_like(scale_x)], dim=1),
             torch.stack([torch.zeros_like(scale_y), scale_y, torch.zeros_like(scale_y)], dim=1),
             torch.stack([torch.zeros_like(scale_y), torch.zeros_like(scale_y), torch.zeros_like(scale_y)], dim=1)], dim=1)
        shear_mat = torch.stack(
            [torch.stack([torch.ones_like(shear_x), shear_x, torch.zeros_like(shear_x)], dim=1),
             torch.stack([shear_y, torch.ones_like(shear_y), torch.zeros_like(shear_y)], dim=1),
             torch.stack([torch.zeros_like(shear_y), torch.zeros_like(shear_y), torch.ones_like(shear_y)], dim=1)], dim=1)
        trans = torch.stack([trans_x, trans_y, torch.zeros_like(trans_y)], dim=1).unsqueeze(dim=2)

        
        mat = torch.matmul(shear_mat, torch.matmul(scale_mat, torch.matmul(rot_mat_y, rot_mat_x)))
        mat = torch.cat([mat, trans], dim=-1)
        mat = mat[:, :2, :3] # 对2维图像只取前2*3
        grid = nnf.affine_grid(mat, src.size(), align_corners=False)
        return nnf.grid_sample(src, grid, align_corners=False, mode=self.mode), grid.permute(0, 3, 1, 2)




class AffineNet(nn.Module):
    def __init__(self, config):
        super(AffineNet, self).__init__()
        embed_dim = config.embed_dim
        self.img_size = config.img_size
        
        # Feature extraction layers
        self.features = nn.Sequential(
            nn.Conv2d(config.in_chans, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Affine MLP
        self.aff_mlp = nn.Sequential(
            nn.Linear(128 * (config.img_size[0] // 4) * (config.img_size[1] // 4), 100),
            nn.LeakyReLU(),
            nn.Linear(100, 3)
        )

        # Scale MLP
        self.scl_mlp = nn.Sequential(
            nn.Linear(128 * (config.img_size[0] // 4) * (config.img_size[1] // 4), 100),
            nn.LeakyReLU(),
            nn.Linear(100, 3)
        )

        # Translation MLP
        self.trans_mlp = nn.Sequential(
            nn.Linear(128 * (config.img_size[0] // 4) * (config.img_size[1] // 4), 100),
            nn.LeakyReLU(),
            nn.Linear(100, 3)
        )

        # Shear MLP
        self.shear_mlp = nn.Sequential(
            nn.Linear(128 * (config.img_size[0] // 4) * (config.img_size[1] // 4), 100),
            nn.LeakyReLU(),
            nn.Linear(100, 6)
        )

        self.inst_norm = nn.InstanceNorm2d(embed_dim * 8)

    def softplus(self, x):  # Softplus
        return torch.log(1 + torch.exp(x))

    def forward(self, mov, fix):
        
        x_cat = torch.cat((mov, fix), dim=1)
        
        # Feature extraction
        features_out = self.features(x_cat)
        x5 = torch.flatten(features_out, start_dim=1)

        # Affine transformation MLP
        aff = self.aff_mlp(x5) * 0.1
        
        # Scale MLP
        scl = self.scl_mlp(x5) * 0.1
        
        # Translation MLP
        trans = self.trans_mlp(x5) * 0.1
        
        # Shear MLP
        shr = self.shear_mlp(x5) * 0.1

        # Clamp and scale the outputs
        aff = torch.clamp(aff, min=-1, max=1) * np.pi
        scl = scl + 1
        scl = torch.clamp(scl, min=0, max=5)

        return aff, scl, trans, shr


CONFIGS = {
    'AffineParam': configs.get_aff_config(),
}
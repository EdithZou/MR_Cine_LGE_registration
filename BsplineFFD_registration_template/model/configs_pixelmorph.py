import ml_collections
'''
********************************************************
                   Swing Transformer
********************************************************
if_transskip (bool): Enable skip connections from Transformer Blocks
if_convskip (bool): Enable skip connections from Convolutional Blocks
patch_size (int | tuple(int)): Patch size. Default: 4
in_chans (int): Number of input image channels. Default: 2 (for moving and fixed images)
embed_dim (int): Patch embedding dimension. Default: 96
depths (tuple(int)): Depth of each Swin Transformer layer.
num_heads (tuple(int)): Number of attention heads in different layers.
window_size (tuple(int)): Image size should be divisible by window size, 
                     e.g., if image has a size of (160, 192, 224), then the window size can be (5, 6, 7)
mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
pat_merg_rf (int): Embed_dim reduction factor in patch merging, e.g., N*C->N/4*C if set to four. Default: 4. 
qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
drop_rate (float): Dropout rate. Default: 0
drop_path_rate (float): Stochastic depth rate. Default: 0.1
ape (bool): Enable learnable position embedding. Default: False
spe (bool): Enable sinusoidal position embedding. Default: False
patch_norm (bool): If True, add normalization after patch embedding. Default: True
use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False 
                       (Carried over from Swin Transformer, it is not needed)
out_indices (tuple(int)): Indices of Transformer blocks to output features. Default: (0, 1, 2, 3)
img_size (int | tuple(int)): Input image size, e.g., (160, 192, 224)
-------------------------- Unique Parameters to TransMorph-bspl --------------------------
cps (tuple(int)): Control point spacing for B-spline lattice. Default: (3, 3, 3)
resize_channels (tuple(int)): Channel number for the last two convolutional layers. Default: (32, 32)
'''
def default_unet_features():
    nb_features = [
        [16, 32, 32, 32],             # encoder
        [32, 32, 32, 32, 32, 16, 16]  # decoder
    ]
    return nb_features

def get_pm_config():
    '''
    TransMorph-bspl
    '''
    config = ml_collections.ConfigDict()
    config.img_size = (256, 256)
    config.cps = (4, 4)
    config.enc_channels = (128, 32, 32, 32, 32)
    config.dec_channels = (32, 32, 32, 32)
    config.resize_channels = (2, 2)
    config.mode = 'bilinear'
    config.inshape = 4
    config.nb_unet_features=None
    config.nb_unet_levels=None
    config.unet_feat_mult=1,
    config.nb_unet_conv_per_level=1
    config.int_steps=7
    config.int_downsize=2
    config.bidir=False
    config.use_probs=False
    config.src_feats=1
    config.trg_feats=1
    config.infeats = 2
    config.unet_half_res=False
    return config

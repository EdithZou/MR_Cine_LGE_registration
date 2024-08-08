import ml_collections

def get_aff_config():
    '''
    TransMorph-affine
    '''
    config = ml_collections.ConfigDict()
    config.img_size = (128, 128)
    config.in_chans = 2
    config.embed_dim = 128
    config.mode = 'bilinear'
    return config
from .resnet_fpn import ResNetFPN_8_2, ResNetFPN_16_4
from .resnet_e2 import E2_ResNetFPN_8_2
from .resnet_sesn import SESN_ResNetFPN_8_2
from .resnet_sesn2 import WideResNet

def build_backbone(config):
    if config['backbone_type'] == 'ResNetFPN':
        if config['resolution'] == (8, 2):
            return ResNetFPN_8_2(config['resnetfpn'])
        elif config['resolution'] == (16, 4):
            return ResNetFPN_16_4(config['resnetfpn'])
    elif config['backbone_type'] == 'E2ResNetFPN':
        if config['resolution'] == (8, 2):
            return E2_ResNetFPN_8_2(config['resnetfpn'])
        else:
            raise ValueError(f"LOFTR.RESOLUTION {config['resolution']} not supported with E2ResNetFPN backbone type.")
   
    elif config['backbone_type'] == 'SESNResNetFPN':
        if config['resolution'] == (8, 2):
            return SESN_ResNetFPN_8_2(config['resnetfpn'])
        else:
            raise ValueError(f"LOFTR.RESOLUTION {config['resolution']} not supported with SESNResNetFPN backbone type.")
    elif config['backbone_type'] == 'SESNResNetFPN1':
        if config['resolution'] == (8, 2):
            return WideResNet(config['resnetfpn'])
        else:
            raise ValueError(f"LOFTR.RESOLUTION {config['resolution']} not supported with SESNResNetFPN backbone type.")
    else:
        
        raise ValueError(f"LOFTR.BACKBONE_TYPE {config['backbone_type']} not supported.")

import argparse
import torch
from mmcv import Config
from protogcn.models import build_model
from protogcn.smp import *

"""
Notes:

1. Change 'protogcn/models/recognizers/recognizergcn.py #L78-L84 to:
    ### Compute Model Complexity ###
    # return self.forward_train(keypoint, label=torch.tensor([0]).to(keypoint.device), **kwargs)
    
2. Run `CUDA_VISIBLE_DEVICES=0 python tools/model_complexity.py configs/ntu60_xsub/j.py`

"""

def parse_args():
    parser = argparse.ArgumentParser(description='Compute model complexity')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--input-shape', type=str, default='1,1,2,100,25,3', 
                       help='input shape in format: 1,1,M,T,V,C')
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    
    model = build_model(cfg.model)
    model.eval()
    model.cuda()
    
    input_shape = tuple(map(int, args.input_shape.split(',')))
    dummy_input = torch.randn(input_shape).cuda()
    
    params, flops = fnp(model, dummy_input)

if __name__ == '__main__':
    main()

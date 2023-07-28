from __future__ import print_function, division
from UCDCN_models import *
from UCDCN_options import args
import torch
from thop import profile
import time
import numpy as np
from tqdm import tqdm


__Author__ = 'Quanhao Guo'


# train the model
def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    network = NestedUNet(3, 1, args)
    network.to(device)
    input = torch.randn(1, 3, 128, 128).cuda()
    t_all = []
    for i in tqdm(range(1000)):
        t1 = time.time()
        network(input)
        t2 = time.time()
        t_all.append(t2 - t1)
        if i==0:
            break
    network
    flops, params = profile(network, inputs=(input, ))
    print('average time:', np.mean(t_all) / 1)
    print('average fps:',1 / np.mean(t_all))

    print('fastest time:', min(t_all) / 1)
    print('fastest fps:',1 / min(t_all))

    print('slowest time:', max(t_all) / 1)
    print('slowest fps:',1 / max(t_all))
    print("------- params: %.2fMB ------- flops: %.2fG" % (params / (1000 ** 2), flops / (1000 ** 3))) 

if __name__ == '__main__':
    main(args)

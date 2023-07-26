import argparse


__Author__ = 'Quanhao Guo'

# 'Experiments/20220413-15.20/models/checkpoint/ckpt_64000.pth'
parser = argparse.ArgumentParser(description='UCDCN Traing Options')

parser.add_argument('-t', '--train_path', type=str, default='../Protocol_2/Train_files')
parser.add_argument('-v', '--val_path', type=str, default='../Protocol_2/Test_files')
parser.add_argument('-c', '--checkpoint', type=str, default=None, help='checkpoint path')
parser.add_argument('-r', '--resume', type=int, default=0)
parser.add_argument('-s', '--seed', type=int, default=7777)
parser.add_argument('-p', '--probs', type=float, default=0.9, help='Normalize std for image')
parser.add_argument('-o', '--pretrained', type=str, default='./best.pth', help='Pretrained Backbone')
parser.add_argument('-b', '--backbone', type=bool, default=False, help='Wether training backbone')

parser.add_argument('--val-freq', type=int, default=1000, help='Validation freq when training') # classfier 500, default 1000

parser.add_argument('--SCHEDULER', type=str, default='poly')
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--learning-rate', type=float, default=0.0003) # classfier 0.01, default 0.0001
parser.add_argument('--MOMENTUM', type=float, default=0.9)
parser.add_argument('--EPSILON', type=float, default=1e-8)
parser.add_argument('--WEIGHT-DECAY', type=float, default=1e-4) # 1e-4 1e-3
parser.add_argument('--POLY-POWER', type=float, default=0.9)
parser.add_argument('--WARMUP-FACTOR', type=float, default=1.0 / 3)
parser.add_argument('--WARMUP-METHOD', type=str, default='linear')
parser.add_argument('--batch-size', type=int, default=128) # classfier 128, default 64
parser.add_argument('--max-iters', type=int, default=10000) # classfier 2500, default 100000
parser.add_argument('--warmup-iters', type=int, default=500)

parser.add_argument('--class-head', type=str, default='Linear')

parser.add_argument('--sparse_sampling', type=bool, default=False)
parser.add_argument('--sampleing_gap', type=int, default=5, help='--sparse_sampling must be true. Only effective for training')
parser.add_argument('--image-size', type=int, default=128, help='Size of image')
parser.add_argument('--depth-size', type=int, default=128, help='Size of depth map')
parser.add_argument('--brightness', type=float, default=0.5)
parser.add_argument('--contrast', type=float, default=0)
parser.add_argument('--saturation', type=float, default=0)
parser.add_argument('--degrees', type=int, default=45)
parser.add_argument('--expand', type=bool, default=False)
parser.add_argument('--mean', type=list, default=[0.485, 0.456, 0.406], help='Normalize mean for image')
parser.add_argument('--std', type=list, default=[0.229, 0.224, 0.225], help='Normalize std for image')


args = parser.parse_known_args()[0]

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False


if __name__ == '__main__':
    print(args)

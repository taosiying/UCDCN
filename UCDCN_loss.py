import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch


__Author__ = 'Quanhao Guo'


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def SmoothL1Loss(predictions, targets):
    criterion = nn.SmoothL1Loss()
    loss = criterion(predictions, targets)

    return loss


def SmoothL2Loss(predictions, targets):
    criterion = torch.nn.MSELoss()
    loss = criterion(predictions, targets)

    return loss


def contrast_depth_conv(input):
    ''' compute contrast depth in both of (out, label) '''
    '''
        input  32x32
        output 8x32x32
    '''
    input = input.squeeze(1)

    kernel_filter_list =[
                        [[1,0,0],[0,-1,0],[0,0,0]], [[0,1,0],[0,-1,0],[0,0,0]], [[0,0,1],[0,-1,0],[0,0,0]],
                        [[0,0,0],[1,-1,0],[0,0,0]], [[0,0,0],[0,-1,1],[0,0,0]],
                        [[0,0,0],[0,-1,0],[1,0,0]], [[0,0,0],[0,-1,0],[0,1,0]], [[0,0,0],[0,-1,0],[0,0,1]]
                        ]
    
    kernel_filter = np.array(kernel_filter_list, np.float32)
    
    kernel_filter = torch.from_numpy(kernel_filter.astype(np.float)).float().to(device)
    # weights (in_channel, out_channel, kernel, kernel)
    kernel_filter = kernel_filter.unsqueeze(dim=1)
    
    input = input.unsqueeze(dim=1).expand(input.shape[0], 8, input.shape[1],input.shape[2])
    
    contrast_depth = F.conv2d(input, weight=kernel_filter, groups=8)  # depthwise conv
    
    return contrast_depth


class Contrast_depth_loss(nn.Module):    # Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
    def __init__(self):
        super(Contrast_depth_loss,self).__init__()
        return

    def forward(self, out, label): 
        '''
        compute contrast depth in both of (out, label),
        then get the loss of them
        tf.atrous_convd match tf-versions: 1.4
        '''
        contrast_out = contrast_depth_conv(out)
        contrast_label = contrast_depth_conv(label)

        # criterion = nn.MSELoss()
        criterion = nn.SmoothL1Loss()
    
        loss = criterion(contrast_out, contrast_label)
        # loss = torch.pow(contrast_out - contrast_label, 2)
        # loss = torch.mean(loss)
    
        return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


def contrast_depth_loss(predictions, targets):
    criterion = Contrast_depth_loss()
    loss = criterion(predictions, targets)

    return loss


def cross_entropy_loss(predictions, targets):
    criterion_CEL = nn.CrossEntropyLoss()
    loss = criterion_CEL(predictions, targets)
    
    return loss


def focal_loss(predictions, targets):
    criterion_focal_loss = FocalLoss()
    loss = criterion_focal_loss(predictions, targets)
    
    return loss


if __name__ == '__main__':
    input_x, input_y = torch.ones(128, 128), torch.ones(128, 128) * 3
    L1 = SmoothL1Loss(input_x, input_y)
    L2 = SmoothL2Loss(input_x, input_y)
    print('L1 loss:', L1)
    print('L2 loss:', L2)
    
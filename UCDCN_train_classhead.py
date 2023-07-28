from __future__ import print_function, division
from collections import OrderedDict
import matplotlib.pyplot as plt
from UCDCN_models import *
from UCDCN_logger import Logger
from UCDCN_dataloader import *
from UCDCN_solver import *
from UCDCN_loss import *
from UCDCN_options import args
import torchvision
import random
import shutil
import torch
import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


__Author__ = 'Quanhao Guo'


# def random seed
def seed(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    log.logger.info('Global Random seed is %d!'.format(args.seed))


# plot visual_image
@torch.no_grad()
def plot_val_image(val_loader, network, batchAntiNormalize, path, iters, log):
    loss_sum = 0.0
    network.eval().cuda()
    plot_i = 0
    for data in val_loader:
        plot_i += 1
        image, binary_label = data[0].cuda(), data[-1].cuda()
        predictions, binary_prediction = network(image)
        loss = focal_loss(binary_prediction, binary_label)
        accuracy_rate = sum(binary_prediction.argmax(1) == binary_label) / len(binary_label)
        loss_sum += loss.item()
        data_anti = batchAntiNormalize(data[0])
        fig = plt.gcf()
        fig.set_size_inches(60,60)
        for i in range(args.batch_size * 2):
            ax_img = plt.subplot(math.ceil(args.batch_size * 2 / 16), 16, i + 1)
            if i % 2 == 0:
                plt_img = data_anti[i//2]
                ax_img.imshow(plt_img, cmap='gray')
                ax_img.set_title('depth:' +
                                 str(data[4].cpu()[i//2].item()) +
                                 ' pred:' +
                                 str(binary_prediction.cpu().argmax(1)[i//2].item()))
            else:
                ax_img.imshow(predictions.cpu().numpy()[i//2].reshape(128, 128), cmap='gray')
        plt.savefig(path + '{}_{}.png'.format(str(iters), str(plot_i)))
        
        output_log = 'Eval:{Validation_STEPS:03d}/{TOTAL_STEPS:06d} :' \
                     'binary_loss: {Binary_Loss:.8f} | ' \
                     'Acc : {Acc_Rate:.8f}'.format(
            Validation_STEPS=plot_i,
            TOTAL_STEPS=25,
            Binary_Loss=loss.item(),
            Acc_Rate=accuracy_rate
        )
        log.logger.info(output_log)
        del fig
    return loss_sum / plot_i




# train the model
def main(args):
    time_now = time.strftime("%Y%m%d-%H.%M", time.localtime())
    if not os.path.exists(os.path.join('Experiments', time_now)):
        os.makedirs(os.path.join('Experiments', time_now))
    log = Logger('Experiments/' + time_now+'/training.log',level='info')
    log.logger.info(args)
    log.logger.info('Preparing for the visual images. Loading...')
    log.logger.info('Preparing visual images finished!')
    pre_loss = 1000
    
    train_transform = transforms.Compose([Resize(), 
                                          ColorJitter(), 
                                          RandomHorizontalFlip(),
                                          ToTensor(),
                                          Normalize()])
    val_transform = transforms.Compose([Resize(),
                                        ToTensor(),
                                        Normalize()])

    train_dataset = MyDataset(args.train_path, transform=train_transform, mode='Train')
    train_sampler = make_data_sampler(train_dataset, shuffle=True, distributed=False)
    train_batch_sampler = make_batch_data_sampler(train_sampler, args.batch_size, args.max_iters, drop_last=True)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_sampler=train_batch_sampler,
                              num_workers=2,
                              pin_memory=True)
    
    val_dataset = MyDataset(args.val_path, transform=val_transform, mode='Val')
    val_sampler = make_data_sampler(val_dataset, shuffle=True, distributed=False)
    val_batch_sampler = make_batch_data_sampler(val_sampler, args.batch_size, 25, drop_last=False)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_sampler=val_batch_sampler,
                            num_workers=2,
                            pin_memory=True)
    batchAntiNormalize = BatchAntiNormalize(args.image_size, args.mean, args.std)

    log.logger.info('Start Training!')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    network = NestedUNet(3, 1, args)
    network.to(device)
    optimizer = get_optimizer(network, args.optimizer, args)
    iters = 0
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
        state_dict = OrderedDict()
        for key, value in checkpoint['net'].items():
            if 'classfier' in key:
                continue
            state_dict[key] = value
        network.load_state_dict(state_dict, strict=False)
        log.logger.info('Load pretrained backbone model from ' + args.checkpoint)
        # iters = checkpoint['steps']
        try:
            pre_loss = checkpoint['class_loss']
        except:
            pre_loss = 1000
    else:
        log.logger.info('Training from scratch for classhead!')
        log.logger.info('[Info] initializing weights...')
        init_weights(network)
    
    lr_scheduler = get_scheduler(optimizer, args.max_iters, args.warmup_iters, args, -1)
    
    if not os.path.isdir('Experiments/' + time_now+'/models_classfier/checkpoint'):
        os.makedirs('Experiments/' + time_now+'/models_classfier/checkpoint')
    
    train_params = 0
    for k, v in network.named_parameters():
        if 'classfier' not in k:
            v.requires_grad_(False)
        else:
            train_params += v.numel()
    log.logger.info('Params of Classfier Head:{}'.format(train_params))
    for data in train_loader:
        iters += 1
        network.train()
        image, binary_label = data[0].to(device), data[-1].to(device)
        predictions, binary_prediction = network(image)
        binary_loss = focal_loss(binary_prediction, binary_label)
        accuracy_rate = sum(binary_prediction.argmax(1) == binary_label) / len(binary_label)
        optimizer.zero_grad()
        binary_loss.backward()
        optimizer.step()
        output_log = 'Train:{Training_STEPS:03d}/{TOTAL_STEPS:06d} :' \
                     'binary_loss: {Binary_Loss:.8f} | ' \
                     'Acc : {Acc_Rate:.8f} | ' \
                     'LR: {LR:.8f}'.format(
            Training_STEPS=iters,
            TOTAL_STEPS=args.max_iters,
            Binary_Loss=binary_loss.item(),
            Acc_Rate=accuracy_rate,
            LR=optimizer.param_groups[0]['lr']
        )
        lr_scheduler.step()
        log.logger.info(output_log)

        if iters % args.val_freq == 0:
            save_image_path = 'Experiments/' + time_now+'/models_classfier/images/' + str(iters) + '/'
            if not os.path.exists(save_image_path):
                os.makedirs(save_image_path)
            eval_loss = plot_val_image(val_loader, network, batchAntiNormalize, save_image_path, iters, log)
            checkpoint = {
                "net": network.state_dict(),
                'optimizer':optimizer.state_dict(),
                "steps": iters,
                "class_loss": eval_loss
            }
            torch.save(checkpoint, 'Experiments/' + time_now+'/models_classfier/checkpoint/ckpt_%s.pth' %(str(iters)))
            if eval_loss < pre_loss:
                torch.save(checkpoint, 'Experiments/' + time_now+'/models_classfier/checkpoint/best.pth')
                log.logger.info('Save best model at %d!' % (iters))
                pre_loss = eval_loss
if __name__ == '__main__':
    main(args)

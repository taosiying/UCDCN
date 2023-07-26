from __future__ import print_function, division
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from UCDCN_models import *
from UCDCN_logger import Logger
from UCDCN_dataloader import *
from UCDCN_loss import *
from UCDCN_options import args
import torchvision
import random
import shutil
import torch
import time
import os

'''
0.006016943603754044 0.0009987922385334969 0.05399608984589577 1.0
0.009247217327356339 0.001129336655139923 0.46450790762901306 0.96875
0.0116812689229846 0.0012209228007122874 0.7665342092514038 0.921875
0.01200161688029766 0.0012834924273192883 0.8476740121841431 0.9375
0.005358130671083927 0.0009851802606135607 0.0992167741060257 0.984375
0.00745875108987093 0.0009779080282896757 0.4008135497570038 0.96875
0.009874583221971989 0.0010435277363285422 0.5642639398574829 0.96875
0.00885461550205946 0.001340285176411271 0.3016929030418396 0.96875
0.007003260776400566 0.0008078271057456732 0.30308932065963745 0.96875
'''


__Author__ = 'Quanhao Guo'


# create visual_image
def visual_image(args):
    if not args.resume:
        if os.path.exists('visual_image'):
            shutil.rmtree('visual_image')

        os.makedirs("./visual_image/real")
        shutil_real_images_depth_list = os.listdir(os.path.join(args.val_path, 'real'))
        shutil_real_images_depth_list.sort(key=lambda x: int(x.split('.')[0].split('_')[0]))
        for i in range(32):
            choose_id = random.randint(0, len(shutil_real_images_depth_list) // 2)
            if 'depth' in shutil_real_images_depth_list[choose_id]:
                image = shutil_real_images_depth_list[choose_id].replace("_depth", "")
                depth = shutil_real_images_depth_list[choose_id]
            else:
                image = shutil_real_images_depth_list[choose_id]
                depth = image.replace('.jpg', '_depth.jpg')
            shutil.copy(os.path.join(os.path.join(args.val_path, 'real'), image), os.path.join("./visual_image/real", image))
            shutil.copy(os.path.join(os.path.join(args.val_path, 'real'), depth), os.path.join("./visual_image/real", depth))
            shutil_real_images_depth_list.remove(image)
            shutil_real_images_depth_list.remove(depth)

        os.makedirs("./visual_image/spoof")
        shutil_spoof_images_list = os.listdir(os.path.join(args.val_path, 'spoof'))
        for i in range(32):
            choose_id = random.randint(0, len(shutil_spoof_images_list))
            image = shutil_spoof_images_list[choose_id]
            shutil.copy(os.path.join(os.path.join(args.val_path, 'spoof'), image), os.path.join("./visual_image/spoof", image))
            shutil_spoof_images_list.remove(image)


# train the model
def main(args):
    time_now = time.strftime("%Y%m%d-%H.%M", time.localtime())
    if not os.path.exists(os.path.join('Experiments', time_now)):
        os.makedirs(os.path.join('Experiments', time_now))
    log = Logger('Experiments/' + time_now+'/training.log',level='info')
    log.logger.info(args)
    log.logger.info('Preparing for the visual images. Loading...')
    visual_image(args)
    log.logger.info('Preparing visual images finished!')
    LEARNING_RATE = args.learning_rate
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    START_EPOCH = args.start_epoch
    GLOBAL_STEP = args.global_step
    ADD_IMAGE = args.add_image
    
    torch.manual_seed(7777)
    torch.cuda.manual_seed(7777)
    train_transform = transforms.Compose([Resize(), 
                                          # ColorJitter(), 
                                          RandomRotation(), 
                                          RandomHorizontalFlip(),
                                          ToTensor(), 
                                          # RandomErasing(), 
                                          # RandomCutOut(),
                                          Normalize()])
    val_transform = transforms.Compose([Resize(),
                                        ToTensor(),
                                        Normalize()])
    visual_transform = transforms.Compose([Resize(), 
                                           ToTensor(),
                                           Normalize()])
    train_dataset = MyDataset(args.train_path, transform=train_transform, mode='Train')
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    # val_dataset = MyDataset(args.val_path, transform=val_transform, mode='Val', Sparse_sampling=True)
    # val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    '''
    visual_dataset = MyDataset('./visual_image', transform=visual_transform)
    visual_loader = DataLoader(dataset=visual_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    visual_images, visual_labels, visual_binary_labels = iter(visual_loader).next()['image'], iter(visual_loader).next()['label'], iter(visual_loader).next()['binary_label']
    plot_visual_images = BatchAntiNormalize()(visual_images)
    '''
    writer = SummaryWriter('Experiments/' + time_now + '/UCDCN_logs')
    '''
    fig_images = plt.figure()
    fig_images.set_size_inches(20, 23)
    log.logger.info('Preparing for the TensorBoard')
    for i in range(64):
        ax_img = plt.subplot(8, 8, i+1)
        if visual_binary_labels[i].item() == 1:
            ax_img.set_title('1')
        else:
            ax_img.set_title('0')
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        ax_img.imshow(plot_visual_images[i].permute(1, 2, 0))
    
    writer.add_figure('IMAGES', fig_images, 0)
    
    fig_labels = plt.figure()
    fig_labels.set_size_inches(20, 23)

    for i in range(64):
        ax_label = plt.subplot(8, 8, i+1)
        if visual_binary_labels[i].item() == 1:
            ax_label.set_title('1')
        else:
            ax_label.set_title('0')
        ax_label.set_xticks([])
        ax_label.set_yticks([])
        ax_label.imshow(visual_labels[i][0])
    
    writer.add_figure('LABELS', fig_labels, 0)
    log.logger.info('Preparing the TensorBoard finished!')
    '''
    log.logger.info('Start Training!')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    network = NestedUNet(3, 1)
    
    network.to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE, weight_decay=0.00005)
    
    if args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location=lambda storage, loc: storage)
        network.load_state_dict(checkpoint['net'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        log.logger.info('Load pretrained model from ' + args.pretrained)
    elif args.resume:
        path_checkpoint = args.checkpoint
        checkpoint = torch.load(path_checkpoint, map_location=lambda storage, loc: storage)
        network.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        START_EPOCH = checkpoint['epoch']
        GLOBAL_STEP = checkpoint['GLOBAL_STEP']
        # ADD_IMAGE = checkpoint['ADD_IMAGE']
        log.logger.info('Load checkpoint from ' + path_checkpoint)
    else:
        print('[Info] initializing weights...')
        init_weights(network)

    
    
    scheduler = StepLR(optimizer, step_size=1000, gamma=0.95)
    
    if not os.path.isdir('Experiments/' + time_now+'/models/checkpoint'):
        os.makedirs('Experiments/' + time_now+'/models/checkpoint')
    
    val_plot = 0
    network.linear1.requires_grad = False
    network.linear2.requires_grad = False
    for epoch in range(START_EPOCH, EPOCHS):
        network.train()
        for i, data in enumerate(train_loader):
            GLOBAL_STEP += 1
            image, label, binary_label = data['image'].to(device), data['label'].to(device), data['binary_label'].to(device)
            predictions, binary_prediction = network(image)
            smoothL1_loss = SmoothL1Loss(predictions, label)
            contrast_loss = contrast_depth_loss(predictions, label)
            # accuracy_rate = sum(binary_prediction.argmax(1) == binary_label) / len(binary_label)
            accuracy_rate = 0.0
            # binary_loss = focal_loss(binary_prediction, binary_label)
            binary_loss = torch.tensor(1.0)
            loss = smoothL1_loss + contrast_loss
            # loss = 5 * smoothL1_loss + 10 * contrast_loss + binary_loss
            # loss = smoothL1_loss + contrast_loss + binary_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            output_log = '({Training_EPOCH:03d}/{Global_Step:06d}) :' \
                         'smoothL1_loss: {SmoothL1_Loss:.8f} | ' \
                         'contrast_loss: {Contrast_Loss:.8f} | ' \
                         'binary_loss: {Binary_Loss:.8f} | ' \
                         'Train TotalLoss: {Training_TotalLoss:.8f} | ' \
                         'LR: {LR:.8f} | ' \
                         'Acc : {Acc_Rate:.8f}'.format(
                Training_EPOCH=epoch + 1,
                Global_Step=GLOBAL_STEP,
                SmoothL1_Loss=smoothL1_loss.item(),
                Contrast_Loss=contrast_loss.item(),
                Binary_Loss=binary_loss.item(),
                Training_TotalLoss=loss.item(),
                LR=optimizer.param_groups[0]['lr'],
                Acc_Rate=accuracy_rate
            )
            scheduler.step()
            log.logger.info(output_log)

            writer.add_scalar('Loss/train', loss.item(), GLOBAL_STEP)

            if GLOBAL_STEP % args.val_freq == 0:
                '''
                val_plot += 1
                with torch.no_grad():
                    sum_val_acc_rate = 0.0
                    network.eval()
                    visual_predictions, binary_predictions = network(visual_images.to(device))
                    visual_predictions = (visual_predictions * 255).to(torch.uint8).cpu()
                    binary_predictions = binary_predictions.cpu()
                    binary_predictions = torch.argmax(binary_predictions, dim=1)
                    
                    fig_predictions = plt.figure()
                    fig_predictions.set_size_inches(20, 23)
    
                    for i in range(64):
                        ax_predictions = plt.subplot(8, 8, i+1)
                        if binary_predictions[i].item() == 1:
                            ax_predictions.set_title('1')
                        else:
                            ax_predictions.set_title('0')
                        ax_predictions.set_xticks([])
                        ax_predictions.set_yticks([])
                        ax_predictions.imshow(visual_predictions[i].permute(1, 2, 0))
    
                    writer.add_figure('PREDICTIONS', fig_predictions, ADD_IMAGE)
                    
                    ADD_IMAGE += 1

                    for valid_i, data in enumerate(val_loader):
                        image, label, binary_label = data['image'].to(device), data['label'].to(device), data['binary_label'].to(device)
                        predictions, binary_prediction = network(image)
                        smoothL1_loss = SmoothL1Loss(predictions, label)
                        contrast_loss = contrast_depth_loss(predictions, label)
                        binary_loss = cross_entropy_loss(binary_prediction, binary_label)
                        accuracy_rate = sum(binary_prediction.argmax(1) == binary_label) / len(binary_label)
                        loss = 5 * smoothL1_loss + 10 * contrast_loss + binary_loss
                        output_log = '({Valition_EPOCH:03d}/{Val_Step:06d}) :' \
                        'smoothL1_loss: {SmoothL1_Loss:.8f} | ' \
                        'contrast_loss: {Contrast_Loss:.8f} | ' \
                        'binary_loss: {Binary_Loss:.8f} | ' \
                        'Valiation TotalLoss: {Valiation_Loss:.8f} | ' \
                        'Acc : {Acc_Rate:.8f}'.format(
                            Valition_EPOCH=epoch + 1,
                            Val_Step=valid_i+1,
                            SmoothL1_Loss=smoothL1_loss.item(),
                            Contrast_Loss=contrast_loss.item(),
                            Binary_Loss=binary_loss.item(),
                            Valiation_Loss=loss.item(),
                            Acc_Rate=accuracy_rate
                        )
                        
                        sum_val_acc_rate += accuracy_rate
                        log.logger.info(output_log)

                        writer.add_scalar('Loss/validation/Val_Plot'+str(val_plot), loss.item(), valid_i+1)
                    log.logger.info('Total validation accuracy rate: {Total_val_acc_rate:.8f}'.format(Total_val_acc_rate=sum_val_acc_rate / (valid_i+1)))
                '''
                checkpoint = {
                    "net": network.state_dict(),
                    'optimizer':optimizer.state_dict(),
                    "epoch": epoch,
                    "GLOBAL_STEP": GLOBAL_STEP,
                    # "ADD_IMAGE": ADD_IMAGE
                }
                torch.save(checkpoint, 'Experiments/' + time_now+'/models/checkpoint/ckpt_%s.pth' %(str(GLOBAL_STEP)))
    writer.close()


if __name__ == '__main__':
    # Train the network
    main(args)

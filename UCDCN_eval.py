from UCDCN_options import args
from UCDCN_models import *
from UCDCN_logger import Logger
from UCDCN_dataloader import *
import torch
import time
import os
import sys
# python UCDCN_eval.py --checkpoint Experiments\20211216-20.53\models_classfier\checkpoint\best.pth
@torch.no_grad()
def main(args):
    time_now = time.strftime("%Y%m%d-%H.%M", time.localtime())
    if not os.path.exists(os.path.join('Experiments', time_now)):
        os.makedirs(os.path.join('Experiments', time_now))
    log = Logger('Experiments/' + time_now+'/eval.log',level='info')
    log.logger.info(args)
    val_transform = transforms.Compose([Resize(),
                                        ToTensor(),
                                        Normalize()])
    val_dataset = MyDataset(args.val_path, transform=val_transform, mode='Val')
    val_sampler = make_data_sampler(val_dataset, shuffle=True, distributed=False)
    val_batch_sampler = make_batch_data_sampler(val_sampler, args.batch_size, drop_last=False)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_sampler=val_batch_sampler,
                            num_workers=0,
                            pin_memory=True)
    log.logger.info(str(len(val_dataset)) + ' IMAGES! Start Evaluation!')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    network = NestedUNet(3, 1, args)
    
    network.to(device)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
        network.load_state_dict(checkpoint['net'])
    else:
        log.logger.info('NO MODEL')
        sys.exit()
    network.eval()
    iters = 0
    total_acc = 0
    apcer = 0
    bpcer = 0
    nums = 0
    for data in val_loader:
        iters += 1
        begin_time = time.time()
        image, binary_label = data[0].cuda(), data[-1].cuda()
        _, binary_prediction = network(image)
        # accuracy_rate = sum(binary_prediction.argmax(1) == binary_label) / len(binary_label)
        binary_prediction = (binary_prediction.softmax(1).argmax(1) >= args.probs).long()
        CER = binary_prediction - binary_label
        accuracy_rate = sum(binary_prediction == binary_label) / len(binary_label)
        nums += len(binary_label)
        total_acc +=  sum(binary_prediction == binary_label)
        apcer += sum(CER == 1)
        bpcer += sum(CER == -1)
        output_log = 'Eval:{Validation_STEPS:04d}/{TOTAL_STEPS:04d} :' \
                     'BatchACC : {BatchACC:.5f} | ' \
                     'TotalACC: {TotalACC:.5f} | ' \
                     'APCER: {APCER:.5f} | ' \
                     'BPCER: {BPCER:.5f} | ' \
                     'ACER: {ACER:.5f} | ' \
                     'RightPred: {RightPred:07d} | ' \
                     'TotalPred: {TotalPred:07d}'.format(
            Validation_STEPS=iters,
            TOTAL_STEPS=len(val_loader),
            BatchACC=accuracy_rate,
            TotalACC=total_acc / nums,
            APCER=apcer / nums,
            BPCER=bpcer / nums,
            ACER=(apcer+bpcer) / nums / 2,
            RightPred=total_acc.item(),
            TotalPred=nums 
        )
        log.logger.info(output_log)

if __name__ == '__main__':
    main(args)
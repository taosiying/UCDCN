from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import math
import pickle
import torch.distributed as dist
from torch.utils.data.sampler import Sampler, BatchSampler, RandomSampler, SequentialSampler
from UCDCN_options import args
from PIL import Image
import numpy as np
import argparse
import random
import torch
import sys
import os
"""
code is heavily based on https://github.com/facebookresearch/maskrcnn-benchmark
"""


__Author__ = 'Quanhao Guo'

# define the Mydataset class to get the item one by one
class MyDataset(Dataset):
    '''
    Define Dataset
    '''
    def __init__(self, root_dir, transform=None, mode=None, Sparse_sampling=args.sparse_sampling, sampleing_gap=args.sampleing_gap):
        '''
        Initial the variables:image path and label path
        '''
        self.real_depth = []
        self.real_images = []
        self.root_dir = root_dir
        self.transform = transform
        self.real_dir, self.fake_dir = os.listdir(root_dir)[0], os.listdir(root_dir)[1]
        for image in os.listdir(os.path.join(root_dir, self.real_dir)):
            if image.split('_')[-1] == 'depth.jpg':
                self.real_depth.append(os.path.join(root_dir, self.real_dir, image))
            else:
                self.real_images.append(os.path.join(root_dir, self.real_dir, image))
        self.real_depth.sort(key=lambda x: int(x.split('\\')[-1].split('_')[0]))
        self.real_images.sort(key=lambda x: int(x.split('\\')[-1].split('.')[0]))
        self.fake_images = [os.path.join(root_dir, self.fake_dir, image) for image in os.listdir(os.path.join(root_dir, self.fake_dir))]
        self.fake_images.sort(key=lambda x: int(x.split('\\')[-1].split('.')[0]))
        self.fake_depth = ['blank.png'] * len(self.fake_images)
        self.images = self.real_images + self.fake_images
        self.labels = self.real_depth + self.fake_depth
        self.binary_labels = [1] * len(self.real_depth) + [0] * len(self.fake_depth)
        self.length = len(self.images)
        
        # Sparse sampling
        if Sparse_sampling:
            self.sampling_images = []
            self.sampling_labels = []
            self.sampling_binary_labels = []
            if mode == 'Train':
                sampleing_gap = sampleing_gap
            elif mode == 'Val':
                sampleing_gap = self.length // args.val_length
                randnum = random.randint(0,100)
                random.seed(randnum)
                random.shuffle(self.images)
                random.seed(randnum)
                random.shuffle(self.labels)
                random.seed(randnum)
                random.shuffle(self.binary_labels)
            else:
                sampleing_gap = 1
            for i in range(len(self.images)):
                if i % sampleing_gap == 0:
                    self.sampling_images.append(self.images[i])
                    self.sampling_labels.append(self.labels[i])
                    self.sampling_binary_labels.append(self.binary_labels[i])
            self.images = self.sampling_images
            self.labels = self.sampling_labels
            self.binary_labels = self.sampling_binary_labels
            self.length = len(self.sampling_images)
            

    def __len__(self):
        '''
        return the length of dataset
        '''
        return self.length

    def __getitem__(self, idx):
        '''
        get the item
        '''
        image = Image.open(self.images[idx])
        label = Image.open(self.labels[idx]).convert('L')

        # sample = {'image':image, 'image_name': self.images[idx], 'label':label, 'label_name': self.labels[idx], 'binary_label': self.binary_labels[idx]}
        sample = (image, label, self.images[idx], self.labels[idx], self.binary_labels[idx])

        if self.transform:
            sample = self.transform(sample)
        sample = (np.asarray(sample[0]), np.asarray(sample[1]), sample[2], sample[3], sample[4])
        return sample


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.IntTensor([tensor.numel()]).to("cuda")
    size_list = [torch.IntTensor([0]).to("cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to("cuda"))
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to("cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.reduce(values, dst=0)
        if dist.get_rank() == 0 and average:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        return DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)
    return sampler


def make_batch_data_sampler(sampler, images_per_batch, num_iters=None, start_iter=0, drop_last=True):
    batch_sampler = BatchSampler(sampler, images_per_batch, drop_last=drop_last)
    if num_iters is not None:
        batch_sampler = IterationBasedBatchSampler(batch_sampler, num_iters, start_iter)
    return batch_sampler


class DistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset: offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class IterationBasedBatchSampler(BatchSampler):
    """
    Wraps a BatchSampler, resampling from it until
    a specified number of iterations have been sampled
    """

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration <= self.num_iterations:
            # if the underlying sampler has a set_epoch method, like
            # DistributedSampler, used for making each process see
            # a different split of the dataset, then set it
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                iteration += 1
                if iteration > self.num_iterations:
                    break
                yield batch

    def __len__(self):
        return self.num_iterations

    
##################################################################################################################
# define the transoforms [Resize, RandomErasing, RandomCutOut, ColorJitter, RandomRotation, Normalize, ToTensor] #
##################################################################################################################
class Resize():
    '''
    Resize the image and lable
    '''
    def __init__(self, image_size=args.image_size, depth_size=args.depth_size):
        self.image_size = image_size
        self.depth_size= depth_size
        
    def __call__(self, sample):
        image, label = sample[0], sample[1]
        sample = (transforms.Resize([self.image_size, self.image_size])(image),
                  transforms.Resize([self.image_size, self.image_size])(label),
                  sample[2],
                  sample[3],
                  sample[4])
        return sample

    
class RandomErasing():
    '''
    Random Erasing the image no label
    Input must be tensor
    '''
    def __init__(self):
        self.flag = random.randint(0, 1)
        self.value_list = [(random.random(), random.random(), random.random()), 'random', (random.randint(0,1), random.randint(0,1), random.randint(0,1))]
        random.shuffle(self.value_list)

    def __call__(self, sample):
        if self.flag:
            image = sample[0]
            if not torch.is_tensor(image):
                image = transforms.ToTensor()(image)
            sample = (transforms.RandomErasing(p=1, scale=(0.02, 0.25), ratio=(0.3, 3.3), value=self.value_list[0], inplace=False)(image),
                      sample[1],
                      sample[2],
                      sample[3],
                      sample[4])
        
        return sample


class RandomCutOut():
    '''
    Random Cut Out for the image no label
    Input must be tensor
    '''
    def __init__(self):
        self.flag = random.randint(0, 1)
        
    def __call__(self, sample):
        if self.flag:
            image = sample[0]
            if not torch.is_tensor(image):
                image = transforms.ToTensor()(image)
            h, w = image.shape[1], image.shape[2]
            mask = np.ones((h, w), np.float32)
            y = np.random.randint(h)
            x = np.random.randint(w)
            length_new = np.random.randint (1, 50)
            y1 = np.clip(y - length_new // 2, 0, h)
            y2 = np.clip(y + length_new // 2, 0, h)
            x1 = np.clip(x - length_new // 2, 0, w)
            x2 = np.clip(x + length_new // 2, 0, w)
            mask[y1: y2,x1: x2] = 0.
            mask = torch.from_numpy(mask)
            mask = mask.expand_as(image)
            image *= mask
            sample = (image,
                      sample[1],
                      sample[2],
                      sample[3],
                      sample[4])
        
        return sample


class ColorJitter():
    '''
    Random ColorJitter for image no label
    '''
    def __init__(self, brightness=args.brightness, contrast=args.contrast, saturation=args.saturation):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, sample):
        image = sample[0]
        sample = (transforms.ColorJitter(brightness=self.brightness, contrast=self.contrast, saturation=self.saturation)(image),
                  sample[1],
                  sample[2],
                  sample[3],
                  sample[4])
        
        return sample


class RandomRotation():
    '''
    Random Rotation for image and label
    '''
    def __init__(self, degrees=args.degrees, expand=args.expand):
        self.degrees = degrees
        self.expand = expand
        self.seed = torch.random.seed()
        
        
    def __call__(self, sample):
        image, label = sample[0], sample[1]
        if random.randint(0, 1):
            torch.manual_seed(self.seed)
            image = transforms.RandomRotation(degrees=self.degrees, expand=self.expand)(image)
            torch.manual_seed(self.seed)
            label = transforms.RandomRotation(degrees=self.degrees, expand=self.expand)(label)
        
        sample = (image,
                  label,
                  sample[2],
                  sample[3],
                  sample[4])
        return sample


class RandomHorizontalFlip():
    '''
    Random Flip for image and label
    '''
    def __call__(self, sample):
        if random.randint(0, 1):
            image, label = sample[0], sample[1]
            image = transforms.RandomHorizontalFlip(1.0)(image)
            label = transforms.RandomHorizontalFlip(1.0)(label)
            sample = (image,
                      label,
                      sample[2],
                      sample[3],
                      sample[4])
        
        return sample


class Normalize():
    '''
    Normalize the image no label
    Input must be tensor

    initial:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    '''
    def __init__(self, mean=args.mean, std=args.std):
        self.mean = mean
        self.std = std
        
    def __call__(self, sample):
        image, label = sample[0], sample[1]
        if not torch.is_tensor(image):
            image = transforms.ToTensor()(image)
            label = transforms.ToTensor()(label)
        # sample['image'], sample['label'] = transforms.Normalize(mean=self.mean, std=self.std)(image), transforms.Normalize(mean=[0.0], std=[1.0])(label)
        image, label = transforms.Normalize(mean=self.mean, std=self.std)(image), label
        sample = (image,
                  label,
                  sample[2],
                  sample[3],
                  sample[4])
        return sample


class ToTensor():
    '''
    Translate PIL.Image or Numpy.Array to Torch.Tensor
    '''
    def __call__(self, sample):
        image, label = sample[0], sample[1]
        image, label = transforms.ToTensor()(image), transforms.ToTensor()(label)
        
        sample = (image,
                  label,
                  sample[2],
                  sample[3],
                  sample[4])
        
        return sample


#############################################
# define the Batch Anti-Normalize for imshow#
#############################################
class BatchAntiNormalize():
    def __init__(self, image_size=args.image_size, mean=args.mean, std=args.std):
        self.t_mean = torch.FloatTensor(mean).view(3, 1, 1).expand(1, 3, image_size, image_size)
        self.t_std = torch.FloatTensor(std).view(3, 1, 1).expand(1, 3, image_size, image_size)

    def __call__(self, images):
        images = images * self.t_std +self.t_mean
        return images.permute(0, 2, 3, 1).cpu().numpy()


if __name__ == '__main__':
    '''python ./UCDCN_dataloader.py -i F:/FaceAntiSpoofing/ALL_DATASET/OULU_F10/DATASET/SCALE_0.8/Test_files'''
    # test the dataloader function
    parser = argparse.ArgumentParser(description='test the dataloader function')
    parser.add_argument('-i', '--input_path', type=str, default='./Demo_images')
    args = parser.parse_args()
    torch.manual_seed(7)
    train_dataset = MyDataset(args.input_path,
                        transform=transforms.Compose([Resize(),
                          # ColorJitter(),
                          # RandomRotation(),
                          ToTensor(),
                          # RandomErasing(),
                          # RandomCutOut(),
                          # Normalize()]),
                          ]),
                        Sparse_sampling=False,
                        mode='Val')
    # train_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=True, num_workers=0)

    train_sampler = make_data_sampler(train_dataset, shuffle=True, distributed=False)
    train_batch_sampler = make_batch_data_sampler(train_sampler, 1, 100, drop_last=True)
    train_loader = DataLoader(dataset=train_dataset,
                            batch_sampler=train_batch_sampler,
                            num_workers=8,
                            pin_memory=True)

    k = 0
    for i in train_loader:
        k += 1
        if len(i[3][0]) > 9:
            print(i[0].shape, i[1].shape, i[2], ' '*(30 - len(i[2][0])), i[3], ' '*(40 - len(i[3][0])-2), i[4], k)
        else:
            print(i[0].shape, i[1].shape, i[2], ' '*(30 - len(i[2][0])), i[3], ' '*(40 - len(i[3][0])), i[4], k)

        
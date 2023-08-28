import os
import sys
import cv2
import torch
import numpy as np
from torchvision import transforms
import torch.nn.functional as F

from datasets.utils.configs import *

def get_default_mtl_configs():
    all_tasks = ['semseg', 'human_parts', 'sal', 'normals', 'edge', 'depth']
    default_mtl_configs = {
        'semseg': {'task_flagval': cv2.INTER_NEAREST, 'task_infer_flagval': cv2.INTER_NEAREST},
        'human_parts': {'task_flagval': cv2.INTER_NEAREST, 'task_infer_flagval': cv2.INTER_NEAREST},
        'sal': {'task_flagval': cv2.INTER_NEAREST, 'task_infer_flagval': cv2.INTER_LINEAR},
        'normals': {'task_flagval': cv2.INTER_CUBIC, 'task_infer_flagval': cv2.INTER_LINEAR, 'normloss':1},
        'edge': {'task_flagval': cv2.INTER_NEAREST, 'task_infer_flagval': cv2.INTER_LINEAR, 'edge_w':0.95, 'eval_edge':False},
        'depth':{'task_flagval': cv2.INTER_NEAREST, 'task_infer_flagval': cv2.INTER_LINEAR, 'depthloss':'l1'},
        'image': {'task_flagval': cv2.INTER_CUBIC}
    }
    return all_tasks, default_mtl_configs

def get_transformations(data_config, default_mtl_configs, all_tasks):
    """ Return transformations for training and evaluationg """
    from datasets import custom_transforms as tr

    # Training transformations
    if data_config['dataname'].lower() == 'nyud':
        # Horizontal flips with probability of 0.5
        transforms_tr = [tr.RandomHorizontalFlip()]
        
        # Rotations and scaling
        transforms_tr.extend([
            tr.ScaleNRotate(
                rots=[0],
                scales=[1.0, 1.2, 1.5],
                flagvals={x: default_mtl_configs[x]['task_flagval'] for x in default_mtl_configs}
            )
        ])

    elif data_config['dataname'].lower() == 'pascalcontext':
        # Horizontal flips with probability of 0.5
        transforms_tr = [tr.RandomHorizontalFlip()]
        
        # Rotations and scaling
        transforms_tr.extend([tr.ScaleNRotate(rots=(-20, 20), scales=(.75, 1.25),
                                              flagvals={x: default_mtl_configs[x]['task_flagval'] for x in default_mtl_configs})])

    else:
        raise ValueError('Invalid train db name'.format(data_config['dataname']))


    # Fixed Resize to input resolution
    transforms_tr.extend([tr.FixedResize(resolutions={x: tuple(TRAIN_SCALE[data_config['dataname'].lower()]) for x in default_mtl_configs},
                                         flagvals={x: default_mtl_configs[x]['task_flagval'] for x in default_mtl_configs})])
    transforms_tr.extend([tr.AddIgnoreRegions(), tr.ToTensor(),
                          tr.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    transforms_tr = transforms.Compose(transforms_tr)

    
    # Testing (during training transforms)
    transforms_ts = []
    transforms_ts.extend([tr.FixedResize(resolutions={x: tuple(TEST_SCALE[data_config['dataname'].lower()]) for x in default_mtl_configs},
                                         flagvals={x: default_mtl_configs[x]['task_flagval'] for x in default_mtl_configs})])
    transforms_ts.extend([tr.AddIgnoreRegions(), tr.ToTensor(),
                          tr.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    transforms_ts = transforms.Compose(transforms_ts)

    return transforms_tr, transforms_ts

def partition_data(dataset_configs, args):
    print('Partitioning data .......')
    all_tasks, default_mtl_configs = get_default_mtl_configs()

    data_tools = {}
    for data_config in dataset_configs:
        train_transforms, val_transforms = get_transformations(data_config, default_mtl_configs, all_tasks)
        n_nets = sum(data_config['task_dict'].values())
        print('TRAINING %d NETS on %s'%(n_nets, data_config['dataname']))

        task_list = []
        for task_name in data_config['task_dict']:
            task_list += [task_name] * data_config['task_dict'][task_name]
        assert len(task_list) == n_nets

        if args.partition == 'homo':
            idxs = np.random.permutation(NUM_TRAIN_IMAGES[data_config['dataname'].lower()])
            batch_idxs = np.array_split(idxs, n_nets)
            net_task_dataidx_map = [{
                'task': task_list[i],
                'dataidx': batch_idxs[i]} for i in range(n_nets)]
        
        data_tools[data_config['dataname']] = {}
        data_tools[data_config['dataname']]['n_nets'] = n_nets
        data_tools[data_config['dataname']]['task_list'] = task_list
        data_tools[data_config['dataname']]['lr'] = data_config['lr']
        data_tools[data_config['dataname']]['nworkers'] = data_config['nworkers']
        data_tools[data_config['dataname']]['batch_size'] = data_config['batch_size']
        data_tools[data_config['dataname']]['train_transforms'] = train_transforms
        data_tools[data_config['dataname']]['val_transforms'] = val_transforms
        data_tools[data_config['dataname']]['net_task_dataidx_map'] = net_task_dataidx_map
    
    return data_tools, default_mtl_configs

def init_models(net_task_dataidx_map, n_nets, args, dataname):
    '''
    Initialize the local LeNets
    Please note that this part is hard coded right now
    '''

    nets = {net_i: None for net_i in range(n_nets)}

    # get the backbone model that is commonly shared by all clients

    for net_index in range(n_nets):
        task = net_task_dataidx_map[net_index]['task']
        backbone, backbone_channels = get_backbone(args)
        head = get_head(args.head, backbone_channels, task, dataname)
        from models.models import SingleTaskModel
        model = SingleTaskModel(backbone, head, task)
        nets[net_index] = model.cuda()

    return nets

def get_backbone(args):
    """ Return the backbone """

    backbone_type, backbone_pretrain = args.backbone, args.backbone_pretrain

    if backbone_type == 'resnet18':
        from models.resnet import resnet18
        backbone = resnet18(backbone_pretrain)
        backbone_channels = 512
    
    elif backbone_type == 'resnet50':
        from models.resnet import resnet50
        backbone = resnet50(backbone_pretrain)
        backbone_channels = 2048
    
    elif backbone_type == 'resnet101':
        from models.resnet import resnet101
        backbone = resnet101(backbone_pretrain)
        backbone_channels = 2048
    
    else:
        raise NotImplementedError

    if args.backbone_dilated: # Add dilated convolutions
        assert(backbone_type in ['resnet18', 'resnet50', 'resnet101'])
        from models.resnet_dilated import ResnetDilated
        backbone = ResnetDilated(backbone)

    return backbone, backbone_channels

def get_head(head_type, backbone_channels, task, dataname):
    """ Return the decoder head """

    if head_type == 'deeplab':
        from models.aspp import DeepLabHead
        return DeepLabHead(backbone_channels, get_output_num(task, dataname.lower()))

    else:
        raise NotImplementedError

def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def get_output(output, task):
    output = output.permute(0, 2, 3, 1)
    
    if task == 'normals':
        output = (F.normalize(output, p = 2, dim = 3) + 1.0) * 255 / 2.0
    
    elif task in {'semseg', 'human_parts'}:
        _, output = torch.max(output, dim=3)
    
    elif task in {'edge', 'sal'}:
        output = torch.squeeze(255 * 1 / (1 + torch.exp(-output)))
    
    elif task in {'depth'}:
        pass
    
    else:
        raise ValueError('Select one of the valid tasks')

    return output

def get_schedule_weight(args):
    if args.max_weight <= 1:
        return np.array([args.max_weight]*args.comm_round)
    else:
        return np.arange(start=1, stop=args.max_weight, step=(args.max_weight-1)/args.comm_round)
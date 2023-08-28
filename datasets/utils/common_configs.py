
from torch.utils.data import DataLoader

from .custom_collate import collate_mil

def get_train_dataset(dataname, tasks, transform, dataidxs, overfit=False):
    """ Return the train dataset """

    # print('Preparing train loader for db: {}'.format(dataname))

    if dataname.lower() == 'pascalcontext':
        from datasets.pascal_context import PASCALContext
        database = PASCALContext(split=['train'],
                                 transform=transform,
                                 retname=True,
                                 tasks=tasks,
                                 dataidxs=dataidxs,
                                 overfit=overfit)

    elif dataname.lower() == 'nyud':
        from datasets.nyud import NYUD_MT_truncated
        database = NYUD_MT_truncated(split='train',
                                     transform=transform,
                                     tasks=tasks,
                                     dataidxs=dataidxs,
                                     overfit=overfit)

    else:
        raise NotImplemented("train_db_name: Choose among PASCALContext and NYUD")

    return database

def get_train_dataloader(configs, ds):
    """ Return the train dataloader """

    trainloader = DataLoader(ds, batch_size=configs['batch_size'], shuffle=True, drop_last=True,
                             num_workers=configs['nworkers'], collate_fn=collate_mil)
    return trainloader

def get_val_dataset(dataname, tasks, transform, dataidxs, overfit=False):
    """ Return the validation dataset """

    # print('Preparing val loader for db: {}'.format(db_name))

    if dataname.lower() == 'pascalcontext':
        from datasets.pascal_context import PASCALContext
        database = PASCALContext(split=['val'],
                                 transform=transform, 
                                 retname=True,
                                 tasks=tasks,
                                 dataidxs=dataidxs,
                                 overfit=overfit)
    
    elif dataname.lower() == 'nyud':
        from datasets.nyud import NYUD_MT_truncated
        database = NYUD_MT_truncated(split='val',
                                     transform=transform,
                                     tasks=tasks,
                                     dataidxs=dataidxs,
                                     overfit=overfit)

    else:
        raise NotImplemented("test_db_name: Choose among PASCALContext and NYUD")

    return database


def get_val_dataloader(configs, ds):
    """ Return the validation dataloader """

    testloader = DataLoader(ds, batch_size=configs['batch_size'], shuffle=False, drop_last=False,
                            num_workers=configs['nworkers'])
    return testloader

def get_loss(default_mtl_configs, task=None):
    """ Return loss function for a specific task """

    if task == 'edge':
        from losses.loss_functions import BalancedCrossEntropyLoss
        criterion = BalancedCrossEntropyLoss(size_average=True, pos_weight=default_mtl_configs['edge']['edge_w'])

    elif task == 'semseg' or task == 'human_parts':
        from losses.loss_functions import SoftMaxwithLoss
        criterion = SoftMaxwithLoss()

    elif task == 'normals':
        from losses.loss_functions import NormalsLoss
        criterion = NormalsLoss(normalize=True, size_average=True, norm=default_mtl_configs['normals']['normloss'])

    elif task == 'sal':
        from losses.loss_functions import BalancedCrossEntropyLoss
        criterion = BalancedCrossEntropyLoss(size_average=True)

    elif task == 'depth':
        from losses.loss_functions import DepthLoss
        criterion = DepthLoss(default_mtl_configs['depth']['depthloss'])

    else:
        raise NotImplementedError('Undefined Loss: Choose a task among '
                                  'edge, semseg, human_parts, sal, depth, or normals')

    return criterion
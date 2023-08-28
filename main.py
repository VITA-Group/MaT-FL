import os
import sys
import copy
import yaml
import torch
import argparse
import numpy as np

from utils import *
from dg_utils import aggregate
from datasets.utils.common_configs import *

from evaluation.evaluate_utils import save_model_predictions, eval_all_results

def local_train(task, train_dl, loss_ft, lr, model, args):

    local_optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr , weight_decay=0.0001, amsgrad=True)
    
    for epoch in range(args.local_epochs):
        model.train()

        epoch_loss_collector = []
        for batch_idx, batch in enumerate(train_dl):
            x = batch['image'].cuda(non_blocking=True)
            target = batch[task].cuda(non_blocking=True)

            if torch.min(target) == 255 and 'human' in task:
                continue

            local_optimizer.zero_grad()

            out = model(x)
            loss = loss_ft(out[task], target)
            epoch_loss_collector.append(loss.item())

            loss.backward()
            local_optimizer.step()
        
        print('Epoch: %d Avg Loss: %.4f'%(epoch, sum(epoch_loss_collector)/len(epoch_loss_collector)))

    return model

def evaluation(mtl_configs, database, test_dl, model, save_dir_root, client_idx, task, dataidx=None):
    save_model_predictions(mtl_configs, test_dl, model, save_dir_root, client_idx, tasks=[task])
    curr_result = eval_all_results(save_dir_root, database, client_idx, tasks=[task], dataidx=dataidx)
    return curr_result


def main(all_nets, test_dls, train_losses, data_tools, args, mtl_configs):

    if args.aggregation in ['model-soup','rws', 'stable-rws']:
        schedule_weight = get_schedule_weight(args)
    
    save_ckpt = {}
    for model_idx in all_nets:
        save_ckpt[model_idx] = copy.deepcopy(all_nets[model_idx]['model'].state_dict())

    results = {}
    for cr in range(args.comm_round):
        eval_bool = True if args.eval and (cr+1)%args.eval_freq==0 else False
        if eval_bool:
            results[cr] = {}
        
        for model_idx in all_nets:
            task, train_dl, model, dataname = \
                all_nets[model_idx]['task'], all_nets[model_idx]['train_dl'], all_nets[model_idx]['model'], all_nets[model_idx]['dataname']
            lr = data_tools[dataname]['lr'] * (args.lr_decay_cr ** cr)
            print('CR %d [lr %.5f] (NET %2d) DATASET: %s on %s [training images: %d]'%(cr, lr, model_idx, dataname, task, len(train_dl.dataset)))

            model = local_train(task=task,
                train_dl=train_dl, loss_ft=train_losses[task], lr=lr,
                model=model.cuda(), args=args)
            
            if task != 'edge' and eval_bool:
                results[cr][model_idx] = evaluation(mtl_configs, database=dataname, test_dl=test_dls[dataname][task]['dl'], model=model, 
                    save_dir_root=args.root_dir, client_idx=model_idx, task=task, dataidx=test_dls[dataname][task]['dataidx'])

        if args.save_all_ckpts:
            save_ckpt_temp = {}
            for model_idx in all_nets:
                save_ckpt_temp[model_idx] = copy.deepcopy(all_nets[model_idx]['model'].state_dict())

            torch.save({
                'result': results,
                'ckpts': save_ckpt_temp
            }, os.path.join(args.root_dir, 'checkpoint_%d.pth.tar'%cr))
            del save_ckpt_temp

        all_nets = aggregate(all_nets, save_ckpt, args.aggregation, root_dir=args.root_dir,
            weight=schedule_weight[cr] if args.aggregation in ['model-soup','rws', 'stable-rws'] else None, cluster_num=args.cluster_num, th=args.th)
        
        save_ckpt = {}
        for model_idx in all_nets:
            save_ckpt[model_idx] = copy.deepcopy(all_nets[model_idx]['model'].state_dict())
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MTL-FL')
    parser.add_argument('--mtl_configs', default='configs/config_example.yml')
    parser.add_argument('--optimizer', default='adam')
    parser.add_argument('--eval_freq', type=int, default=5)
    parser.add_argument('--eval_num', type=int, default=None)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--lr-decay-cr', type=float, default=0.99)

    parser.add_argument('--backbone', default='resnet18')
    parser.add_argument('--backbone_pretrain', action='store_true')
    parser.add_argument('--backbone_dilated', action='store_false')
    parser.add_argument('--head', default='deeplab')

    parser.add_argument('--local_epochs', type=int, default=5)
    parser.add_argument('--comm_round', type=int, default=10,
                        help='how many round of communications we shoud use')
    parser.add_argument('--partition', default='homo')
    
    parser.add_argument('--seed', type=int, default=42, help='random seed. for reproducibility.')
    parser.add_argument('--root_dir', type=str, required=True, help='root dir of results')

    parser.add_argument('--aggregation', required=True)
    parser.add_argument('--aggregate_decoder', action='store_true')

    parser.add_argument('--cluster_num', type=int, default=2)
    parser.add_argument('--max_weight', type=float)
    parser.add_argument('--th', type=float)

    parser.add_argument('--save_all_ckpts', action='store_true')

    args = parser.parse_args()
    os.makedirs(args.root_dir, exist_ok=True)

    with open(args.mtl_configs, 'r') as stream:
        configs = yaml.safe_load(stream)
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    data_tools, default_mtl_configs = partition_data(configs['Dataset'], args)

    all_n_nets = sum([data_tools[dataname]['n_nets'] for dataname in data_tools])
    all_nets = {idx: {} for idx in range(all_n_nets)}
    net_idx = 0

    for data_idx, dataname in enumerate(data_tools):
        net_task_dataidx_map, n_nets = data_tools[dataname]['net_task_dataidx_map'], data_tools[dataname]['n_nets']
        nets = init_models(net_task_dataidx_map, n_nets, args, dataname)


        for worker_index in range(n_nets):
            task = net_task_dataidx_map[worker_index]['task']
            dataidxs = net_task_dataidx_map[worker_index]['dataidx']
            
            train_ds_local = get_train_dataset(dataname=dataname, tasks=[task], 
                transform=data_tools[dataname]['train_transforms'], dataidxs=dataidxs, overfit=False)
            train_dl_local = get_train_dataloader(configs=data_tools[dataname], ds=train_ds_local)

            all_nets[net_idx]['task'] = task
            all_nets[net_idx]['dataidxs'] = dataidxs
            all_nets[net_idx]['dataname'] = dataname
            all_nets[net_idx]['model'] = nets[worker_index]
            all_nets[net_idx]['train_dl'] = train_dl_local

            net_idx += 1
    
    test_dls = {}
    for dataname in data_tools:
        test_dls[dataname] = {}
        for task in set(data_tools[dataname]['task_list']):

            if args.eval_num is not None:
                print(f"Use {args.eval_num} data for evaluation....")
                dataidx = np.random.choice(NUM_TEST_IMAGES[dataname.lower()], args.eval_num, replace=False)
            else:
                print("Use the whole testset for evaluation....")
                dataidx = None
            
            test_ds_local = get_val_dataset(dataname=dataname, tasks=[task], 
                transform=data_tools[dataname]['val_transforms'], 
                dataidxs=dataidx, overfit=False)
            test_dl_local = get_val_dataloader(configs=data_tools[dataname], ds=test_ds_local)

            test_dls[dataname][task] = {'dl':test_dl_local, 'dataidx':dataidx}
    
    train_losses = {}
    for task in set([all_nets[net_idx]['task'] for net_idx in all_nets]):
        train_losses[task] = get_loss(default_mtl_configs, task)

    
    main(
        all_nets=all_nets,
        test_dls=test_dls,
        train_losses=train_losses,
        data_tools=data_tools,
        args=args,
        mtl_configs=default_mtl_configs
    )
        


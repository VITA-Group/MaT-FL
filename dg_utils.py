import os
import sys
import torch
from sklearn.cluster import AgglomerativeClustering

def get_aggregation_keys(all_keys, type='default'):
    if type == 'default':
        return list(filter(lambda x: 'backbone' in x and 'num' not in x, all_keys))
    elif type == 'exclude_bn':
        return list(filter(lambda x: 'backbone' in x and 'bn' not in x and 'downsample.1' not in x, all_keys))


def get_model_soup(ckpts, keys):

    model_soup = {}
    for key in keys:
        model_soup[key] = torch.mean(torch.stack([ckpt[key] for ckpt in ckpts.values()], dim=0), dim=0)
        
    return model_soup

def get_within_task_scores(all_nets):
    
    all_metadata = [all_nets[model_idx]['dataname'] + all_nets[model_idx]['task'] for model_idx in range(len(all_nets))]
    ref_dict = {x:ref_idx for ref_idx, x in enumerate(set(all_metadata))} 
    ref = torch.tensor([ref_dict[all_metadata[model_idx]] for model_idx in range(len(all_nets))]).cuda()
    scores = torch.eq(ref.view(-1,1), ref.view(1,-1)).float()
    scores = scores/scores.sum(dim=1, keepdim=True)

    return scores

def get_grouping_score(ckpts, keys, cluster_num):
    model_soup = get_model_soup(ckpts, keys)

    delta_list = []
    for key in keys:
        temp_delta = torch.stack([ckpt[key] for ckpt in ckpts.values()], dim=0) - model_soup[key]
        delta_list.append(temp_delta.reshape([len(temp_delta), -1]))

    delta = torch.cat(delta_list, dim=1)
    clustering = AgglomerativeClustering(n_clusters=cluster_num, affinity='cosine',linkage='average').fit(delta.cpu())
    print(clustering.labels_)
    cluster_results = torch.tensor(clustering.labels_).cuda()
    scores = torch.eq(cluster_results.view(-1,1), cluster_results.view(1,-1)).float()
    scores = scores/scores.sum(dim=1, keepdim=True)
    
    return scores

def get_soft_grouping_score(ckpts, keys, th=0):
    eps = 1e-8

    model_soup = get_model_soup(ckpts, keys)

    delta_list = []
    for key in keys:
        temp_delta = torch.stack([ckpt[key] for ckpt in ckpts.values()], dim=0) - model_soup[key]
        delta_list.append(temp_delta.reshape([len(temp_delta), -1]))

    delta = torch.cat(delta_list, dim=1)

    inner_prod = torch.matmul(delta, delta.T)
    norm = torch.linalg.norm(delta, ord=2, dim=1, keepdim=True)
    denom = torch.maximum(torch.matmul(norm, norm.T), eps*torch.ones(inner_prod.shape).cuda())
    # scores = torch.clamp(inner_prod / denom + 1, min=0.0)
    scores = torch.clamp(inner_prod / denom + th, min=0.0)

    scores = scores/scores.sum(dim=1, keepdim=True)
    print(scores)
    
    return scores


def aggregate(all_nets, save_ckpt, aggregation_type, root_dir, weight, cluster_num, th) -> dict:

    print(weight, cluster_num)

    if aggregation_type == 'single-model':
        # no aggregation involved
        return all_nets
    
    else:
        new_ckpt = {model_idx: all_nets[model_idx]['model'].state_dict() for model_idx in save_ckpt}
        name_keys = new_ckpt[0].keys()
    
        if aggregation_type in ['single-task', 'naive-avg', 'grouping', 'soft-grouping']: 
            aggregation_keys = get_aggregation_keys(name_keys)
            if aggregation_type == 'single-task':
                # no cross task aggregation
                scores = get_within_task_scores(all_nets)

            elif aggregation_type == 'naive-avg':
                # avg models from all datasets and tasks without grouping
                scores = torch.ones([len(save_ckpt), len(save_ckpt)]).float().cuda()
                scores = scores/scores.sum(dim=1, keepdim=True)
            
            elif aggregation_type == 'grouping':
                scores = get_grouping_score(new_ckpt, aggregation_keys, cluster_num)
            
            elif aggregation_type == 'soft-grouping':
                scores = get_soft_grouping_score(new_ckpt, aggregation_keys, th)
            
            else:
                assert False
        
            cos_sim_dict = {}
            for key in aggregation_keys:
                temp_weight = torch.stack([ckpt[key] for ckpt in new_ckpt.values()], dim=0)
                reshaped_weights = temp_weight.reshape(len(save_ckpt), -1)
                orig_shape = temp_weight.shape
                reweighted_weights = torch.matmul(scores, reshaped_weights).reshape(orig_shape)

                for model_idx in range(len(save_ckpt)):
                    new_ckpt[model_idx][key] = reweighted_weights[model_idx]

        
        elif aggregation_type == 'rws':
            aggregation_keys = get_aggregation_keys(name_keys, 'exclude_bn')
            print('Using weight : %.3f'%weight)

            # Warning: TEMP hard code
            scores_1 = torch.tensor([0.5, 0.5, 0.5, 0.5, 1, 1]).cuda()
            scores_2 = torch.tensor([0.5, 0.5, 1, 1]).cuda()
            weight_combine = torch.diag(torch.tensor([0.1*weight]*10)).cuda().float()

            for key in aggregation_keys:
                param_1 = torch.stack([new_ckpt[model_idx][key] for model_idx in range(0, 6)], dim=0)
                param_2 = torch.stack([new_ckpt[model_idx][key] for model_idx in range(6, 10)], dim=0)

                orig_shape = param_1.shape[1:]

                param_1 = param_1.reshape(len(param_1), -1)
                param_2 = param_2.reshape(len(param_2), -1)

                base_1 = torch.matmul(scores_1, param_1).squeeze()
                base_2 = torch.matmul(scores_2, param_2).squeeze()
                
                vectors_1 = get_projection(base_1, param_1)
                vectors_2 = get_projection(base_2, param_2)
                vectors = torch.cat([vectors_1, vectors_2], dim=0)
                print(key)
                print(torch.cosine_similarity(vectors.unsqueeze(0), vectors.unsqueeze(1), dim=2))

                vector_combinations = torch.sum(torch.matmul(weight_combine, vectors), dim=0).reshape(orig_shape)

                for model_idx in range(0, 6):
                    new_ckpt[model_idx][key] = vector_combinations + base_1.reshape(orig_shape)
                
                for model_idx in range(6, 10):
                    new_ckpt[model_idx][key] = vector_combinations + base_2.reshape(orig_shape)

        elif aggregation_type == 'stable-rws':
            aggregation_keys = get_aggregation_keys(name_keys, 'exclude_bn')
            print('Using weight : %.3f'%weight)

            # Warning: TEMP hard code
            scores_1 = torch.tensor([0.5, 0.5, 0.5, 0.5, 1, 1]).cuda()
            scores_2 = torch.tensor([0.5, 0.5, 1, 1]).cuda()

            soft_grouping_info = weight * get_ad_hoc_grouping_maritx()

            for key in aggregation_keys:
                param_1 = torch.stack([new_ckpt[model_idx][key] for model_idx in range(0, 6)], dim=0)
                param_2 = torch.stack([new_ckpt[model_idx][key] for model_idx in range(6, 10)], dim=0)

                orig_shape = param_1.shape[1:]

                param_1 = param_1.reshape(len(param_1), -1)
                param_2 = param_2.reshape(len(param_2), -1)

                base_1 = torch.matmul(scores_1, param_1).squeeze()
                base_2 = torch.matmul(scores_2, param_2).squeeze()
                
                vectors_1 = get_projection(base_1, param_1)
                vectors_2 = get_projection(base_2, param_2)
                vectors = torch.cat([vectors_1, vectors_2], dim=0)

                vector_combinations = torch.matmul(soft_grouping_info, vectors)

                for model_idx in range(0, 6):
                    new_ckpt[model_idx][key] = vector_combinations[model_idx].reshape(orig_shape) + base_1.reshape(orig_shape)
                
                for model_idx in range(6, 10):
                    new_ckpt[model_idx][key] = vector_combinations[model_idx].reshape(orig_shape) + base_2.reshape(orig_shape)

        
        elif aggregation_type == 'model-soup':
            aggregation_keys = get_aggregation_keys(name_keys, 'exclude_bn')
            starting_ckpt = save_ckpt[0]
            print('Using weight : %.3f'%weight)

            for key in aggregation_keys:
                temp_updates = torch.stack([ckpt[key] for ckpt in new_ckpt.values()], dim=0) - starting_ckpt[key]
                global_updates = torch.mean(temp_updates, dim=0) * weight
                for model_idx in range(len(save_ckpt)):
                    new_ckpt[model_idx][key] = global_updates + starting_ckpt[key]

        else:
            assert False


        for model_idx in range(len(save_ckpt)):
            all_nets[model_idx]['model'].load_state_dict(new_ckpt[model_idx]) 
        
        
        return all_nets


def get_projection(a, param):
    assert len(a.shape) == 1
    a = a.reshape(1, -1)
    try:
        projection_matrix = (a.T.matmul(a)) / (a.matmul(a.T))
        projections = projection_matrix.matmul(param.T).T
        return param - projections
    except:
        a = a / torch.norm(a)
        norm = torch.norm(param, dim=1).mul(torch.cosine_similarity(param, a, dim=1))
        projections = torch.diag(norm).matmul(a.repeat(len(param),1)) 
        return param - projections

def get_ad_hoc_grouping_maritx(th=0):
    data = torch.tensor([
        [ 1.0000,  0.9887, -0.9681, -0.9667, -0.7483, -0.9611,  0.5222,  0.6569,  0.5143, -0.7432],
        [ 0.9887,  1.0000, -0.9705, -0.9692, -0.7410, -0.9638,  0.5269,  0.6605,  0.5193, -0.7491],
        [-0.9681, -0.9705,  1.0000,  0.9897,  0.5860,  0.9859, -0.5322, -0.6497,  -0.5384,  0.7570],
        [-0.9667, -0.9692,  0.9897,  1.0000,  0.5790,  0.9874, -0.5318, -0.6488,  -0.5387,  0.7567],
        [-0.7483, -0.7410,  0.5860,  0.5790,  1.0000,  0.5531, -0.3301, -0.4706,  -0.2806,  0.4685],
        [-0.9611, -0.9638,  0.9859,  0.9874,  0.5531,  1.0000, -0.5307, -0.6458,  -0.5398,  0.7556],
        [ 0.5222,  0.5269, -0.5322, -0.5318, -0.3301, -0.5307,  1.0000,  0.5610,  0.2213, -0.6949],
        [ 0.6569,  0.6605, -0.6497, -0.6488, -0.4706, -0.6458,  0.5610,  1.0000,  0.3111, -0.7888],
        [ 0.5143,  0.5193, -0.5384, -0.5387, -0.2806, -0.5398,  0.2213,  0.3111,  1.0000, -0.7689],
        [-0.7432, -0.7491,  0.7570,  0.7567,  0.4685,  0.7556, -0.6949, -0.7888,  -0.7689,  1.0000]]).cuda().float()
    data = torch.clamp(data+th, min=0)
    data = data / data.sum(dim=1, keepdim=True)
    return data

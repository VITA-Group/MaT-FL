# Mat-FL
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

The official implementation of paper [Many-Task Federated Learning: A New Problem Setting and a Simple Baseline](https://openaccess.thecvf.com/content/CVPR2023W/FedVision/papers/Cai_Many-Task_Federated_Learning_A_New_Problem_Setting_and_a_Simple_CVPRW_2023_paper.pdf) [CVPRW 2023].

Ruisi Cai, Xiaohan Chen, Shiwei Liu, Jayanth Srinivasa, Myungjin Lee, Ramana Kompella, Zhangyang Wang

## Abstract
Federated Learning (FL) was originally proposed to effectively exploit more data that are distributed at local clients even though the local data follow non-i.i.d. distributions. The fundamental intuition is that, the more data we can use the better model we are likely to obtain in spite of the increased difficulty of learning due to the non-i.i.d. data distribution, or data heterogeneity. With this intuition, we strive to further scale up FL to cover more clients to participate and increase the effective coverage of more user data, by enabling FL to handle collaboration between clients that perform different yet related task types, i.e., enabling a new level of heterogeneity: task heterogeneity, which can be entangled with data heterogeneity and lead to more intractable clients. However, solving such compound heterogeneities from both data and task levels raises major challenges, against the current global, static, and identical federated aggregation ways across clients. To tackle this new and challenging FL setting, we propose an intuitive clustering-based training baseline to tackle the significant data and task heterogeneities. Specifically, each agent dynamically infers its "proximity" with others by comparing their layer-wise weight updates sent to the server, and then flexibly determines how to aggregate weights with selected similar clients. We construct new testbeds to examine our novel problem setting and algorithm on two benchmark datasets in multi-task learning: NYU Depth and PASCAL-Context datasets. Extensive experiments demonstrate that our proposed method shows superiority over plain FL algorithms such as FedAvg and FedProx in the 5-task setting on Pascal-Context and even enables jointly federated learning over the combined set of PASCAL-Context and NYU Depth (9 tasks, 2 data domains). 

## Citation
If you find this useful, please cite the following paper:
```
@InProceedings{Cai_2023_CVPR,
    author    = {Cai, Ruisi and Chen, Xiaohan and Liu, Shiwei and Srinivasa, Jayanth and Lee, Myungjin and Kompella, Ramana and Wang, Zhangyang},
    title     = {Many-Task Federated Learning: A New Problem Setting and a Simple Baseline},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2023},
    pages     = {5036-5044}
}
```
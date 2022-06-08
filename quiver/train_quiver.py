import argparse
import csv
import datetime
import os
import time
from pathlib import Path

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from torch_geometric.nn import SAGEConv
from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborSampler

####################
# Import Quiver
####################
import quiver

from graph_sage import GraphSAGE


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ogbn-products')
    parser.add_argument('--model', type=str, default='GraphSAGE')
    parser.add_argument('--gpus', type=int, default=4)
    parser.add_argument('--eval', action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def download_dataset(dataset_name):
    root = os.path.join(os.getcwd(), '../..', 'data', dataset_name)    # dataset storage location for now
    if dataset_name in ['ogbn-products', 'ogbn-papers100M']:
        return PygNodePropPredDataset(dataset_name, root)
    else:
        raise NotImplementedError(f'Dataset {dataset_name} not supported.')


def get_metrics_path():
    timestamp = int(datetime.datetime.now().timestamp())
    metrics_path = os.path.join('metrics', 'quiver', args.dataset, args.model, str(args.gpus))
    Path(metrics_path).mkdir(parents=True, exist_ok=True)
    return f'{metrics_path}/{timestamp}.csv'


def run(rank, world_size, quiver_sampler, quiver_feature, y, edge_index, split_idx, num_features, num_classes, eval, metrics_path):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

    torch.cuda.set_device(rank)
    device = torch.device('cuda:' + str(rank))

    train_idx, val_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']
    train_idx = train_idx.split(train_idx.size(0) // world_size)[rank].to(device)

    train_loader = torch.utils.data.DataLoader(train_idx, batch_size=1024, pin_memory=True)

    if eval and rank == 0:
        subgraph_loader = NeighborSampler(edge_index, node_idx=None,
                                          sizes=[-1], batch_size=1024,
                                          shuffle=False, num_workers=6)

    torch.manual_seed(12345)
    model = GraphSAGE(num_features, 256, num_classes, num_layers=3).to(device)
    model = DistributedDataParallel(model, device_ids=[device])
    optimizer = torch.optim.Adam(model.parameters())

    y = y.to(device)

    with open(metrics_path, 'a') as csv_file:
        metrics_writer = csv.writer(csv_file, delimiter=',')
        for epoch in range(1, 21):
            model.train()
            epoch_start = time.time()

            for seeds in train_loader:
                n_id, batch_size, adjs = quiver_sampler.sample(seeds)
                adjs = [adj.to(device) for adj in adjs]

                optimizer.zero_grad()
                out = model(quiver_feature[n_id], adjs)
                loss = F.nll_loss(out, y[n_id[:batch_size]])
                loss.backward()
                optimizer.step()

            epoch_time = time.time() - epoch_start
            memory = torch.cuda.max_memory_allocated() / 1_000_000
            print(f'Epoch: {epoch}, GPU {rank}: Loss {loss.item()}, GPU Memory: {memory} MB, Epoch Time: {epoch_time}')
            metrics_writer.writerow([rank, epoch, epoch_time])
            csv_file.flush()
            dist.barrier()

            # Evaluation on a single GPU
            if eval and rank == 0 and epoch % 5 == 0:  
                model.eval()
                with torch.no_grad():
                    out = model.module.inference(quiver_feature, device, subgraph_loader)
                res = out.argmax(dim=-1) == y.cpu()
                train_acc = int(res[train_idx].sum()) / train_idx.numel()
                val_acc = int(res[val_idx].sum()) / val_idx.numel()
                test_acc = int(res[test_idx].sum()) / test_idx.numel()
                print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')

            dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    args = parse_arguments()
    dataset = download_dataset(args.dataset)
    data = dataset[0]
    split_idx = dataset.get_idx_split()

    world_size = args.gpus
    metrics_path = get_metrics_path()
    
    ##############################
    # Create Sampler And Feature
    ##############################
    csr_topo = quiver.CSRTopo(data.edge_index)
    quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo, [15, 10, 5], 0, mode='GPU')
    feature = torch.zeros(data.x.shape)
    feature[:] = data.x
    quiver_feature = quiver.Feature(rank=0, device_list=list(range(world_size)), device_cache_size='2G', cache_policy='device_replicate', csr_topo=csr_topo)
    quiver_feature.from_cpu_tensor(feature)

    print('Let\'s use', world_size, 'GPUs!')
    mp.spawn(
        run,
        args=(world_size, quiver_sampler, quiver_feature, data.y.squeeze(), data.edge_index, split_idx, dataset.num_features, dataset.num_classes, args.eval, metrics_path),
        nprocs=world_size,
        join=True
    )
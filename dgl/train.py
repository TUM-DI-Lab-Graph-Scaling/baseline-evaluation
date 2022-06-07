import argparse
import csv
import datetime
import os
import time
from pathlib import Path

import dgl
import torch
import torch.nn.functional as F
import torchmetrics.functional as MF
from dgl import DGLError
from ogb.nodeproppred import DglNodePropPredDataset

from gat import GAT
from graph_sage import GraphSAGE


def download_dataset(dataset_name):
    global dataset
    if dataset_name == "ogbn-products":
        dataset = DglNodePropPredDataset(name="ogbn-products")
    elif dataset_name == "ogbn-papers100M":
        dataset = DglNodePropPredDataset(name="ogbn-papers100M")
    else:
        DGLError(f"Dataset {dataset_name} not supported.")

    graph, node_labels = dataset[0]
    graph = dgl.add_reverse_edges(graph)
    graph.ndata['label'] = node_labels[:, 0]

    node_features = graph.ndata['feat']
    num_features = node_features.shape[1]
    num_classes = (node_labels.max() + 1).item()

    idx_split = dataset.get_idx_split()
    train_nids = idx_split['train']
    valid_nids = idx_split['valid']
    test_nids = idx_split['test']

    return graph, num_features, num_classes, train_nids, valid_nids, test_nids


def run(proc_id, devices, graph, num_features, num_classes, train_nids, valid_nids, test_nids, metrics_file, args):
    dev_id = devices[proc_id]

    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(master_ip='127.0.0.1', master_port='12345')
    if torch.cuda.device_count() < 1:
        device = torch.device('cpu')
        torch.distributed.init_process_group(
            backend='gloo', init_method=dist_init_method, world_size=len(devices), rank=proc_id)
    else:
        torch.cuda.set_device(dev_id)
        device = torch.device('cuda:' + str(dev_id))
        torch.distributed.init_process_group(
            backend='nccl', init_method=dist_init_method, world_size=len(devices), rank=proc_id)

    train_nids = train_nids.to(device)
    valid_nids = valid_nids.to(device)

    global model
    if args.model == "GraphSAGE":
        model = GraphSAGE(num_features, 256, num_classes).to(device)
    elif args.model == "GAT":
        heads = ([8] * (args.num_layers - 1)) + [1]
        model = GAT(graph, 2, num_features, 8, num_classes, heads, .6, .6, 0.2, False)

    if device == torch.device('cpu'):
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=None, output_device=None)
    else:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], output_device=device)

    sampler = dgl.dataloading.NeighborSampler([15, 10, 5], prefetch_node_feats=["feat"], prefetch_labels=["label"])
    train_dataloader = dgl.dataloading.DataLoader(graph, train_nids, sampler, device=device, use_ddp=True,
                                                  batch_size=1024, shuffle=True, drop_last=False, num_workers=0,
                                                  use_uva=True)
    valid_dataloader = dgl.dataloading.DataLoader(graph, valid_nids, sampler, device=device, use_ddp=False,
                                                  batch_size=1024, shuffle=False, drop_last=False, num_workers=0,
                                                  use_uva=True)

    opt = torch.optim.Adam(model.parameters())

    best_accuracy = 0

    with open(metrics_file, 'a') as csv_file:
        metrics_writer = csv.writer(csv_file, delimiter=',')
        for epoch in range(20):
            model.train()
            start = time.time()

            for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
                inputs = blocks[0].srcdata['feat']
                labels = blocks[-1].dstdata['label']

                predictions = model(blocks, inputs)
                loss = F.cross_entropy(predictions, labels)

                opt.zero_grad()
                loss.backward()
                opt.step()

                if it % 20:
                    accuracy = MF.accuracy(predictions, labels)
                    mem = torch.cuda.max_memory_allocated() / 1000000
                    print(f"GPU {proc_id}: Loss {loss.item()} Accuracy {accuracy.item()} GPU Memory {mem} MB")

            end = time.time()
            epoch_time = end - start
            print(f"GPU {proc_id} Epoch {epoch} time: {epoch_time}")
            metrics_writer.writerow([proc_id, epoch, epoch_time])
            csv_file.flush()

            model.eval()

            if proc_id == 0:
                predictions = []
                labels = []
                for it, (input_nodes, output_nodes, blocks) in enumerate(valid_dataloader):
                    inputs = blocks[0].srcdata['feat']
                    labels.append(blocks[-1].dstdata['label'])
                    predictions.append(model(blocks, inputs).argmax(1))
                predictions = torch.cat(predictions)
                labels = torch.cat(labels)
                accuracy = MF.accuracy(predictions, labels)
                print(f"Epoch {epoch} Validation Accuracy: {accuracy}")
                if best_accuracy < accuracy:
                    best_accuracy = accuracy
                print(f"Best Validation Accuracy: {best_accuracy}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ogbn-products")
    parser.add_argument("--model", type=str, default="GraphSAGE")
    parser.add_argument("--gpus", type=int, default=4)
    args = parser.parse_args()

    graph, num_features, num_classes, train_nids, valid_nids, test_nids = download_dataset(args.dataset)

    timestamp = int(datetime.datetime.now().timestamp())
    metrics_path = f"metrics/dgl/{args.dataset}/{args.model}/{args.gpus}"
    Path(metrics_path).mkdir(parents=True, exist_ok=True)
    metrics_file = f"{metrics_path}/{timestamp}.csv"

    num_gpus = args.gpus
    import torch.multiprocessing as mp
    mp.spawn(run, args=(list(range(num_gpus)), graph, num_features, num_classes, train_nids, valid_nids, test_nids,
                        metrics_file, args), nprocs=num_gpus)


if __name__ == "__main__":
    main()


import argparse

import dgl
import numpy as np
import sklearn
import torch
import torch.nn.functional as F
import tqdm
from dgl import DGLError
from ogb.nodeproppred import DglNodePropPredDataset

from graph_sage import GraphSAGE
from gat import GAT


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


def run(proc_id, devices, graph, num_features, num_classes, train_nids, valid_nids, test_nids, args):
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

    sampler = dgl.dataloading.NeighborSampler([15, 10])
    train_dataloader = dgl.dataloading.DataLoader(graph, train_nids, sampler, device=device, use_ddp=True,
                                                  batch_size=1024, shuffle=True, drop_last=False, num_workers=0)
    valid_dataloader = dgl.dataloading.DataLoader(graph, valid_nids, sampler, device=device, use_ddp=False,
                                                  batch_size=1024, shuffle=False, drop_last=False, num_workers=0)

    opt = torch.optim.Adam(model.parameters())

    best_accuracy = 0

    for epoch in range(20):
        model.train()

        with tqdm.tqdm(train_dataloader) as tq:
            for step, (input_nodes, output_nodes, mfgs) in enumerate(tq):
                inputs = mfgs[0].srcdata['feat']
                labels = mfgs[-1].dstdata['label']

                predictions = model(mfgs, inputs)

                loss = F.cross_entropy(predictions, labels)
                opt.zero_grad()
                loss.backward()
                opt.step()

                accuracy = sklearn.metrics.accuracy_score(labels.cpu().numpy(),
                                                          predictions.argmax(1).detach().cpu().numpy())

                tq.set_postfix({'loss': '%.03f' % loss.item(), 'acc': '%.03f' % accuracy}, refresh=False)

        model.eval()

        if proc_id == 0:
            predictions = []
            labels = []
            with tqdm.tqdm(valid_dataloader) as tq, torch.no_grad():
                for input_nodes, output_nodes, mfgs in tq:
                    inputs = mfgs[0].srcdata['feat']
                    labels.append(mfgs[-1].dstdata['label'].cpu().numpy())
                    predictions.append(model(mfgs, inputs).argmax(1).cpu().numpy())
                predictions = np.concatenate(predictions)
                labels = np.concatenate(labels)
                accuracy = sklearn.metrics.accuracy_score(labels, predictions)
                print('Epoch {} Validation Accuracy {}'.format(epoch, accuracy))
                if best_accuracy < accuracy:
                    best_accuracy = accuracy

    print(f"Best accuracy: {best_accuracy}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ogbn-products")
    parser.add_argument("--model", type=str, default="GraphSAGE")
    parser.add_argument("--gpus", type=int, default=4)
    args = parser.parse_args()

    graph, num_features, num_classes, train_nids, valid_nids, test_nids = download_dataset(args.dataset)

    num_gpus = args.gpus
    import torch.multiprocessing as mp
    mp.spawn(run, args=(list(range(num_gpus)), graph, num_features, num_classes, train_nids, valid_nids, test_nids,
                        args), nprocs=num_gpus)


if __name__ == "__main__":
    main()


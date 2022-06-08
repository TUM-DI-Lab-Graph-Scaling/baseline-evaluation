import argparse
import os
import time
import torch
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from tqdm import tqdm

from torch_geometric.loader import NeighborSampler

from gat import GAT

import torch.multiprocessing as mp
import cProfile

def download_dataset(dataset_name):
    if dataset_name == 'ogbn-products':
        root = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'data', 'products')
        dataset = PygNodePropPredDataset('ogbn-products', root)
        evaluator = Evaluator(name='ogbn-products')
    elif dataset_name == 'ogbn-papers100M':
        root = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'data', 'papers100M')
        dataset = PygNodePropPredDataset('ogbn-papers100M', root)
        evaluator = Evaluator(name='ogbn-papers100M')
    else:
        raise NotImplementedError(f'Dataset {dataset_name} not supported.')
    return dataset, evaluator


def create_samplers(dataset):
    split_idx = dataset.get_idx_split()
    data = dataset[0]
    train_loader = NeighborSampler(data.edge_index, 
                                   node_idx=split_idx['train'],
                                   sizes=[15, 10],
                                   batch_size=1024,
                                   shuffle=True,
                                   num_workers=0)
    # subgraph_loader = NeighborSampler(data.edge_index,
    #                                 node_idx=None,
    #                                 sizes=[-1],
    #                                 batch_size=1024,
    #                                 shuffle=False,
    #                                 num_workers=0)
    return train_loader


def run(proc_id, devices, args, dataset, evaluator):

    # Initialize process
    dev_id = devices[proc_id]
    now = time.time()
    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train']
    train_idx = train_idx.split(train_idx.size(0) // len(devices))[proc_id]
    data = dataset[0]
    train_loader = NeighborSampler(data.edge_index, 
                                   node_idx=train_idx,
                                   sizes=[15, 10],
                                   batch_size=1024,
                                   shuffle=True,
                                   num_workers=0)
    print(dev_id,0, 'starting at ')
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
    
    # Initialize model
    global model
    
    model = GAT(dataset.num_features, 128, dataset.num_classes, num_layers=3,
            heads=4).to(device)
    
    # Initialize DistributedDataParallel
    if device == torch.device('cpu'):
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=None, output_device=None)
    else:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], output_device=device,find_unused_parameters=True)
    print(dev_id,time.time()-now, 'model prepared ')
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train']

    x = data.x.to(device)
    y = data.y.squeeze().to(device)
    print(dev_id,time.time()-now, 'data prepared ','x.shape',x.size(),'y.shape',y.size(),'data loader size',train_loader.sizes)
    optimizer = torch.optim.Adam(model.parameters())

    # Conduct training
    best_val_acc = 0
    for epoch in range(1):

        model.train()

        # pbar = tqdm(total=train_idx.size(0))
        # pbar.set_description(f'Epoch {epoch:02d}')

        total_loss = total_correct = 0
        num_batch = 0
        for batch_size, n_id, adjs in train_loader:
            # 'adjs' holds a list of '(edge_index, e_id, size)' tuples.
            adjs = [adj.to(device) for adj in adjs]
            
            optimizer.zero_grad()
            out = model(x[n_id], adjs)
            loss = F.nll_loss(out, y[n_id[:batch_size]])
            loss.backward()
            optimizer.step()

            total_loss += float(loss)
            total_correct += int(out.argmax(dim=-1).eq(y[n_id[:batch_size]]).sum())
            if num_batch == 0:
                print(dev_id,time.time()-now,'finished batch size', batch_size)
            num_batch += 1
        #     pbar.update(batch_size)

        # pbar.close()
        print(dev_id,time.time()-now,'finished epoch, num of batch', num_batch)
        loss = total_loss / len(train_loader)
        acc = total_correct / train_idx.size(0)
        print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Training Accuracy: {acc:.4f}')
        
        # if proc_id == 0:
        #     with torch.no_grad():
        #         model.eval()

        #         out = model.module.inference(x, subgraph_loader, device)

        #         y_true = y.cpu().unsqueeze(-1)
        #         y_pred = out.argmax(dim=-1, keepdim=True)

        #         val_acc = evaluator.eval({
        #             'y_true': y_true[split_idx['valid']],
        #             'y_pred': y_pred[split_idx['valid']],
        #         })['acc']
        #     print(f'Validation Accuracy: {val_acc:.4f}')
        #     if val_acc > best_val_acc:
        #         best_val_acc = val_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ogbn-products')
    parser.add_argument('--gpus', type=int, default=4)
    args = parser.parse_args()

    dataset, evaluator = download_dataset(args.dataset)
    # train_loader = create_samplers(dataset)

    # import pdb;pdb.set_trace()
    mp.spawn(run, args=(list(range(args.gpus)), args, dataset, evaluator), nprocs=args.gpus)
    # cProfile.run('worker()')


if __name__ == '__main__':
   main()

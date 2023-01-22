from __future__ import division

import time

import torch
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Used_memory = 0
record_time = 0
avg_run_time = 0


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def random_planetoid_splits(data, num_classes):
    # Set new random planetoid splits:
    # * 20 * num_classes labels for training
    # * 500 labels for validation
    # * 1000 labels for testing

    indices = []
    for i in range(num_classes):
        index = (data.y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:20] for i in indices], dim=0)

    rest_index = torch.cat([i[20:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(rest_index[:500], size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index[500:1500], size=data.num_nodes)

    return data


def run(dataset, model, runs, epochs, lr, weight_decay, early_stopping,
        permute_masks=None, logger=None):

    val_losses, accs, durations = [], [], []
    data = dataset[0]
    if permute_masks is not None:
        data = permute_masks(data, dataset.num_classes)
    data = data.to(device)

    model.to(device).reset_parameters()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if torch.cuda.is_available():
        torch.cuda.synchronize()


    best_val_loss = float('inf')
    test_acc = 0
    val_loss_history = []

    dur = 0
    for epoch in range(1, epochs + 1):
        t_start = time.perf_counter()
        train(model, optimizer, data, epoch)
        t_end = time.perf_counter()
       	dur += t_end - t_start
        eval_info = evaluate(model, data)
        eval_info['epoch'] = epoch

        if logger is not None:
            logger(eval_info)

        if eval_info['val_loss'] < best_val_loss:
            best_val_loss = eval_info['val_loss']
            test_acc = eval_info['test_acc']

        val_loss_history.append(eval_info['val_loss'])
        if early_stopping > 0 and epoch > epochs // 2:
            tmp = tensor(val_loss_history[-(early_stopping + 1):-1])
            if eval_info['val_loss'] > tmp.mean().item():
                break
        

    if torch.cuda.is_available():
        torch.cuda.synchronize()


    val_losses.append(best_val_loss)
    accs.append(test_acc)
    durations.append(dur/epochs)

    loss, acc, duration = tensor(val_losses), tensor(accs), tensor(durations)

    print('Val Loss: {:.4f}, Test Accuracy: {:.3f} ± {:.3f}, Avg Duration: {:.3f}'.
          format(loss.mean().item(),
                 acc.mean().item(),
                 acc.std().item(),
                 duration.mean().item()))
    
    #OUTPUT we need
    global Used_memory
    global avg_run_time
    global record_time
    avg_run_time = avg_run_time *1. / record_time
    Used_memory /= (1024**3)
    print('^^^{:6f}^^^{:6f}'.format(Used_memory, avg_run_time))

def train(model, optimizer, data, epoch):
    global Used_memory
    global avg_run_time
    global record_time
    torch.cuda.synchronize()
    t_start = time.perf_counter()

    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

    now_mem = torch.cuda.memory_allocated(0)
    Used_memory = max(now_mem, Used_memory)
    print('now_mem : ', now_mem, 'max_mem', Used_memory)

    loss.backward()
    optimizer.step()

    torch.cuda.synchronize()
    t_end = time.perf_counter()
    print('Epoch time', t_end - t_start, 's')
    if epoch >=3:
        record_time += 1
        avg_run_time += t_end-t_start



def evaluate(model, data):
    model.eval()

    with torch.no_grad():
        logits = model(data)

    outs = {}
    for key in ['train', 'val', 'test']:
        mask = data['{}_mask'.format(key)]
        loss = F.nll_loss(logits[mask], data.y[mask]).item()
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

        outs['{}_loss'.format(key)] = loss
        outs['{}_acc'.format(key)] = acc

    return outs

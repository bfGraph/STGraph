import time
import traceback

import numpy as np
import pynvml
import torch
import torch.nn.functional as F
from rich.progress import Progress

from stgraph.utils import DataTable
from stgraph.dataset import CoraDataLoader
from stgraph.graph.static.static_graph import StaticGraph
from .model import GCN
from .utils import accuracy, generate_test_mask, generate_train_mask, \
    to_default_device


def train(
    dataset: str,
    lr: float,
    num_epochs: int,
    num_hidden: int,
    num_layers: int,
    weight_decay: float,
    self_loops: bool,
    output_file_path: str,
) -> int:
    with open(output_file_path, "w") as f:
        if torch.cuda.is_available():
            print("🎉 CUDA is available", file=f)
        else:
            print("😔 CUDA is not available", file=f)
            return 1

        tmp = StaticGraph([(0, 0)], [1], 1)

        if dataset == "Cora":
            dataloader = CoraDataLoader()
        else:
            print("😔 Unrecognized dataset", file=f)
            return 1

        features = torch.FloatTensor(dataloader.get_all_features())
        labels = torch.LongTensor(dataloader.get_all_targets())

        train_mask = generate_train_mask(len(features), 0.6)
        test_mask = generate_test_mask(len(features), 0.6)

        train_mask = torch.BoolTensor(train_mask)
        test_mask = torch.BoolTensor(test_mask)

        cuda = True
        torch.cuda.set_device(0)
        features = features.cuda()
        labels = labels.cuda()
        train_mask = train_mask.cuda()
        test_mask = test_mask.cuda()
        edge_weight = [1 for _ in range(len(dataloader.get_edges()))]

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        initial_used_gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(handle).used
        g = StaticGraph(dataloader.get_edges(), edge_weight, features.shape[0])
        graph_mem = pynvml.nvmlDeviceGetMemoryInfo(handle).used - initial_used_gpu_mem

        degs = torch.from_numpy(g.weighted_in_degrees()).type(torch.int32)
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0
        norm = to_default_device(norm)
        g.set_ndata("norm", norm.unsqueeze(1))

        num_feats = features.shape[1]
        n_classes = int(max(labels) - min(labels) + 1)

        model = GCN(g, num_feats, num_hidden, n_classes, num_layers, F.relu)
        model.cuda()

        loss_fcn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        dur = []
        Used_memory = 0
        table = DataTable(
            f"STGraph GCN on {dataloader.name} dataset",
            ["Epoch", "Time(s)", "Train Accuracy", "Used GPU Memory (Max MB)"],
        )

        try:
            with Progress() as progress:
                epoch_progress = progress.add_task(
                    f"{dataset}",
                    total=num_epochs
                )
                for epoch in range(num_epochs):
                    torch.cuda.reset_peak_memory_stats(0)
                    model.train()
                    if cuda:
                        torch.cuda.synchronize()
                    t0 = time.time()

                    # forward
                    logits = model(g, features)
                    loss = loss_fcn(logits[train_mask], labels[train_mask])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    now_mem = torch.cuda.max_memory_allocated(0) + graph_mem
                    Used_memory = max(now_mem, Used_memory)

                    if cuda:
                        torch.cuda.synchronize()

                    run_time_this_epoch = time.time() - t0

                    if epoch >= 3:
                        dur.append(run_time_this_epoch)

                    train_acc = accuracy(logits[train_mask], labels[train_mask])
                    table.add_row(
                        [epoch, run_time_this_epoch, train_acc, (now_mem * 1.0 / (1024**2))]
                    )

                    progress.update(epoch_progress, advance=1)

            table.display(output_file=f)
            print("Average Time taken: {:6f}".format(np.mean(dur)), file=f)
            return 0

        except Exception as e:
            print("---------------- Error ----------------\n", file=f)
            print(e, file=f)
            print("\n", file=f)

            traceback.print_exc(file=f)
            print("\n", file=f)

            if "out of memory" in str(e):
                table.add_row(["OOM", "OOM", "OOM", "OOM"])
                table.display(output_file=f)

            return 1

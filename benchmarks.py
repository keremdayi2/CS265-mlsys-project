import importlib
from typing import Any, Dict, List
import data_utils
import tabulate

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.fx as fx
from torch.testing._internal.distributed._tensor.common_dtensor import (
    ModelArgs,
    Transformer,
    )
from torchvision.models import resnet18, resnet50
from graph_prof import GraphProfiler
from graph_tracer import SEPFunction, compile
from recompute import RecomputePolicy

from activation_checkpoint import *

import sys
import argparse

model_names: List[str] = [
    "Transformer",
    "Resnet18",
    "Resnet50",
]

model_batch_sizes: Dict[str, List[int]] = {
    "Transformer": [256, 512, 1024, 2048],
    "Resnet18": [16, 32, 64, 128],
    "Resnet50": [16, 32, 64, 128, 256],
}

class Experiment:
    def __init__(self, model_name: str, batch_size: int, extra_args={}):
        assert model_name in model_names, f"Model {model_name} not found in model names {model_names}"
        dev = torch.device("cuda")
        self.model_name = model_name
        self.batch_size = batch_size
        
        if 'memory_cap' in extra_args.keys():
            self.memory_cap = extra_args['memory_cap'] * 1e9
        else:
            self.memory_cap = 40 * 1e9

        if self.model_name == "Transformer":

            vocab_size = 2048
            bsz, seq_len = self.batch_size, 256
            with torch.device(dev):
                model_args = ModelArgs(
                    n_layers=8,
                    n_heads=4,
                    vocab_size=vocab_size,
                    max_seq_len=seq_len,
                    dropout_p=0.1,
                )
                self.model = Transformer(model_args)
            src = torch.randint(0, vocab_size, (bsz, seq_len), device=dev)
            tgt = torch.randint(0, vocab_size, (bsz, seq_len), device=dev)
            self.example_inputs = (src, tgt)

            def transformer_train_step(
                model: nn.Module, optim: optim.Optimizer, example_inputs: Any
            ):
                loss = self.loss_fn(model(example_inputs[0]), example_inputs[1])
                loss = SEPFunction.apply(loss)
                loss.backward()
                optim.step()
                optim.zero_grad()

            self.train_step = transformer_train_step
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-2, foreach=True, capturable=True)

        elif self.model_name in ["Resnet18", "Resnet50"]:
            inp = torch.randn(self.batch_size, 3, 224, 224, device=dev)
            num_classes = 10
            target = torch.randint(0, num_classes, (self.batch_size,), device=dev)
            self.example_inputs = (inp, target)
            with torch.device(dev):
                self.model = resnet18() if self.model_name == "Resnet18" else resnet50()

            def resnet_train_step(
                model: nn.Module, optim: optim.Optimizer, example_inputs: Any
            ):
                loss = self.loss_fn(model(example_inputs[0]), example_inputs[1])
                loss = SEPFunction.apply(loss)
                loss.backward()
                optim.step()
                optim.zero_grad()

            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-2, foreach=True, capturable=True)
            self.train_step = resnet_train_step

    def loss_fn(self, logits: torch.Tensor, targets: torch.Tensor):
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
        )

    def init_opt_states(self):
        for param in self.model.parameters():
            if param.requires_grad:
                param.grad = torch.rand_like(param)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def graph_transformation(self, gm: fx.GraphModule, args: Any) -> fx.GraphModule:
        print(gm.graph.print_tabular())
        warm_up_iters, profile_iters = 4, 4
        graph_profiler = GraphProfiler(gm)

        with torch.no_grad():
            for _ in range(warm_up_iters):
                graph_profiler.run(*args)
            graph_profiler.reset_stats()

            for _ in range(profile_iters):
                graph_profiler.run(*args)
            graph_profiler.aggregate_stats()
            graph_profiler.print_stats(f'{model_name}_{batch_size}_pre')

        # create recompute policy here
        recompute_policy = RecomputePolicy([stats for name, stats in graph_profiler.name_to_stats.items()], graph_profiler.name_to_node)
        recomputation_list = recompute_policy.get_recomputation(self.memory_cap)

        sys.stderr.write(f'Number of nodes to recompute {len(recomputation_list)}\n')
        gm = activation_checkpointing(gm, recomputation_list)

        # # do some check
        # recomputation_names = [rp.name for rp in recomputation_list]
        
        # for node in gm.graph.nodes:
        #     if node.name in recomputation_names:
        #         pass
            
        recompute_profiler = GraphProfiler(gm)

        # re-run the profiler to gather statistics on the new computational graph
        with torch.no_grad():
            for _ in range(warm_up_iters):
                recompute_profiler.run(*args)
            recompute_profiler.reset_stats()

            for _ in range(profile_iters):
                   recompute_profiler.run(*args)
            recompute_profiler.aggregate_stats()
            recompute_profiler.print_stats(f'{model_name}_{batch_size}_post')
    
        return gm

    def run(self):
        self.train_step(self.model, self.optimizer, self.example_inputs)
        print("Successful.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmarks")
    parser.add_argument("--model_idx", type=int, required=True, help="Model index to run")
    parser.add_argument("--batch_idx", type=int, required=True, help="Model index to run")
    parser.add_argument("--mem_cap", type=float, required=True, help="Model index to run")

    args = parser.parse_args()

    model_idx = args.model_idx
    batch_idx = args.batch_idx
    memory_cap = args.mem_cap

    # parse experiment arguments
    model_name = model_names[model_idx]
    batch_size = model_batch_sizes[model_name][batch_idx]
    sys.stderr.write(f'Running experiment. Model: {model_name}, Batch size: {batch_size} Memory cap: {memory_cap}\n')

    exp = Experiment(model_name, batch_size, {'memory_cap' : memory_cap})

    exp.init_opt_states()
    compiled_fn = compile(exp.train_step, exp.graph_transformation)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # test model to get sample output
    compiled_fn(exp.model, exp.optimizer, exp.example_inputs)
    max_memory = torch.cuda.max_memory_allocated()
    sys.stderr.write(f'max_memory: {max_memory/1e9:0.2f}GB\n')

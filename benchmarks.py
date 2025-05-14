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

import sys
import argparse

model_names: List[str] = [
    "Transformer",
    "Resnet18",
    "Resnet50",
]

model_batch_sizes: Dict[str, int] = {
    "Transformer": 4,
    "Resnet18": 16,
    "Resnet50": 4,
}


class Experiment:
    def __init__(self, model_name: str, batch_size: int, extra_args=[]):
        assert model_name in model_names, f"Model {model_name} not found in model names {model_names}"
        dev = torch.device("cuda")
        self.model_name = model_name
        self.batch_size = batch_size

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
        warm_up_iters, profile_iters = 2, 3
        graph_profiler = GraphProfiler(gm)

        with torch.no_grad():
            for _ in range(warm_up_iters):
                graph_profiler.run(*args)
            graph_profiler.reset_stats()

            for _ in range(profile_iters):
                graph_profiler.run(*args)
            graph_profiler.aggregate_stats()
            graph_profiler.print_stats()

            # create recompute policy here
            recompute_policy = RecomputePolicy([stats for name, stats in graph_profiler.name_to_stats.items()])
            recomputation_list = recompute_policy.get_recomputation(0.65*1e9)

            obj_list = data_utils.obj_list_to_array(list(recomputation_list))
            maxcolwidths = [12] * len(obj_list[0])
            print(tabulate.tabulate(obj_list, tablefmt="grid", maxcolwidths = maxcolwidths, floatfmt=".2f"))



        return gm

    def run(self):
        self.train_step(self.model, self.optimizer, self.example_inputs)
        print("Successful.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmarks")
    parser.add_argument("--model_idx", type=int, required=True, help="Model index to run")

    args = parser.parse_args()

    model_idx = args.model_idx

    model_name = model_names[model_idx]
    sys.stderr.write(f'{model_name}\n')
    
    exp = Experiment(model_name, model_batch_sizes[model_name])


    exp.init_opt_states()
    compiled_fn = compile(exp.train_step, exp.graph_transformation)
    compiled_fn(exp.model, exp.optimizer, exp.example_inputs)

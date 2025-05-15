import logging
import os
from functools import wraps
from typing import Any
import sys


import torch
import torch.fx as fx
import torch.multiprocessing as mp
import torch.nn as nn

from graph_prof import GraphProfiler
from graph_tracer import SEPFunction, compile

from activation_checkpoint import *
from recompute import *

# This is the dummy model that is for use in starter code. But we will
# experiment with Resnet and Transformer model.


class DummyModel(nn.Module):
    def __init__(self, layers: int, dim: int):
        super().__init__()
        modules = []
        for _ in range(layers):
            modules.extend([nn.Linear(dim, dim), nn.ReLU()])
        self.mod = nn.Sequential(*modules)

    def forward(self, x):
        return self.mod(x)


# We wrap the loss with a separator function to call a
# dummy function 'SEPFunction', which is the separator function, that will call
# an identity operator at the end of the forward pass. This identity operator
# will get recorded in the computational graph and will inform you where the
# backward pass ends.


# This is the train_step function that takes in a model, optimizer and an input
# mini batch and calls the forward pass, loss function and the optimizer step. A
# computational graph corresponding to a train_step will be captured by the
# compiler.


def train_step(
    model: torch.nn.Module, optim: torch.optim.Optimizer, batch: torch.Tensor
):
    loss =  model(batch).sum()
    loss = SEPFunction.apply(loss)
    loss.backward()
    optim.step()
    optim.zero_grad()


# Below is a user defined function that accepts a graph module and arguments of
# used to run the graph. You can essentially do any operation, graph
# modification, profiling etc. inside this function. Subsequent to modifications
# or graph analysis, the function expects you to return the modified graph back.
# In the given example, we just print the graph, and then initilize the graph
# profiler. The graph profiler extends the class fx.Interpreter, that allows you
# to run the graph node by node, more explanation in graph_prof.py.


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
    recomputation_list = recompute_policy.get_recomputation(0.7*1e9)

    sys.stderr.write(f'{recomputation_list[0]}\n')
    print('Recompute data')
    print(f'Number of nodes {len(recomputation_list)}')

    gm = activation_checkpointing(gm, recomputation_list)

    return gm


# We first initialize the model, pass it to the wrapper model, then create a
# random input mini-batch and initilize the optimizer. We then call the compile
# function that takes in two arguments, a train_step function and a
# graph_transformation function. The train_step function is the one that will be
# traced by the compiler and a computational graph for the same will be created.
# This computational graph is then passed to the graph_transformation function
# to do any graph profiling, modifications and optimizations. This modified
# graph is stored and will be returned as the compiled function. In essence we
# do the following inside the compile function:

# def compile (train_step, graph_transformation):
#     @wraps(train_step)
#     def inner(*args, **kwargs):
#         if not_compiled:
#             original_graph, input_args = graph_tracer(train_step)
#             modified_graph = graph_transformation(original_graph, input_args)
#         output = modified_graph(*args, **kwargs)
#         return output
#     return inner


def experiment():
    sys.stderr.write("Running starter_code.py\n")

    logging.getLogger().setLevel(logging.DEBUG)
    torch.manual_seed(20)
    batch_size = 1000
    layers = 4
    dim = 100
    num_iters = 5

    device_str = 'cuda:0'
    model = DummyModel(dim=dim, layers=layers).to(device_str)
    batch = torch.randn(batch_size, dim).to(device_str)
    optim = torch.optim.Adam(
        model.parameters(), lr=0.01, 
         foreach = True,
         # fused=True,
         capturable=True
        )

    for param in model.parameters():
        if param.requires_grad:
            param.grad = torch.rand_like(param, device=device_str)

    optim.step()
    optim.zero_grad()

    compiled_fn = compile(train_step, graph_transformation)
    compiled_fn(model, optim, batch)


if __name__ == "__main__":
    experiment()

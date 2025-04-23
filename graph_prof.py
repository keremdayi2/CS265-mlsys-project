from enum import Enum
from typing import Dict
import torch
import torch.fx as fx
from typing import Dict, Any

import sys
from tabulate import tabulate


class OP(str, Enum):
    CALL_FUNCTION = "call_function"
    CALL_MODULE = "call_module"
    CALL_METHOD = "call_method"
    GET_ATTR = "get_attr"
    OUTPUT = "output"
    PLACEHOLDER = "placeholder"


class NodeType(Enum):
    """
    NodeType is a enum that records the type of the tensors in the graph.
    """

    PARAM = 0
    ACT = 1
    ACT_DISCARD = 2
    GRAD = 3
    GRAD_INTERMEDIATE = 4
    OPT_STATE = 5
    OTHER = 6


# This is an example graph_profiler that extends the fx.Interpreter class, it
# will perform graph execution by running the graph node by node.


class GraphProfiler(fx.Interpreter):
    def __init__(self, module: fx.GraphModule, garbage_collect_values: bool = True):
        super().__init__(module, garbage_collect_values)

        # You should perform the static analysis of the graph here. In
        # particular you might want to find the intermediate
        # nodes/activations/feature_maps in the graph that will be defined as
        # those nodes which are not parameters (not placeholder node types) but
        # are created during the forward pass and are also used in the backward
        # pass for computation.

        # The boundary between the forward pass and backward pass can be
        # identified by locating the node '%sep : [num_users=1] =
        # call_function[target=torch.ops.separator.sep.default]' which will
        # define the end of the forward pass. You will see the loss function
        # after thsi operation and then you will encounter a node named,
        # '%sep_backward : [num_users=1] =
        # call_function[target=torch.ops.separator.sep_backward.default]'. This
        # node marks the beginning of the backward pass.

        # For these intermediate nodes in the graph, you will record their last
        # use in the forward pass and their first use in the backward pass.

        # The parameters of the models are the placeholder (input) nodes of the
        # graph. Note that not all the placeholder nodes of the graph are
        # parameters. The optimizer's states and the input mini-batch are also
        # placeholder nodes that given as inputs to the graph.

        # The parameters and gradients of the model can be otained using the
        # optimizer node's arguments. The optimizer node can be identified by
        # the node '%_fused_adam : [num_users=3] =
        # call_function[target=torch.ops.aten._fused_adam.default]'.
        # The argument at position 0 is the list of parameter nodes, while the
        # argument at position 1 is the list of gradient nodes.

        # Printing the input nodes, node users and node names.
        self.name_to_node = {}
        self.name_to_rank = {}

        # The nodes in the graph are stored in a dictionary. The key is the
        # dictionaries of each run
        self.name_to_size = {}
        self.name_to_runtime = {}
        self.name_to_result_ptrs = {}

        self.sep_rank = None # used to determine where the forward pass ends
        self.sep_backward_rank = None

        self.op_start_rank = None

        self.name_to_cuda_memory = {}
        self.name_to_predicted_memory = {}

        # here we 
        # 1) find where the SEP operator appears
        # 2) set the ranks
        # 3) find the parameters and gradients as well
        # 4) initalize name_to_node mapping
        for rank, node in enumerate(self.module.graph.nodes):
            if node.target == torch.ops.separator.sep.default:
                self.sep_rank = rank

            if node.target == torch.ops.separator.sep_backward.default:
                self.sep_backward_rank = rank

            if self.op_start_rank is None and node.op == 'call_function':
                self.op_start_rank = rank

            # _foreach_lerp_ is linear interpolation which helps us identify the gradients and optimizer states.
            if node.target == torch.ops.aten._foreach_lerp_.Scalar: 
                opt_states = node.args[0]
                grads = node.args[1]
                
                # sys.stderr.write(f'Momentum term: {opt_states}\n')
                # sys.stderr.write(f'Gradients: {grads}\n')

                self.grad_name = [g.name for g in grads]

            # this is the final adam step which updates the params
            if node.target == torch.ops.aten._foreach_addcdiv.Scalar:
                params = node.args[0] # first argument is the parameters that are updated.
                self.param_name = [p.name for p in params]

            self.name_to_rank[node.name] = rank
            self.name_to_node[node.name] = node
            self.name_to_size[node.name] = []
            self.name_to_runtime[node.name] = []

        # end of calculation of gradients which can be used to 
        # tag the beginning of the optimizer stage
        g_end = max([self.name_to_rank[m] for m in self.grad_name])
        self.optimizer_start_rank = g_end + 1

        # find first and last uses during forward and backward passes
        self.first_forward, \
            self.last_forward, \
                self.first_backward, \
                    self.last_backward = self._find_first_last_use() 

        assert self.param_name != None, "Could not find params"
        assert self.grad_name != None, "Could not find grads"

        sys.stderr.write(f'Gradients: {self.grad_name}\n')
        sys.stderr.write(f'Params: {self.param_name}\n')

        # finally tag all node types as 
        # PARAM, ACT, GRAD, GRAD_INTERMEDIATE
        self.name_to_nodetype = self._tag_node_types()

        # del self.name_to_result_ptrs

        # to free all the fake allocations
        torch.cuda.empty_cache()

    # return the first/last forward uses and first/last backward uses of all nodes
    # returns 4 dictionaries corresponding to these. keys are names of nodes
    def _find_first_last_use(self):
        keys = self.name_to_node.keys()

        first_forward = dict.fromkeys(keys)
        last_forward = dict.fromkeys(keys)

        first_backward = dict.fromkeys(keys)
        last_backward = dict.fromkeys(keys)

        # sys.stderr.write(f'First forward initialized: {first_forward}\n')
        for n in self.module.graph.nodes:
            name = n.name
            users = n.users 
            users_name = [m.name for m in users]

            # find forward users based on appearing before SEP operator
            forward_users = list(filter(
                    lambda x: self.name_to_rank[x] <= self.sep_rank, 
                    users_name
                    ))

            # sort according to rank
            if len(forward_users) > 0: # non-zero forward uses
                forward_users = sorted(forward_users, key=lambda x: self.name_to_rank[x])
                first_forward[name] = forward_users[0]
                last_forward[name] = forward_users[-1]

            # find backward users
            backward_users = list(filter(
                    lambda x: self.name_to_rank[x] > self.sep_rank, 
                    users_name
                    ))

            if len(backward_users) > 0:
                backward_users = sorted(backward_users, key=lambda x: self.name_to_rank[x])
                first_backward[name] = backward_users[0]
                last_backward[name] = backward_users[-1]

        # out = sys.stderr
        out = sys.stdout

        out.write(f'First forward: {first_forward}\n')
        out.write(f'Last forward: {last_forward}\n')
        out.write(f'First backward: {first_backward}\n')
        out.write(f'Last backward: {last_backward}\n')

        return first_forward, last_forward, first_backward, last_backward

    # TODO: implement
    def _tag_node_types(self):
        op_types = {}

        # initially tag everything as other
        for n in self.module.graph.nodes:
            op_types[n.name] = NodeType.OTHER

        # set the params and the gradients
        for name in self.param_name:
            op_types[name] = NodeType.PARAM

        for name in self.grad_name:
            op_types[name] = NodeType.GRAD

        # set the activations
        for n in self.module.graph.nodes:
            name = n.name
            rank = self.name_to_rank[name]

            if rank >= self.op_start_rank and rank <= self.sep_rank:
                if op_types[name] == NodeType.OTHER:
                    # if this will be reused in the backward pass 
                    # those are what we need
                    if self.first_backward[name] is None:
                        op_types[name] = NodeType.ACT_DISCARD 
                    else:
                        op_types[name] = NodeType.ACT

        # set the intermediate gradients
        for n in self.module.graph.nodes:
            name = n.name
            rank = self.name_to_rank[name]

            if rank >= self.sep_backward_rank and rank < self.optimizer_start_rank:
                # if this is not a main gradient, set to intermediate
                if op_types[name] == NodeType.OTHER:
                    op_types[name] = NodeType.GRAD_INTERMEDIATE

        return op_types

    def _is_iterable_type(self, obj):
        iterable_types = [list, tuple]
        return any(isinstance(obj, tp) for tp in iterable_types)

    # do a BFS search in order to retrieve the tensors in the list
    def _unpack_nodes(self, lst):
        result_tensors = []

        Q = [a for a in lst]

        result_tensors = []

        while len(Q) > 0:
            a = Q.pop(0)

            if isinstance(a, fx.Node):
                result_tensors.append(a)
            elif isinstance(a, (list, tuple)):
                for e in a:
                    Q.append(e)
            elif isinstance(a, (int, bool, float)) or a is None:
                # we can ignore these
                continue
            else:
                sys.stderr.write(f'Got unhandled type while unpacking {type(a)}\n')

        return result_tensors

    def _unpack_tensors(self, lst):
        result_tensors = []

        Q = [a for a in lst]

        result_tensors = []

        while len(Q) > 0:
            a = Q.pop(0)

            if isinstance(a, torch.Tensor):
                result_tensors.append(a)
            elif isinstance(a, (list, tuple)):
                for e in a:
                    Q.append(e)
            elif isinstance(a, (int, bool, float)) or a is None:
                # we can ignore these
                continue
            else:
                sys.stderr.write(f'Got unhandled type while unpacking {type(a)}\n')

        return result_tensors

    def _get_memory_usage(self, n:fx.Node, result : Any) -> int:
        # we can check whether new memory was allocated using the following pattern
        # x.storage().data_ptr() == y.storage().data_ptr()
        # if no new-memory was allocated, we can set memory usage to 0

        args = n.args
        size_bytes = 0

        # TODO: Add minimum pytorch allocation

        if n.op == OP.CALL_FUNCTION:
            result_unpacked = self._unpack_tensors([result])
            arg_unpacked = self._unpack_nodes(args) 

            # by this point, all elements are guaranteed to be torch.tensors
            # we can check if there is any data_ptr shared between result and args
            arg_ptrs = []
            for a in arg_unpacked:
                arg_ptrs += self.name_to_result_ptrs[a.name]
            arg_ptrs = set(arg_ptrs)

            # sys.stderr.write(f'{arg_ptrs}\n')

            size_bytes = 0
            for i, r in enumerate(result_unpacked):
                if not r.storage().data_ptr() in arg_ptrs:
                    # means this memory is new allocated
                    size_bytes += r.nelement() * r.element_size()
                # else:
                #     sys.stderr.write(f'Found overlapping memory in {n.name}\n')


        elif n.op == OP.PLACEHOLDER:
            if isinstance(result, torch.Tensor):
                size_bytes = result.nelement() * result.element_size()
            else:
                sys.stderr.write(f'{n.name}: got unhandled placeholder. Got {type(result)}\n')
        else:
            sys.stderr.write(f'{n.name}: got unhandled operation. Got {n.op}\n')

        return size_bytes

    def run(
        self,
        *args,
        initial_env: Dict[fx.Node, Any] | None = None,
        enable_io_processing: bool = True
    ) -> Any:
        return super().run(
            *args, initial_env=initial_env, enable_io_processing=enable_io_processing
        )

    def run_node(self, n: fx.Node) -> Any:
        # If you are in the backward pass region and one of the feature maps 'x'
        # was swapped out, and if node 'n' will use this feature map 'x' as one
        # of its inputs then you swap 'x' back to the GPU memory here.

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()

        # you can start measuring the run-time of a node here
        result = super().run_node(n)
        # you can end measuring the run-time of a node here HINT: Use
        # torch.cuda.Events for doing time measurements of operations.

        end_event.record()

        torch.cuda.synchronize()

        torch.cuda.empty_cache()
        self.name_to_cuda_memory[n.name] = torch.cuda.memory_allocated()
        # print(f'{n.name} -- memory allocated: {self.name_to_cuda_memory[n.name]} \n')

        # sys.stderr.write(f'Tensors: {self._unpack_tensors([result])}')

        self.name_to_result_ptrs[n.name] = [r.storage().data_ptr() for r in self._unpack_tensors([result])]

        execution_time_ms = start_event.elapsed_time(end_event)

        # add the execution time to the dictionary
        self.name_to_runtime[n.name].append(execution_time_ms)

        # now, we start calculating the memory allocated by this node
        # the memory usage function accounts for reused memory
        size_bytes = self._get_memory_usage(n, result)

        if size_bytes is not None:
            self.name_to_size[n.name].append(size_bytes)

        # If you are in the forward pass region and if the current node 'n' is
        # the last user of a feature map 'x', then it should be swapped out to
        # the CPU memory here.

        return result

    def aggregate_stats(self) -> None:
        # You are expected run the profiler for x warm-up iterations and y
        # actual measurement iterations. The run-time measurement then needs to
        # be averaged over the y runs.

        self.name_to_size_agg = {}
        self.name_to_runtime_agg = {}

        for k, v in self.name_to_size.items():
            self.name_to_size_agg[k] = float(torch.Tensor(v).mean().item())

        for k, v in self.name_to_runtime.items():
            self.name_to_runtime_agg[k] = float(torch.Tensor(v).mean().item())

    def print_stats(self) -> None:
        stats_table = []

        # add the column names
        stats_table.append([
            'name',
            'op',
            'target',
            'all_input_nodes',
            'users',
            'memory',
            'runtime',
            'node_type',
            'mem_cuda'
        ])

        for name in self.name_to_node.keys():
            # make sure we made measurements on all nodes
            if name not in self.name_to_size_agg.keys():
                sys.stderr.write(f"{name} not in name_to_size!!")
            elif name not in self.name_to_runtime_agg.keys():
                sys.stderr.write(f"{name} not in name_to_runtime!!")
            else:
                node = self.name_to_node[name]
                node_props = [
                    node.name,
                    node.op,
                    node.target,
                    node.all_input_nodes,
                    node.users,
                    self.name_to_size_agg[name],
                    self.name_to_runtime_agg[name],
                    self.name_to_nodetype[name],
                    self.name_to_cuda_memory[name]
                ]

                stats_table.append(node_props)

        maxcolwidths = [15] * len(stats_table[0])

        print(tabulate(stats_table, tablefmt="grid", maxcolwidths = maxcolwidths, floatfmt=".2f"))
            
    def reset_stats(self) -> None:
        # The statistics must be cleared out after x warm-up iterations and
        # reset before the actual measurement begins.

        # set all the measurements array to empty for resetting
        for node in self.module.graph.nodes:
            self.name_to_size[node.name] = []
            self.name_to_runtime[node.name] = []

            self.name_to_size_agg = {}
            self.name_to_runtime_agg = {}

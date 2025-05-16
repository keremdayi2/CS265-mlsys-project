# code for recomputations
from typing import List, Tuple, Set
from dataclasses import dataclass, fields
import numpy as np

import torch.fx as fx
import torch

from graph_prof import *


# have functions here to update recomputation ratio
# total compute time etc.
class RecomputeStats(NodeStats):
    # object to store recompute stats
    def __init__(self, parent: NodeStats):
        for field in fields(parent):
            setattr(self, field.name, getattr(parent, field.name))
        
        # initially, the computation time is equal
        # to the runtime and recompute count is 0
        self.recomp_srcs : Set[RecomputeStats] | None
        self.total_compute_time = self.runtime_agg 
        self.recomp_time = self.runtime_agg
        self.recompute_count = 0

        # tag intermediate node
        self.is_intermediate = (not (self.last_forward is None or self.first_backward is None)) and self.op != 'placeholder'

        # initial recompute ratio
        self.recompute_ratio = self.size_agg / self.total_compute_time
    
    # implement these to be able to use set operations
    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return (self.name == other.name)

@dataclass
class RecomputeNode:    
    name : str
    node : fx.Node
    first_bw : fx.Node
    first_bw_rank : int
    recomp_srcs : List[fx.Node] = field(default_factory=list)
        
# node stats should be a sufficient way of computing the recomputation policy
class RecomputePolicy:
    def __init__(self, nodes: List[NodeStats], name_to_node : Dict[str, fx.Node]):
        # create a list of nodes with recomputation stats
        # that inherit the properties of nodestats but offer additional functionality
        self.nodes : List[RecomputeStats] = [RecomputeStats(node) for node in nodes]
        self.name_to_rank = {node.name : node.rank for node in self.nodes}
        self.name_to_stats = {node.name : node for node in self.nodes}
        self.name_to_node = name_to_node

        for n in self.nodes:
            n.recomp_srcs = self._get_srcs(n, set())

    def _get_srcs(self, n : RecomputeStats, recompute_srcs: Set[RecomputeStats]):
        # look at the immediate dependency nodes
        for src in map(lambda x: self.name_to_stats[x], n.inputs):
            # if these are intermediate or parameters, then can add to our set
            # otherwise, recurse to get their depencendices
            if src.is_intermediate or src.op == 'placeholder':
                recompute_srcs.add(src)
            else:
                assert src.rank < n.rank
                # if we already computed the srcs, then we can use those
                # else, we need to recursively find the srcs
                if src.recomp_srcs is None:
                    self._get_srcs(src, recompute_srcs)
                else:
                    recompute_srcs.update(src.recomp_srcs)
        
        return recompute_srcs

    # return the list 
    def get_recomputation(self, memory_cap : int):
        # initially add all intermediate nodes (nodes used in forward pass 
        # and gradient computation) in the candidates
        # and initialize recompute nodes as empty
        candidates : Set[RecomputeStats] = set([n for n in self.nodes if n.is_intermediate])
        recompute_nodes : List[RecomputeStats] = []

        _, peak_mem = self._simulate_memory(recompute_nodes)
        sys.stderr.write(f"Initial peak memory usage: {peak_mem/1e9:.2f}GB\n")

        # continue until no candidates left
        # or we achieved our memory goal
        while len(candidates) > 0 and peak_mem > memory_cap: 
            # check peak memory to see if we can stop
            _, peak_mem = self._simulate_memory(recompute_nodes)

            # add the max recomputation_ratio candidate 
            # to recompute nodes and remove from candidates
            cand = self._max_recomp_candidate(candidates)
            candidates.remove(cand)

            # update existing recomputations
            recomp_cnt = 1 # recomputed once for itself
            for rp in recompute_nodes:
                if cand in rp.recomp_srcs:
                    rp.recomp_srcs.remove(cand)
                    # add the recomp srcs of the cand to the recomputation node
                    rp.recomp_srcs.update(cand.recomp_srcs)
                    rp.recomp_time += cand.recomp_time
                    recomp_cnt += 1

            recompute_nodes.append(cand)
            # update all future candidates

            # this part is from the paper 
            for future_cand in candidates:
                if cand in future_cand.recomp_srcs:
                    # if cand is active, no need to do anything
                    active_bw_interval = (self.name_to_rank[cand.first_backward], self.name_to_rank[cand.last_backward])
                    first_bw_access = self.name_to_rank[future_cand.first_backward]

                    # if the current node will be active at the time
                    # when the future node will be recomputed, then no need to recompute this node
                    if first_bw_access >= active_bw_interval[0] and first_bw_access <= active_bw_interval[1]:
                        continue

                    # otherwise, we would need to recompute the future node
                    future_cand.recomp_srcs.remove(cand)
                    future_cand.recomp_srcs.update(cand.recomp_srcs)
                    future_cand.recomp_time += cand.recomp_time
                    future_cand.total_compute_time = future_cand.recomp_time

                    # update the total recomputation time
                    for rp in recompute_nodes:
                        if future_cand in rp.recomp_srcs:
                            future_cand.total_compute_time += future_cand.recomp_time
                elif future_cand in cand.recomp_srcs:
                    future_cand.total_compute_time = recomp_cnt * future_cand.recomp_time
                
                future_cand.recompute_ratio = future_cand.size_agg / future_cand.total_compute_time
        
        if len(candidates) == 0:
            sys.stderr.write('WARNING: recomputation candidates terminated before memory goal achieved\n')
        
        _, peak_mem = self._simulate_memory(recompute_nodes)
        sys.stderr.write(f"Predicted peak memory usage: {peak_mem/1e9:.2f}GB\n")

        ret_nodes = [RecomputeNode(n.name, self.name_to_node[n.name], self.name_to_node[n.first_backward], self.name_to_rank[n.first_backward], [self.name_to_node[src.name] for src in n.recomp_srcs]) for n in recompute_nodes]

        # # some sanity check
        # for rp in ret_nodes:
        #     node = rp.node
        #     srcs = rp.recomp_srcs

        #     for src in srcs:
        #         assert self.name_to_stats[src.name].rank < self.name_to_stats[node.name].rank
        #         assert self.name_to_rank[self.name_to_stats[src.name].last_backward] > self.name_to_rank[self.name_to_stats[node.name].last_backward]

        # return the parsed list
        return ret_nodes

    # iterate over all candidates and find the one with the maximum recompute ratio
    def _max_recomp_candidate(self, candidates : Set[RecomputeStats]) -> RecomputeStats:
        max_candidate = next(iter(candidates))

        for c in candidates:
            if c.recompute_ratio > max_candidate.recompute_ratio:
                max_candidate = c

        return max_candidate


    # simulate memory
    # returns the memory usage array over time and 
    # the peak memory during forward and backward passes
    def _simulate_memory(self, recompute_nodes: List[RecomputeStats]) -> Tuple[np.array, float]:
        total_memory = np.zeros(len(self.nodes))

        forward_start, backward_end = len(self.nodes), 0
        for node in self.nodes:
            if node.is_intermediate:
                forward_start = min(forward_start, self.name_to_rank[node.first_forward])
                backward_end = max(backward_end, self.name_to_rank[node.last_backward])

        for node in self.nodes:
            # for params and opt states that are 
            # always in memory
            if node.op == 'placeholder': # static nodes
                total_memory += node.size_agg
            elif node not in recompute_nodes:
                # then usual computation applies
                start = node.rank 
                end = self.name_to_rank[node.last_use] if node.last_use is not None else len(self.nodes)
                total_memory[start:end] += node.size_agg
            else:
                assert node.is_intermediate
                
                # first find the forward pass memory
                if node.type != NodeType.ACT:
                    sys.stderr.write(f'Got type {node.type}\n')
                    assert False

                # first account for the memory use during forward pass
                # should free after the last forward use
                start = node.rank
                end = self.name_to_rank[node.last_forward] + 1
                total_memory[start:end] += node.size_agg

                # then account for the memory use in the backward pass
                # occupied by this node! It will become active 
                start = self.name_to_rank[node.first_backward]
                end = self.name_to_rank[node.last_backward] + 1
                total_memory[start:end] += node.size_agg

                # finally, account for the extra memory we will allocate while computing this node

                


        # Return memory profile and peak
        return total_memory, total_memory.max()
    
if __name__ == '__main__':
    print('Testing recompute')
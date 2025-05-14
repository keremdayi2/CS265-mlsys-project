# code for recomputations
from typing import List, Tuple, Set
from dataclasses import dataclass, fields
import numpy as np

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

# node stats should be a sufficient way of computing the recomputation policy
class RecomputePolicy:
    def __init__(self, nodes: List[NodeStats]):
        # create a list of nodes with recomputation stats
        # that inherit the properties of nodestats but offer additional functionality
        self.nodes = [RecomputeStats(node) for node in nodes]
        self.name_to_rank = {node.name : node.rank for node in self.nodes}
        self.name_to_stats = {node.name : node for node in self.nodes}

        for n in self.nodes:
            n.recomp_srcs = set()
            for src in map(lambda x: self.name_to_stats[x], n.srcs):
                if src.is_intermediate or src.op == 'placeholder':
                    n.recomp_srcs.add(src)
                else:
                    n.recomp_srcs.update(src.recomp_srcs)

    def get_recomputation(self, memory_cap : int):
        # initially add all intermediate nodes (nodes used in forward pass 
        # and gradient computation) in the candidates
        # and initialize recompute nodes as empty
        candidates : Set[RecomputeStats] = set([n for n in self.nodes if n.is_intermediate])
        recompute_nodes : Set[RecomputeStats] = set()

        # continue until no candidates left
        # or we achieved our memory goal
        while len(candidates) > 0: 
            # check peak memory to see if we can stop
            _, peak_mem = self._simulate_memory(recompute_nodes)
            if memory_cap > peak_mem:
                break

            # add the max recomputation_ratio candidate 
            # to recompute nodes and remove from candidates
            cand = self._max_recomp_candidate(candidates)
            
            candidates.remove(cand)
            # update existing recomputations
            recomp_cnt = 1
            for rp in recompute_nodes:
                if cand in rp.recomp_srcs:
                    rp.recomp_srcs.remove(cand)
                    # add the recomp srcs of the cand to the recomputation node
                    rp.recomp_srcs.update(cand.recomp_srcs)
                    rp.recomp_time += cand.recomp_time
                    recomp_cnt += 1

            recompute_nodes.add(cand)
            # update all future candidates

            # this part is from the paper 
            for future_cand in candidates:
                if cand in future_cand.recomp_srcs:
                    # if cand is active, no need to do anything
                    active_bw_interval = (self.name_to_rank[cand.first_backward], self.name_to_rank[cand.last_backward])
                    first_bw_access = self.name_to_rank[future_cand.first_backward]

                    if first_bw_access >= active_bw_interval[0] and first_bw_access <= active_bw_interval[1]:
                        continue

                    future_cand.recomp_srcs.remove(cand)
                    future_cand.recomp_srcs.update(cand.recomp_srcs)
                    future_cand.recomp_time += cand.recomp_time
                    future_cand.total_compute_time = future_cand.recomp_time

                    # update the total recomputation time
                    for rp in recompute_nodes:
                        if future_cand in rp.recomp_srcs:
                            future_cand.total_compute_time += future_cand.recomp_time
                if future_cand in cand.recomp_srcs:
                    future_cand.total_compute_time = recomp_cnt * future_cand.recomp_time
                
                future_cand.recompute_ratio = future_cand.size_agg / future_cand.total_compute_time
        
        return recompute_nodes


    # iterate over all candidates and find the one with the maximum recompute ratio
    def _max_recomp_candidate(self, candidates : Set[RecomputeStats]) -> RecomputeStats:
        max_candidate = next(iter(candidates))

        for c in candidates:
            if c.recompute_ratio > max_candidate.recompute_count:
                max_candidate = c

        return c


    # simulate memory
    def _simulate_memory(self, recompute_nodes : List[RecomputeStats]) -> Tuple[np.array, float]:
        memory_usage = np.zeros(len(self.nodes))

        # TODO: compute max memory over the forward and backward passes
        # (ignore the optimization step since we cannot do anything about that)
        forward_start, backward_start = None, None

        for node in self.nodes:
            # if this node is not being recomputed,
            # add the memory comsumption to its active interval
            
            # if this is a node that is recomputed, then 
            # it is inactive between its last use during forward pass
            # and its first use in backward pass
            if node in recompute_nodes:
                start_fw = self.name_to_rank[node.first_forward]
                end_fw = self.name_to_rank[node.last_forward]

                start_bw = self.name_to_rank[node.first_backward]
                end_bw = self.name_to_rank[node.last_backward]

                memory_usage[start_fw:end_fw] += node.size_agg
                memory_usage[start_bw:end_bw] += node.size_agg

            start = node.rank

            # for placeholders, activity starts at 0
            if node.op == 'placeholder':
                start = 0
            
            end = len(self.nodes)

            # if there is a last use for this node, set 
            # that as the end
            if not node.last_use is None:
                end = self.name_to_rank[node.last_use]
            
            memory_usage[start:end] += node.size_agg
        
        return memory_usage, memory_usage.max()
    
if __name__ == '__main__':
    print('Testing recompute')
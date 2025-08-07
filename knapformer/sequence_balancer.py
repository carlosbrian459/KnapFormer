from copy import deepcopy
import itertools
import logging
import math
import re
import time
from typing import Any, cast

from einops import rearrange
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.nn.functional import all_gather, all_to_all_single

logger = logging.getLogger(__name__)

"""
For convention, we assume the following ordered dicts:
- chunk_id2gpu_id, chunk_id2chunk_len are sorted dicts by the memory layout of (gpu_id, chunk_id)
- gpu_id2chunk_ids, gpu_id2chunk_lens are sorted dicts with increasing gpu_id, then on each gpu,
    the chunk_ids are sorted in increasing order

The other dicts need not necessarily be ordered.
"""


def fast_all2all_chunks(
    packed_chunks: torch.Tensor,
    this_gpu_id: int,
    group_gpu_ids: list[int],
    src_gpu_id2chunk_ids: dict[int, list[int]],
    src_gpu_id2chunk_lens: dict[int, list[int]],
    trgt_chunk_id2gpu_id: dict[int, int],
    process_group: dist.ProcessGroup | None = None,
    sort_by_chunk_id_gpu_id: bool = True,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], dict[int, list[int]], dict[int, list[int]]]:
    """
    packed_chunks: (l1+l2+...+ln, d) or (l1+l2+...+ln, h, d)
    this_gpu_id: int
    group_gpu_ids: list[int]; must be sorted
    src_gpu_id2chunk_ids: dict[int, list[int]]; keys must be sorted
    src_gpu_id2chunk_lens: dict[int, list[int]]; keys must be sorted
    trgt_chunk_id2gpu_id: dict[int, int]; keys must be sorted by (trgt_gpu_id, chunk_id),
        i.e., trgt_gpu_id is the primary key and chunk_id is the secondary key
    """
    shape, device, dtype = packed_chunks.shape, packed_chunks.device, packed_chunks.dtype

    # Determine where to send the chunks on current gpu
    send_buckets: dict[int, list[torch.Tensor]] = {gpu_id: [] for gpu_id in group_gpu_ids}
    send_numels = dict.fromkeys(group_gpu_ids, 0)
    for chunk, chunk_id in zip(
        packed_chunks.split(src_gpu_id2chunk_lens[this_gpu_id], dim=0),
        src_gpu_id2chunk_ids[this_gpu_id],
        strict=True,
    ):
        trgt_gpu_id = trgt_chunk_id2gpu_id[chunk_id]
        send_buckets[trgt_gpu_id].append(chunk)
        send_numels[trgt_gpu_id] += chunk.numel()

    # Determine where to recv the chunks from other gpus to this gpu
    recv_chunk_id_gpu_id_pairs = {gpu_id: [] for gpu_id in group_gpu_ids}
    recv_chunk_numels = {gpu_id: [] for gpu_id in group_gpu_ids}
    recv_numels = dict.fromkeys(group_gpu_ids, 0)
    for gpu_id in group_gpu_ids:
        for chunk_id, chunk_len in zip(src_gpu_id2chunk_ids[gpu_id], src_gpu_id2chunk_lens[gpu_id], strict=True):
            if trgt_chunk_id2gpu_id[chunk_id] == this_gpu_id:
                recv_chunk_id_gpu_id_pairs[gpu_id].append((chunk_id, gpu_id))
                chunk_numel = chunk_len * math.prod(shape[1:])
                recv_chunk_numels[gpu_id].append(chunk_numel)
                recv_numels[gpu_id] += chunk_numel

    send_buffer = torch.cat(list(itertools.chain(*send_buckets.values())), dim=0).ravel()
    send_split_sizes = list(send_numels.values())
    recv_split_sizes = list(recv_numels.values())
    recv_buffer = torch.empty(sum(recv_split_sizes), device=device, dtype=dtype)
    recv_buffer = all_to_all_single(recv_buffer, send_buffer, recv_split_sizes, send_split_sizes, group=process_group)

    if sort_by_chunk_id_gpu_id:
        recv_chunk_id_gpu_id_pairs = list(itertools.chain(*recv_chunk_id_gpu_id_pairs.values()))
        already_sorted = all(
            recv_chunk_id_gpu_id_pairs[i] < recv_chunk_id_gpu_id_pairs[i + 1]
            for i in range(len(recv_chunk_id_gpu_id_pairs) - 1)
        )
        if not already_sorted:
            recv_chunk_numels = list(itertools.chain(*recv_chunk_numels.values()))
            recv_buffer = recv_buffer.split(recv_chunk_numels, dim=0)
            sorted_tuples = sorted(
                zip(range(len(recv_chunk_id_gpu_id_pairs)), recv_chunk_id_gpu_id_pairs, strict=True),
                key=lambda x: x[1],
            )
            recv_buffer = torch.cat([recv_buffer[i] for i, _ in sorted_tuples], dim=0)

    recv_chunks = recv_buffer.reshape(-1, *shape[1:])
    return recv_chunks


class UlyssesBag:
    def __init__(
        self,
        bag_id: int,
        this_gpu_id: int,
        gpu_ids: list[int],
        gpu_id2chunk_ids: dict[int, list[int]],
        gpu_id2chunk_lens: dict[int, list[int]],
        chunk_id2gpu_id: dict[int, int],
        chunk_id2chunk_len: dict[int, int],
        chunk_id2seq_id: dict[int, int],
        seq_id2seq_len: dict[int, int],
        process_group: dist.ProcessGroup | None = None,
    ) -> None:
        self.bag_id = bag_id
        self.this_gpu_id = this_gpu_id
        self.gpu_ids = gpu_ids
        self.gpu_id2chunk_ids = gpu_id2chunk_ids
        self.gpu_id2chunk_lens = gpu_id2chunk_lens
        self.chunk_id2gpu_id = chunk_id2gpu_id
        self.chunk_id2chunk_len = chunk_id2chunk_len
        self.chunk_id2seq_id = chunk_id2seq_id
        self.seq_id2seq_len = seq_id2seq_len
        self.process_group = process_group
        assert not (len(self.gpu_ids) > 1 and self.process_group is None), (
            f"process_group must be provided when there are multiple GPUs, "
            f"but got {len(self.gpu_ids)} GPUs and process_group is None"
        )

        self.all_chunk_ids, self.all_chunk_lens = [], []
        for gpu_id in self.gpu_ids:
            self.all_chunk_ids.extend(self.gpu_id2chunk_ids[gpu_id])
            self.all_chunk_lens.extend(self.gpu_id2chunk_lens[gpu_id])
        # Sort by increasing chunk_ids
        sorted_tuples = sorted(
            zip(range(len(self.all_chunk_ids)), self.all_chunk_ids, self.all_chunk_lens, strict=True),
            key=lambda x: x[1],
        )
        self.sorted_indices, self.sorted_all_chunk_ids, self.sorted_all_chunk_lens = zip(*sorted_tuples, strict=True)

        self.gpu_id2sorted_all_chunk_ids, self.gpu_id2sorted_all_chunk_lens = {}, {}
        for gpu_id in self.gpu_ids:
            self.gpu_id2sorted_all_chunk_ids[gpu_id] = deepcopy(self.sorted_all_chunk_ids)
            self.gpu_id2sorted_all_chunk_lens[gpu_id] = deepcopy(self.sorted_all_chunk_lens)

        device = torch.cuda.current_device()
        self.this_chunk_lens = torch.tensor(self.gpu_id2chunk_lens[self.this_gpu_id], dtype=torch.int32, device=device)
        # self.this_chunk_lens = self.gpu_id2chunk_lens[self.this_gpu_id]

        # Build ordered list of sequence lengths by iterating sorted chunk IDs
        this_attn_seq_lens = []
        seen_seq_ids = set()
        for chunk_id in self.sorted_all_chunk_ids:
            seq_id = self.chunk_id2seq_id[chunk_id]
            if seq_id not in seen_seq_ids:
                seen_seq_ids.add(seq_id)
                this_attn_seq_lens.append(self.seq_id2seq_len[seq_id])
        self.this_attn_seq_lens = torch.tensor(this_attn_seq_lens, dtype=torch.int32, device=device)
        # self.this_attn_seq_lens = this_attn_seq_lens

    def pre_attn(
        self, packed_q: torch.Tensor, packed_k: torch.Tensor, packed_v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        packed_q, packed_k, packed_v: (l'1+l'2+...+l'n, h, d), i.e., partial seqs full heads
        return: (l1+l2+...+ln, h//#gpus, d), i.e., full seqs partial heads

        comm message: partial seqs partial heads, partial seqs partial heads,...
        """
        assert packed_q.shape == packed_k.shape == packed_v.shape, (
            f"packed_q, packed_k, packed_v must have the same shape, "
            f"but got {packed_q.shape}, {packed_k.shape}, {packed_v.shape}"
        )
        assert (
            packed_q.ndim == 3
        ), f"packed_q, packed_k, packed_v must be of shape (l1+l2+...+ln, h, d), but got {packed_q.shape}"

        # If there is only one GPU, no need to do all-to-all
        if len(self.gpu_ids) == 1:
            return self.this_attn_seq_lens, packed_q, packed_k, packed_v

        # Otherwise, do ulysses_pre_attn
        packed_chunks = torch.cat([packed_q, packed_k, packed_v], dim=-1)
        num_gpus = len(self.gpu_ids)
        num_heads, head_dim = packed_chunks.shape[1], packed_chunks.shape[2]
        dtype, device = packed_chunks.dtype, packed_chunks.device
        num_heads_per_gpu = num_heads // num_gpus
        assert num_heads_per_gpu * num_gpus == num_heads, (
            f"num_heads_per_gpu * num_gpus must be equal to num_heads, "
            f"but got {num_heads_per_gpu} * {num_gpus} != {num_heads}"
        )

        send_tensors, recv_numels, recv_chunk_ids, recv_chunk_numels = [], [], [], []
        for gpu_id, head_chunks in zip(self.gpu_ids, packed_chunks.chunk(num_gpus, dim=1), strict=True):
            send_tensors.append(head_chunks)

            recv_numels.append(sum(self.gpu_id2chunk_lens[gpu_id]) * num_heads_per_gpu * head_dim)
            recv_chunk_ids.extend(self.gpu_id2chunk_ids[gpu_id])
            recv_chunk_numels.extend([ll * num_heads_per_gpu * head_dim for ll in self.gpu_id2chunk_lens[gpu_id]])

        send_split_sizes = [t.numel() for t in send_tensors]
        send_buffer = torch.cat(send_tensors, dim=0).ravel()
        recv_split_sizes = recv_numels
        recv_buffer = torch.empty(sum(recv_split_sizes), device=device, dtype=dtype)
        recv_buffer = all_to_all_single(
            recv_buffer, send_buffer, recv_split_sizes, send_split_sizes, group=self.process_group
        )

        # re-order the recv_chunks on each gpu by the order of chunk_ids
        recv_buffer = recv_buffer.split(recv_chunk_numels, dim=0)
        sorted_tuples = sorted(zip(range(len(recv_chunk_ids)), recv_chunk_ids, strict=True), key=lambda x: x[1])
        recv_chunks = torch.cat([recv_buffer[i] for i, _ in sorted_tuples], dim=0)
        recv_chunks = recv_chunks.reshape(-1, num_heads_per_gpu, head_dim)
        recv_q, recv_k, recv_v = recv_chunks.chunk(3, dim=-1)

        return self.this_attn_seq_lens, recv_q, recv_k, recv_v

    def post_attn(self, packed_chunks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        packed_chunks: (l1+l2+...+ln, h//#gpus, d), i.e., full seqs partial heads
        return: (l'1+l'2+...+l'n, h, d), i.e., partial seqs full heads
        """
        assert (
            packed_chunks.ndim == 3
        ), f"packed_chunks must be of shape (l1+l2+...+ln, h//#gpus, d), but got {packed_chunks.shape}"

        # If there is only one GPU, no need to do all-to-all
        if len(self.gpu_ids) == 1:
            return self.this_chunk_lens, packed_chunks

        recv_chunks = fast_all2all_chunks(
            packed_chunks,
            self.this_gpu_id,
            self.gpu_ids,
            self.gpu_id2sorted_all_chunk_ids,
            self.gpu_id2sorted_all_chunk_lens,
            self.chunk_id2gpu_id,
            self.process_group,
            sort_by_chunk_id_gpu_id=False,
        )
        recv_chunks = rearrange(recv_chunks, "(g ll) hh dd -> ll (g hh) dd", g=len(self.gpu_ids))
        return self.this_chunk_lens, recv_chunks


def tflops_estimator(seq_len: int, d: int, num_layers: int, include_bwd: bool = False) -> float:
    """
    l: sequence length
    d: token dimension
    """
    fwd_tflops = (24 * seq_len * (d * d) + 4 * (seq_len * seq_len) * d) * num_layers / 1e12

    final_tflops = fwd_tflops
    if include_bwd:
        bwd_tflops = 2 * fwd_tflops
        recompute_tflops = fwd_tflops  # assume activation checkpointing
        final_tflops += bwd_tflops + recompute_tflops
    return final_tflops


def workload_estimator(seq_len: int, d: int, gamma: float = 1.0) -> float:
    """
    l: sequence length
    d: token dimension
    gamma: a parameter to balance between the MLP and Attn terms
    """
    # divide by 1e12 to get tflops
    return (24 * seq_len * (d * d) + gamma * 4 * (seq_len * seq_len) * d) / 1e12


class SequenceBalancer:
    def __init__(self, bag_specs: str, gamma: float = 1.0) -> None:
        """
        bag_specs (str): written in the form of "g{G1}n{N1}+g{G2}n{N2}+...+g{Gm}n{Nm}", where each one denotes N_i bags of size G_i
        """
        self.bag_specs = bag_specs
        self.gamma = gamma

        self.this_gpu_id = dist.get_rank()
        self.this_device = torch.cuda.current_device()

        # Parse the bag_specs
        parsed_bag_specs = []
        for spec in bag_specs.split("+"):
            spec = spec.strip()
            match = re.match(r"g(\d+)n(\d+)", spec)
            if not match:
                raise ValueError(f"Invalid bag spec format: {spec}. Expected format: g{{G}}n{{N}}")
            num_gpus_in_a_bag, num_such_bags = int(match.group(1)), int(match.group(2))
            parsed_bag_specs.append((num_gpus_in_a_bag, num_such_bags))
        self.parsed_bag_specs = parsed_bag_specs  # (num_gpus_in_a_bag, num_such_bags)
        # sort by increasing num_gpus_in_a_bag
        self.parsed_bag_specs.sort(key=lambda x: x[0])
        # make sure we don't have duplicate num_gpus_in_a_bag
        assert len(self.parsed_bag_specs) == len(
            {num_gpus_in_a_bag for num_gpus_in_a_bag, _ in self.parsed_bag_specs}
        ), f"parsed_bag_specs must not have duplicate num_gpus_in_a_bag, but got {self.parsed_bag_specs}"

        # Initialize the balance process group
        total_gpus_in_parsed_bag_specs = sum(
            num_gpus_in_a_bag * num_such_bags for num_gpus_in_a_bag, num_such_bags in self.parsed_bag_specs
        )
        balance_replicas = dist.get_world_size() // total_gpus_in_parsed_bag_specs
        assert dist.get_world_size() % total_gpus_in_parsed_bag_specs == 0, (
            f"dist.get_world_size() must be divisible by total_gpus_in_parsed_bag_specs, "
            f"but got {dist.get_world_size()} % {total_gpus_in_parsed_bag_specs} != 0"
        )
        mesh = init_device_mesh(
            "cuda",
            (balance_replicas, total_gpus_in_parsed_bag_specs),
            mesh_dim_names=("replica", "balance"),
        )
        self.balance_process_group = mesh["balance"].get_group()

        # Derive the balance gpu ids this gpu belongs to
        this_replica_id = self.this_gpu_id // total_gpus_in_parsed_bag_specs
        self.balance_gpu_ids = list(
            range(
                this_replica_id * total_gpus_in_parsed_bag_specs,
                (this_replica_id + 1) * total_gpus_in_parsed_bag_specs,
            )
        )

        # Instantiate the bags
        (
            self.bag_gpu_options,
            self.bag_gpu_options2bag_ids,
            self.bag_id2gpu_ids,
            self.gpu_id2bag_id,
        ) = (
            [],
            {},
            {},
            {},
        )
        bag_id, gpu_id = 0, this_replica_id * total_gpus_in_parsed_bag_specs
        for num_gpus_in_a_bag, num_such_bags in self.parsed_bag_specs:
            self.bag_gpu_options.append(num_gpus_in_a_bag)
            self.bag_gpu_options2bag_ids[num_gpus_in_a_bag] = []
            for _ in range(num_such_bags):
                self.bag_gpu_options2bag_ids[num_gpus_in_a_bag].append(bag_id)

                self.bag_id2gpu_ids[bag_id] = list(range(gpu_id, gpu_id + num_gpus_in_a_bag))
                for _gpu_id in self.bag_id2gpu_ids[bag_id]:
                    self.gpu_id2bag_id[_gpu_id] = bag_id

                bag_id += 1
                gpu_id += num_gpus_in_a_bag

        # Print bag_id2gpu_ids
        logger.info(f"bag_id2gpu_ids: {self.bag_id2gpu_ids}; self.parsed_bag_specs: {self.parsed_bag_specs}")

        # Create ulysses process groups for bags containing multiple GPUs
        # Note that we need to take care of all balance_replicas
        self.bag_id2process_group = {}
        for replica_id in range(balance_replicas):
            bag_id, gpu_id = 0, replica_id * total_gpus_in_parsed_bag_specs
            for num_gpus_in_a_bag, num_such_bags in self.parsed_bag_specs:
                for _ in range(num_such_bags):
                    gpu_ids = list(range(gpu_id, gpu_id + num_gpus_in_a_bag))
                    process_group = None
                    if len(gpu_ids) > 1:
                        logger.info(
                            f"rank={dist.get_rank()} creating process group for bag {bag_id} with gpu_ids={gpu_ids}"
                        )
                        process_group = dist.new_group(gpu_ids)
                    # only record this replica's process group
                    if replica_id == this_replica_id:
                        if process_group is not None:
                            logger.info(
                                f"rank={dist.get_rank()} recording process group"
                                f" for bag {bag_id} with gpu_ids={gpu_ids}"
                            )
                        self.bag_id2process_group[bag_id] = process_group

                    bag_id += 1
                    gpu_id += num_gpus_in_a_bag

        # For run-time logging
        self.metrics: dict[str, float | int] = {
            "start_time": 0,
            "end_time": 0,
            "total_tokens": 0,
            "duration_sec": 0,
            "avg_per_gpu_tokens": 0,
        }

    def _gather_workloads(self, local_seq_lens: list[int] | torch.Tensor, token_dim: int) -> None:
        local_count = len(local_seq_lens)
        global_counts = all_gather(
            torch.tensor([local_count], dtype=torch.int32, device=self.this_device),
            group=self.balance_process_group,
        )

        global_counts = [c.item() for c in cast(list[torch.Tensor], global_counts)]

        max_count = int(max(global_counts))
        padded_local_seq_lens = torch.zeros(max_count, dtype=torch.int32, device=self.this_device)
        if isinstance(local_seq_lens, list | tuple):
            local_seq_lens = torch.tensor(local_seq_lens, dtype=torch.int32, device=self.this_device)  # type: ignore

        padded_local_seq_lens[:local_count] = local_seq_lens
        global_seq_lens = all_gather(padded_local_seq_lens, group=self.balance_process_group)

        self.local_seq_lens = local_seq_lens

        # Ordered dicts
        (
            self.seq_id2gpu_id,
            self.seq_id2seq_len,
            self.seq_id2seq_workload,
            self.gpu_id2seq_ids,
            self.gpu_id2seq_lens,
            self.gpu_id2seq_workloads,
        ) = {}, {}, {}, {}, {}, {}
        global_seq_id = 0
        for gpu_id, seq_count, seq_lens in zip(
            self.balance_gpu_ids, global_counts, cast(list[torch.Tensor], global_seq_lens), strict=True
        ):
            seq_ids = list(range(int(global_seq_id), int(global_seq_id + seq_count)))
            seq_lens = seq_lens[:seq_count].tolist()
            seq_workloads = [workload_estimator(seq_len, token_dim, self.gamma) for seq_len in seq_lens]
            for seq_id, seq_len, seq_workload in zip(seq_ids, seq_lens, seq_workloads, strict=True):
                self.seq_id2gpu_id[seq_id] = gpu_id
                self.seq_id2seq_len[seq_id] = seq_len
                self.seq_id2seq_workload[seq_id] = seq_workload
            self.gpu_id2seq_ids[gpu_id] = seq_ids
            self.gpu_id2seq_lens[gpu_id] = seq_lens
            self.gpu_id2seq_workloads[gpu_id] = seq_workloads
            global_seq_id += seq_count

        self.gpu_id2total_workload = {}
        for gpu_id in self.balance_gpu_ids:
            self.gpu_id2total_workload[gpu_id] = sum(self.gpu_id2seq_workloads[gpu_id])

    def plan_routing(self, local_seq_lens: list[int] | torch.Tensor, token_dim: int) -> None:
        self.metrics["start_time"] = time.time()

        # Gather the workloads of all sequences on all GPUs in the balance process group
        self._gather_workloads(local_seq_lens, token_dim)

        # Calculate the capacity of each gpu in the balance process group
        total_workload = sum(self.seq_id2seq_workload.values())
        per_gpu_capacity = total_workload / len(self.balance_gpu_ids)
        self.per_gpu_capacity = per_gpu_capacity

        # First pass: assign sequences to bags in a greedy manner
        #   for each sequence, filter out the bags whose full capacity is not enough to hold the sequence
        #   then in the remaining bags, assign the sequence to the bag with the smallest occupancy ratio
        (
            self.seq_id2assigned_bag_id,
            self.bag_id2capacity,
            self.bag_id2assigned_seq_ids,
            self.bag_id2occupancy,
        ) = {}, {}, {}, {}
        for bag_id in self.bag_id2gpu_ids:
            self.bag_id2capacity[bag_id] = per_gpu_capacity * len(self.bag_id2gpu_ids[bag_id])
            self.bag_id2assigned_seq_ids[bag_id] = []
            self.bag_id2occupancy[bag_id] = 0

        for seq_id, seq_workload in sorted(self.seq_id2seq_workload.items(), key=lambda x: x[1], reverse=True):
            candidate_bag_ids = []
            for bag_id in self.bag_id2capacity:
                # make sure the bag has enough capacity and the sequence is long enough to be divided into chunks
                if (self.bag_id2capacity[bag_id] >= seq_workload) and (
                    self.seq_id2seq_len[seq_id] >= len(self.bag_id2gpu_ids[bag_id])
                ):
                    candidate_bag_ids.append(bag_id)
            # if no bag can hold the sequence, use the largest bag option
            if len(candidate_bag_ids) == 0:
                candidate_bag_ids = self.bag_gpu_options2bag_ids[self.bag_gpu_options[-1]]
            # Sort by increasing occupancy ratio
            candidate_bag_ids.sort(key=lambda x: self.bag_id2occupancy[x])
            selected_bag_id = candidate_bag_ids[0]
            self.seq_id2assigned_bag_id[seq_id] = selected_bag_id
            self.bag_id2assigned_seq_ids[selected_bag_id].append(seq_id)
            self.bag_id2occupancy[selected_bag_id] += seq_workload / self.bag_id2capacity[selected_bag_id]

        # sort self.seq_id2assigned_bag_id by increasing seq_id,
        # bag_id2assigned_seq_ids by increasing bag_id, bag_id2occupancy by increasing bag_id
        self.seq_id2assigned_bag_id = dict(sorted(self.seq_id2assigned_bag_id.items(), key=lambda x: x[0]))

        # Second pass: for bags with multiple GPUs, divide the assgined sequences into multiple chunks
        (
            self.chunk_id2gpu_id,
            self.chunk_id2chunk_len,
            self.chunk_id2seq_id,
            self.gpu_id2chunk_ids,
            self.gpu_id2chunk_lens,
            self.balance_chunk_id2gpu_id,
            self.balance_gpu_id2chunk_ids,
            self.balance_gpu_id2chunk_lens,
            self.balance_gpu_id2seq_ids,
        ) = {}, {}, {}, {}, {}, {}, {}, {}, {}
        global_chunk_id = 0
        rolling_int = 0
        for seq_id, assigned_bag_id in self.seq_id2assigned_bag_id.items():
            gpu_id = self.seq_id2gpu_id[seq_id]
            seq_len = self.seq_id2seq_len[seq_id]
            assigned_gpu_ids = self.bag_id2gpu_ids[assigned_bag_id]
            num_gpus_in_bag = len(assigned_gpu_ids)

            for assigned_gpu_id in assigned_gpu_ids:
                if assigned_gpu_id not in self.balance_gpu_id2seq_ids:
                    self.balance_gpu_id2seq_ids[assigned_gpu_id] = []
                self.balance_gpu_id2seq_ids[assigned_gpu_id].append(seq_id)

            chunk_lens = [seq_len // num_gpus_in_bag] * num_gpus_in_bag
            total_len = sum(chunk_lens)
            # randomly modify one to meet the seq_len constraint
            idx = rolling_int % num_gpus_in_bag
            chunk_lens[idx] = seq_len - (total_len - chunk_lens[idx])  # type: ignore
            rolling_int += 1

            if gpu_id not in self.gpu_id2chunk_ids:
                self.gpu_id2chunk_ids[gpu_id] = []
                self.gpu_id2chunk_lens[gpu_id] = []

            for chunk_id, chunk_len, assigned_gpu_id in zip(
                range(global_chunk_id, global_chunk_id + num_gpus_in_bag),
                chunk_lens,
                assigned_gpu_ids,
                strict=True,
            ):
                self.chunk_id2gpu_id[chunk_id] = gpu_id
                self.chunk_id2chunk_len[chunk_id] = chunk_len
                self.chunk_id2seq_id[chunk_id] = seq_id
                self.gpu_id2chunk_ids[gpu_id].append(chunk_id)
                self.gpu_id2chunk_lens[gpu_id].append(chunk_len)

                self.balance_chunk_id2gpu_id[chunk_id] = assigned_gpu_id
                if assigned_gpu_id not in self.balance_gpu_id2chunk_ids:
                    self.balance_gpu_id2chunk_ids[assigned_gpu_id] = []
                self.balance_gpu_id2chunk_ids[assigned_gpu_id].append(chunk_id)
                if assigned_gpu_id not in self.balance_gpu_id2chunk_lens:
                    self.balance_gpu_id2chunk_lens[assigned_gpu_id] = []
                self.balance_gpu_id2chunk_lens[assigned_gpu_id].append(chunk_len)

            global_chunk_id += num_gpus_in_bag

        # Third pass: sort self.balance_gpu_id2chunk_ids, self.balance_gpu_id2chunk_lens
        #       by increasing gpu_id; then sort by increasing chunk_ids on each gpu
        #   sort self.balance_chunk_id2gpu_id by (gpu_id, chunk_id)
        self.balance_gpu_id2chunk_ids = dict(sorted(self.balance_gpu_id2chunk_ids.items(), key=lambda x: x[0]))
        self.balance_gpu_id2chunk_lens = dict(sorted(self.balance_gpu_id2chunk_lens.items(), key=lambda x: x[0]))
        for gpu_id in self.balance_gpu_ids:
            sorted_tuples = sorted(
                zip(
                    self.balance_gpu_id2chunk_ids[gpu_id],
                    self.balance_gpu_id2chunk_lens[gpu_id],
                    strict=True,
                ),
                key=lambda x: x[0],
            )
            (
                self.balance_gpu_id2chunk_ids[gpu_id],
                self.balance_gpu_id2chunk_lens[gpu_id],
            ) = zip(*sorted_tuples, strict=True)

        sorted_tuples = sorted(self.balance_chunk_id2gpu_id.items(), key=lambda x: (x[1], x[0]))
        self.balance_chunk_id2gpu_id = dict(sorted_tuples)
        self.balance_chunk_id2chunk_len = {
            chunk_id: self.chunk_id2chunk_len[chunk_id] for chunk_id in self.balance_chunk_id2gpu_id
        }

        # Estimate the workload of each gpu after balancing
        self.balance_gpu_id2total_workload = {}
        for gpu_id in self.balance_gpu_ids:
            bag_id = self.gpu_id2bag_id[gpu_id]
            num_gpus_in_bag = len(self.bag_id2gpu_ids[bag_id])
            bag_capacity = self.bag_id2capacity[bag_id]
            bag_occupancy = self.bag_id2occupancy[bag_id]
            bag_workload = bag_capacity * bag_occupancy
            self.balance_gpu_id2total_workload[gpu_id] = bag_workload / num_gpus_in_bag

        # Record information for logging
        sorted_workloads = sorted(self.gpu_id2total_workload.values())
        sorted_balance_workloads = sorted(self.balance_gpu_id2total_workload.values())
        self.orig_minmax_workload = (sorted_workloads[0], sorted_workloads[-1])
        self.bala_minmax_workload = (sorted_balance_workloads[0], sorted_balance_workloads[-1])

        # Finally, we set up the ulysses bag containing this gpu
        bag_id = self.gpu_id2bag_id[self.this_gpu_id]
        self.ulysses_bag = UlyssesBag(
            bag_id,
            self.this_gpu_id,
            self.bag_id2gpu_ids[bag_id],
            self.balance_gpu_id2chunk_ids,
            self.balance_gpu_id2chunk_lens,
            self.balance_chunk_id2gpu_id,
            self.balance_chunk_id2chunk_len,
            self.chunk_id2seq_id,
            self.seq_id2seq_len,
            self.bag_id2process_group[bag_id],
        )

        if hasattr(self, "tracking") and self.tracking:
            self.tracked_per_gpu_capacity.append(self.per_gpu_capacity)

            sorted_workloads = sorted(self.gpu_id2total_workload.values())
            self.tracked_original_imbalance_ratio.append(sorted_workloads[-1] / sorted_workloads[0])

            sorted_balance_workloads = sorted(self.balance_gpu_id2total_workload.values())
            self.tracked_balanced_imbalance_ratio.append(sorted_balance_workloads[-1] / sorted_balance_workloads[0])

            total_tflops = 0
            for seq_len in self.seq_id2seq_len.values():
                total_tflops += tflops_estimator(seq_len, token_dim, self.num_layers, self.include_bwd)
            self.tracked_tflops.append(total_tflops)

    def get_routing_plan_summary(self) -> dict[str, Any]:
        summary_keys = [
            "balance_gpu_ids",
            "bag_id2gpu_ids",
            "gpu_id2bag_id",
            "seq_id2seq_len",
            "seq_id2seq_workload",
            "gpu_id2seq_ids",
            "gpu_id2seq_lens",
            "gpu_id2seq_workloads",
            "gpu_id2chunk_ids",
            "gpu_id2chunk_lens",
            "balance_chunk_id2gpu_id",
            "balance_chunk_id2chunk_len",
            "chunk_id2seq_id",
        ]
        return {k: getattr(self, k) for k in summary_keys}

    def nice_print(self) -> None:
        for k, v in self.get_routing_plan_summary().items():
            logger.info(f"Rank {self.this_gpu_id} {k}: {v}")

    def print_imbalance(self) -> None:
        sorted_workloads = sorted(self.gpu_id2total_workload.values())
        sorted_balance_workloads = sorted(self.balance_gpu_id2total_workload.values())

        # report each gpu's workload with respect to the per_gpu_capacity
        def report_workload(gpu_id2workload: dict[int, float], tag: str = "") -> None:
            output_str = f"{tag} - per_gpu_capacity={self.per_gpu_capacity:.2f} - "
            for gpu_id, workload in gpu_id2workload.items():
                output_str += f"{gpu_id}: {workload / self.per_gpu_capacity:.2f} "
            logger.info(output_str)

        report_workload(self.gpu_id2total_workload, "original")
        report_workload(self.balance_gpu_id2total_workload, "balanced")
        logger.info(
            f"max/min workload ratio: {sorted_workloads[-1] / sorted_workloads[0]}"
            f" ---> {sorted_balance_workloads[-1] / sorted_balance_workloads[0]}"
        )

    def turn_on_tracking(self, num_layers: int = 1, include_bwd: bool = False) -> None:
        self.tracking = True
        self.tracked_per_gpu_capacity = []
        self.tracked_original_imbalance_ratio = []
        self.tracked_balanced_imbalance_ratio = []
        self.tracked_tflops = []
        self.num_layers = num_layers
        self.include_bwd = include_bwd

    def turn_off_tracking(self) -> None:
        self.tracking = False
        self.tracked_per_gpu_capacity = []
        self.tracked_original_imbalance_ratio = []
        self.tracked_balanced_imbalance_ratio = []
        self.tracked_tflops = []

    def print_tracking_results(self) -> None:
        logger.info(
            f"Per GPU Capacity: {np.mean(self.tracked_per_gpu_capacity)}, std: {np.std(self.tracked_per_gpu_capacity)}"
        )
        logger.info(
            f"Original Imbalance Ratio: {np.mean(self.tracked_original_imbalance_ratio)}, std: {np.std(self.tracked_original_imbalance_ratio)}"
        )
        logger.info(
            f"Balanced Imbalance Ratio: {np.mean(self.tracked_balanced_imbalance_ratio)}, std: {np.std(self.tracked_balanced_imbalance_ratio)}"
        )

    def route(
        self,
        packed_seqs: torch.Tensor,
        packed_features: list[torch.Tensor] | None = None,
        force_fp32_buffer: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor] | None]:
        """
        packed_seqs: (l1+l2+...+ln, d)
        packed_features: list[torch.Tensor] | None
        """
        assert packed_seqs.ndim == 2, f"packed_seqs must be of shape (l1+l2+...+ln, d), but got {packed_seqs.shape}"

        # torch.promote_types(torch.bfloat16, torch.int32) -> torch.bfloat16
        # pitfalls: torch.tensor([1023]).type(torch.bfloat16) -> tensor([1024.], dtype=torch.bfloat16)

        # Pack input tensors
        seq_dim, seq_dtype, feature_dims, feature_dtypes = (
            packed_seqs.shape[1],
            packed_seqs.dtype,
            [],
            [],
        )
        packed_buffer, packed_buffer_dtype = packed_seqs, seq_dtype
        if packed_features is not None:
            # make sure features dtype are all the same as tokens
            for f in packed_features:
                feature_dims.append(f.shape[1])
                feature_dtypes.append(f.dtype)
                packed_buffer_dtype = torch.promote_types(packed_buffer_dtype, f.dtype)
            if force_fp32_buffer:
                packed_buffer_dtype = torch.float32

            packed_buffer = torch.cat(
                [packed_buffer.type(packed_buffer_dtype)] + [f.type(packed_buffer_dtype) for f in packed_features],
                dim=1,
            )

        packed_buffer = fast_all2all_chunks(
            packed_buffer,
            self.this_gpu_id,
            self.balance_gpu_ids,
            self.gpu_id2chunk_ids,
            self.gpu_id2chunk_lens,
            self.balance_chunk_id2gpu_id,
            self.balance_process_group,
        )

        bala_chunk_lens = self.balance_gpu_id2chunk_lens[self.this_gpu_id]
        packed_buffer = packed_buffer.split([seq_dim] + feature_dims, dim=1)
        packed_seqs = packed_buffer[0].type(seq_dtype)
        if packed_features is not None:
            packed_features = [f.type(dtype) for f, dtype in zip(packed_buffer[1:], feature_dtypes, strict=True)]
            return bala_chunk_lens, packed_seqs, packed_features
        else:
            return bala_chunk_lens, packed_seqs, None

    def reverse_route(
        self,
        packed_seqs: torch.Tensor,
        packed_features: list[torch.Tensor] | None = None,
        force_fp32_buffer: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor] | None]:
        """
        packed_seqs: (l1+l2+...+ln, d)
        """
        assert packed_seqs.ndim == 2, f"packed_seqs must be of shape (l1+l2+...+ln, d), but got {packed_seqs.shape}"

        # Pack input tensors
        seq_dim, seq_dtype, feature_dims, feature_dtypes = (
            packed_seqs.shape[1],
            packed_seqs.dtype,
            [],
            [],
        )
        packed_buffer, packed_buffer_dtype = packed_seqs, seq_dtype
        if packed_features is not None:
            # make sure features dtype are all the same as tokens
            for f in packed_features:
                feature_dims.append(f.shape[1])
                feature_dtypes.append(f.dtype)
                packed_buffer_dtype = torch.promote_types(packed_buffer_dtype, f.dtype)
            if force_fp32_buffer:
                packed_buffer_dtype = torch.float32

            packed_buffer = torch.cat(
                [packed_buffer.type(packed_buffer_dtype)] + [f.type(packed_buffer_dtype) for f in packed_features],
                dim=1,
            )

        packed_buffer = fast_all2all_chunks(
            packed_buffer,
            self.this_gpu_id,
            self.balance_gpu_ids,
            self.balance_gpu_id2chunk_ids,
            self.balance_gpu_id2chunk_lens,
            self.chunk_id2gpu_id,
            self.balance_process_group,
        )

        packed_buffer = packed_buffer.split([seq_dim] + feature_dims, dim=1)
        packed_seqs = packed_buffer[0].type(seq_dtype)

        # Update metrics
        end_time = time.time()
        duration = end_time - self.metrics["start_time"]
        total_tokens = sum(self.seq_id2seq_len.values())
        self.metrics["end_time"] = end_time
        self.metrics["total_tokens"] = total_tokens
        self.metrics["duration_sec"] = duration
        self.metrics["avg_per_gpu_tokens"] = total_tokens / len(self.balance_gpu_ids)

        if packed_features is not None:
            packed_features = [f.type(dtype) for f, dtype in zip(packed_buffer[1:], feature_dtypes, strict=True)]
            return self.local_seq_lens, packed_seqs, packed_features
        else:
            return self.local_seq_lens, packed_seqs, None

    def pre_attn(
        self, packed_q: torch.Tensor, packed_k: torch.Tensor, packed_v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        packed_q, packed_k, packed_v: (l'1+l'2+...+l'n, h,  d)
        """
        seq_lens, packed_q, packed_k, packed_v = self.ulysses_bag.pre_attn(packed_q, packed_k, packed_v)
        return seq_lens, packed_q, packed_k, packed_v

    def post_attn(self, packed_seqs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        packed_seqs: (l'1+l'2+...+l'n, h/#gpus,  d)
        """
        seq_lens, packed_seqs = self.ulysses_bag.post_attn(packed_seqs)
        return seq_lens, packed_seqs

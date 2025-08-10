import logging
import os

import torch
import torch.distributed as dist

from knapformer import SequenceBalancer
from knapformer.utils.fsdp_utils import setup_distributed, teardown_distributed
from knapformer.utils.transformer_utils import TransformerBlock, get_rope_cossin

logger = logging.getLogger(__name__)

# Set NCCL debug level to WARN to reduce verbosity
os.environ["NCCL_DEBUG"] = "WARN"


@torch.no_grad()
def create_transformer_block(d_model: int, d_head: int):
    block = TransformerBlock(d_model, d_head).cuda()
    # broadcast parameters from rank 0 to all ranks
    for param in block.parameters():
        dist.broadcast(param, 0)
    return block


@torch.no_grad()
def create_sequences(
    d_model: int,
    d_head: int,
    d_feature: int,
    token_dtype: torch.dtype = torch.bfloat16,
    feature_dtype: torch.dtype = torch.float32,
):
    """
    Create sequences for all GPUs.
    """
    token_type2len_range = {
        "txt_t5": (64, 256),
        "img_256-res": (220, 286),
        "img_512-res": (880, 1144),
        "img_1024-res": (3520, 4576),
        "img_2048-res": (14080, 18304),
    }

    gpu_id2data = {
        0: (32, "img_256-res"),
        1: (32, "img_256-res"),
        2: (8, "img_512-res"),
        3: (2, "img_1024-res"),
        4: (1, "img_256-res"),  # For SP
        5: (1, "img_256-res"),  # For SP
        6: (1, "img_256-res"),  # For SP
        7: (1, "img_2048-res"),  # For SP
    }
    assert (
        len(gpu_id2data) == dist.get_world_size()
    ), f"Number of GPUs ({len(gpu_id2data)}) must match the number of ranks ({dist.get_world_size()})"

    device = torch.cuda.current_device()
    gpu_id2seq_lens, gpu_id2packed_seqs, gpu_id2packed_features = (
        {},
        {},
        {},
    )
    for gpu_id, (num_seqs, token_type) in gpu_id2data.items():
        min_t5_seq_len, max_t5_seq_len = token_type2len_range["txt_t5"]
        t5_seq_lens = torch.randint(min_t5_seq_len, max_t5_seq_len + 1, (num_seqs,), device=device)

        min_seq_len, max_seq_len = token_type2len_range[token_type]
        seq_lens = t5_seq_lens + torch.randint(min_seq_len, max_seq_len + 1, (num_seqs,), device=device)
        # broadcast seq_lens to all ranks
        dist.broadcast(seq_lens, 0)
        seq_lens = seq_lens.tolist()

        packed_seqs = (0.02 * torch.randn(sum(seq_lens), d_model, device=device)).type(token_dtype)
        # broadcast packed_seqs to all ranks
        dist.broadcast(packed_seqs, 0)

        packed_features = [
            get_rope_cossin(seq_lens, d_head, device=device),
            (0.02 * torch.randn(sum(seq_lens), d_feature, device=device)).type(feature_dtype),
        ]
        # broadcast packed_features to all ranks
        for feature in packed_features:
            dist.broadcast(feature, 0)

        gpu_id2seq_lens[gpu_id] = torch.tensor(seq_lens, device=device)
        gpu_id2packed_seqs[gpu_id] = packed_seqs
        gpu_id2packed_features[gpu_id] = packed_features

    return (
        gpu_id2seq_lens,
        gpu_id2packed_seqs,
        gpu_id2packed_features,
    )


def test_route_and_reverse_route(
    sequence_balancer: SequenceBalancer,
    gpu_id2seq_lens: dict[int, list[int]],
    gpu_id2packed_seqs: dict[int, torch.Tensor],
    gpu_id2packed_features: dict[int, list[torch.Tensor]],
):
    """
    Test route and reverse route.
    """
    this_gpu_id = dist.get_rank()
    this_gpu_seq_lens = gpu_id2seq_lens[this_gpu_id]
    this_gpu_packed_seqs = gpu_id2packed_seqs[this_gpu_id]
    this_gpu_packed_features = gpu_id2packed_features[this_gpu_id]

    sequence_balancer.plan_routing(this_gpu_seq_lens, this_gpu_packed_seqs.shape[-1])
    # sequence_balancer.nice_print()
    sequence_balancer.print_imbalance()

    bala_chunk_lens, bala_seqs, bala_features = sequence_balancer.route(this_gpu_packed_seqs, this_gpu_packed_features)
    seq_lens, packed_seqs, packed_features = sequence_balancer.reverse_route(bala_seqs, bala_features)
    assert (seq_lens == this_gpu_seq_lens).all().item()
    assert torch.allclose(packed_seqs, this_gpu_packed_seqs)
    for packed_feature, this_gpu_feature in zip(packed_features, this_gpu_packed_features, strict=True):
        assert torch.allclose(packed_feature, this_gpu_feature)
    logger.info(f"rank={this_gpu_id} test_route_and_reverse_route passed")
    dist.barrier()


def test_transformer_block_foward_backward(
    sequence_balancer: SequenceBalancer,
    transformer_block: TransformerBlock,
    gpu_id2packed_seqs: dict[int, torch.Tensor],
    gpu_id2packed_features: dict[int, list[torch.Tensor]],
):
    """
    Test forward and backward pass of transformer block.
    """
    this_gpu_id = dist.get_rank()
    this_gpu_packed_seqs = gpu_id2packed_seqs[this_gpu_id]
    this_gpu_seq_lens = gpu_id2seq_lens[this_gpu_id]
    this_gpu_packed_features = gpu_id2packed_features[this_gpu_id]

    this_gpu_packed_seqs.requires_grad = True
    for packed_feature in this_gpu_packed_features:
        packed_feature.requires_grad = True
    for param in transformer_block.parameters():
        param.requires_grad = True

    ########### without sequence balancer ###########
    without_sequence_balancer_out = transformer_block(
        this_gpu_packed_seqs, this_gpu_seq_lens, this_gpu_packed_features[0]
    )
    loss = torch.mean(without_sequence_balancer_out)
    loss.backward()
    without_sequence_balancer_grads = [this_gpu_packed_seqs.grad.clone().detach()] + [
        param.grad.clone().detach() for param in transformer_block.parameters()
    ]

    ########### with sequence balancer ###########
    this_gpu_packed_seqs.grad = None
    for packed_feature in this_gpu_packed_features:
        packed_feature.grad = None
    for param in transformer_block.parameters():
        param.grad = None

    sequence_balancer.plan_routing(this_gpu_seq_lens, this_gpu_packed_seqs.shape[-1])
    bala_chunk_lens, bala_seqs, bala_features = sequence_balancer.route(this_gpu_packed_seqs, this_gpu_packed_features)
    bala_seqs = transformer_block(bala_seqs, bala_chunk_lens, bala_features[0], sequence_balancer)
    seq_lens, with_sequence_balancer_out, _ = sequence_balancer.reverse_route(bala_seqs)
    loss = torch.mean(with_sequence_balancer_out)
    loss.backward()
    with_sequence_balancer_grads = [this_gpu_packed_seqs.grad.clone().detach()] + [
        param.grad.clone().detach() for param in transformer_block.parameters()
    ]

    assert (seq_lens == this_gpu_seq_lens).all().item()
    assert torch.allclose(with_sequence_balancer_out, without_sequence_balancer_out)
    # rtol, atol copied from: https://github.com/Dao-AILab/flash-attention/blob/3669b25206d5938e3cc74a5f7860e31c38af8204/hopper/test_flash_attn.py#L11-L12
    # https://github.com/Dao-AILab/flash-attention/blob/32792d37ec66902e5d82e149971daacbee8b55d7/hopper/benchmark_attn.py#L273-L279
    for with_grad, without_grad in zip(with_sequence_balancer_grads, without_sequence_balancer_grads, strict=True):
        assert torch.allclose(with_grad, without_grad, rtol=0.05, atol=0.05)

    logger.info(f"rank={this_gpu_id} test_transformer_block_foward_backward passed")
    dist.barrier()


if __name__ == "__main__":
    """
    torchrun --nproc_per_node=8 tests/test_transformer.py 2>&1 | tee test_transformer.log
    """
    setup_distributed()
    sequence_balancer = SequenceBalancer("g1n2+g2n1+g4n1")

    ########### test route and reverse route ###########
    gpu_id2seq_lens, gpu_id2packed_seqs, gpu_id2packed_features = create_sequences(
        d_model=4096,
        d_head=128,
        d_feature=1024,
        token_dtype=torch.bfloat16,
        feature_dtype=torch.float32,
    )
    test_route_and_reverse_route(sequence_balancer, gpu_id2seq_lens, gpu_id2packed_seqs, gpu_id2packed_features)

    ########### test transformer block ###########
    gpu_id2seq_lens, gpu_id2packed_seqs, gpu_id2packed_features = create_sequences(
        d_model=4096,
        d_head=128,
        d_feature=1024,
        token_dtype=torch.bfloat16,
        feature_dtype=torch.float32,
    )
    transformer_block = create_transformer_block(4096, 128).to(dtype=torch.bfloat16)
    test_transformer_block_foward_backward(
        sequence_balancer, transformer_block, gpu_id2packed_seqs, gpu_id2packed_features
    )

    teardown_distributed()

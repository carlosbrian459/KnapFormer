import datetime
import logging
import os

import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.nn.functional import all_gather
import torch.nn as nn

logger = logging.getLogger(__name__)


def setup_distributed() -> None:
    """Setup distributed environment for testing."""
    if dist.is_initialized():
        return

    rank = int(os.environ.get("RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    local_rank = int(os.environ.get("LOCAL_RANK", -1))

    logger.info(f"Initializing process group - RANK: {rank}, WORLD_SIZE: {world_size}, LOCAL_RANK: {local_rank}")

    try:
        # Set device for this rank first
        torch.cuda.set_device(local_rank)
        # Use env:// init method with timeout
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=datetime.timedelta(seconds=30),
            device_id=torch.device(f"cuda:{local_rank}"),
        )
        logger.info(f"Process group initialized for rank {rank}")
    except Exception as e:
        logger.error(f"Error initializing process group: {e}")
        raise


def teardown_distributed() -> None:
    """Cleanup distributed environment."""
    if dist.is_initialized():
        dist.barrier()  # make sure all ranks are ready
        dist.destroy_process_group()


def all_gather_with_padding(tensor: torch.Tensor, group: dist.ProcessGroup):
    assert tensor.ndim == 2, f"tensor must be a 2D tensor, but got {tensor.ndim}"
    tensor_bs = all_gather(
        torch.tensor(
            [
                tensor.shape[0],
            ],
            device=tensor.device,
        ),
        group=group,
    )
    tensor_bs = torch.cat(tensor_bs, dim=0).tolist()  # (B1, B2, ..., Bn)

    max_bs = max(tensor_bs)
    padded_tensor = torch.empty((max_bs, tensor.shape[1]), dtype=tensor.dtype, device=tensor.device)
    padded_tensor[: tensor.shape[0]] = tensor
    padded_tensors = all_gather(padded_tensor, group=group)
    tensors = torch.cat([t[:b] for t, b in zip(padded_tensors, tensor_bs, strict=True)], dim=0)
    return tensors


def _apply_ac_to_transformer_block(module: nn.Module, ac_freq: int):
    ptd_checkpoint_wrapper.__dict__.setdefault("_count", 0)
    ptd_checkpoint_wrapper._count += 1
    if (ac_freq > 0) and (ptd_checkpoint_wrapper._count % ac_freq == 0):
        return ptd_checkpoint_wrapper(module, preserve_rng_state=False)
    else:
        return module


def apply_ac(model: nn.Module, ac_freq: int):
    """
    Modified from: https://github.com/pytorch/torchtitan/blob/7d5f3cc698853d2227cf5433776406d0e0345424/torchtitan/models/llama3/infra/parallelize.py#L303

    Apply activation checkpointing to the model.
    """
    for layer_id, transformer_block in model.blocks.items():
        transformer_block = _apply_ac_to_transformer_block(transformer_block, ac_freq)
        model.blocks.register_module(layer_id, transformer_block)


def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    reshard_after_forward_policy: str = "default",
):
    """
    Modified from: https://github.com/pytorch/torchtitan/blob/7d5f3cc698853d2227cf5433776406d0e0345424/torchtitan/models/llama3/infra/parallelize.py#L324

    Apply data parallelism (via FSDP2) to the model.

    Args:
        model (nn.Module): The model to apply data parallelism to.
        dp_mesh (DeviceMesh): The device mesh to use for data parallelism.
        param_dtype (torch.dtype): The data type to use for model parameters.
        reduce_dtype (torch.dtype): The data type to use for reduction operations.
        reshard_after_forward_policy (str, optional): The policy to use for resharding after forward pass. Defaults to "default".
            Other options: "never", "always".
            - "default" applies default resharding behavior, implementing "smart defaults" for known optimal scenarios.
            - "always" will enable `reshard_after_forward` for all forward passes.
            - "never" will disable `reshard_after_forward` for all forward passes.

    """
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=False,
    )
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}

    for layer_id, transformer_block in model.blocks.items():
        if reshard_after_forward_policy == "always":
            reshard_after_forward = True
        elif reshard_after_forward_policy == "never":
            reshard_after_forward = False
        elif reshard_after_forward_policy == "default":
            reshard_after_forward = int(layer_id) < len(model.blocks) - 1
        else:
            raise ValueError(f"Invalid reshard_after_forward_policy: {reshard_after_forward_policy}.")
        fully_shard(
            transformer_block,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )
    fully_shard(model, **fsdp_config, reshard_after_forward=reshard_after_forward)


def common_model_setup(
    model: nn.Module,
    shard_size: int,
    param_dtype: torch.dtype = torch.bfloat16,
    reduce_dtype: torch.dtype = torch.float32,
    ac_freq: int = 0,
):
    ### apply selective activation checkpointing ###
    apply_ac(model, ac_freq)

    ### apply FSDP2 ###
    assert (
        dist.get_world_size() % shard_size == 0
    ), f"world_size {dist.get_world_size()} must be divisible by shard_size {shard_size}"
    dp_mesh = init_device_mesh(
        "cuda",
        (dist.get_world_size() // shard_size, shard_size),
        mesh_dim_names=("replicate", "shard"),
    )
    apply_fsdp(model, dp_mesh, param_dtype, reduce_dtype)

    ### init weights ###
    model.to_empty(device="cuda")
    model.init_weights()

    return model

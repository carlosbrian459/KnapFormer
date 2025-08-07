import logging

import torch.distributed as dist

from utils.fsdp_utils import setup_distributed, teardown_distributed

logger = logging.getLogger(__name__)


def ddp_barrier(message="Synchronizing all nodes"):
    """
    Simple barrier function to synchronize all nodes in distributed training.

    Args:
        message (str): Message to log before barrier (all ranks will log)
    """
    setup_distributed()

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # All ranks log their status
    logger.info(f"[Rank {rank}] {message}... (waiting for {world_size} processes)")

    try:
        # Synchronize all processes
        dist.barrier()

        # All ranks confirm synchronization
        logger.info(f"[Rank {rank}] Synchronized successfully!")
    except Exception as e:
        logger.error(f"[Rank {rank}] Barrier failed: {e}")
        raise

    teardown_distributed()


if __name__ == "__main__":
    """
    Standalone usage for shell scripts:
    torchrun --nnodes=N --nproc_per_node=1 --node_rank=X \
        --master_addr=ADDR --master_port=PORT \
        ddp_barrier.py "Custom message"
    """
    import argparse

    parser = argparse.ArgumentParser(description="DDP Barrier Utility")
    parser.add_argument(
        "message",
        nargs="?",
        default="Synchronizing all nodes",
        help="Message to display during barrier",
    )
    args = parser.parse_args()

    ddp_barrier(args.message)

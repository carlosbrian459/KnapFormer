import logging
import re

import torch
import torch.distributed as dist

from knapformer.utils.transformer_utils import get_rope_cossin

logger = logging.getLogger(__name__)


def parse_data_code(
    code: str,
    vae_spatial_rate: float = 16.0,
    vae_temporal_rate: float = 3.4,  # 17/5 = 3.4
):
    """
    Parses a data code like:
        g4b2i512f8s1
    Parameters:
        vae_spatial_rate: spatial downsampling factor of VAE (can be float)
        vae_temporal_rate: temporal downsampling factor (can be float)
    Returns:
        dict with all metadata and token counts
    """
    pattern = r"g(\d+)b(\d+)i(\d+)f(\d+)s(\d+)"
    match = re.match(pattern, code)
    if not match:
        raise ValueError(f"Invalid data code format: {code}")

    num_gpus, batch_size, resolution, frame_count, smoothness = map(int, match.groups())

    # Compute base spatial tokens using float division, then floor
    spatial_dim = resolution / vae_spatial_rate
    spatial_tokens = torch.floor(torch.tensor(spatial_dim)).item() ** 2

    # Compute temporal token length
    if smoothness == 0:
        temporal_tokens = frame_count
    else:
        temporal_tokens = max(int(frame_count / vae_temporal_rate), 1)

    tokens_per_sample = spatial_tokens * temporal_tokens

    return {
        "code": code,
        "num_gpus": num_gpus,
        "batch_size": batch_size,
        "resolution": resolution,
        "frame_count": frame_count,
        "smoothness": smoothness,
        "tokens_per_sample": tokens_per_sample,
    }


def assign_data_codes_to_ranks(
    data_codes: list[str],
    world_size: int,
    vae_spatial_rate: float = 16.0,
    vae_temporal_rate: float = 3.4,
):
    """
    Assigns data codes to GPU ranks, automatically repeated across groups
    based on world_size and GPUs required per group.

    Args:
        data_codes: list of strings like 'g4b2i256f1s0'
        world_size: total number of GPUs available
        vae_*: VAE parameters for token calculation

    Returns:
        rank_to_data_info: dict[int, dict] mapping rank -> parsed data info
    """
    parsed_data_infos = []
    gpus_per_group = 0

    # First pass: parse and sum GPUs required per group
    for code in data_codes:
        data_info = parse_data_code(
            code,
            vae_spatial_rate=vae_spatial_rate,
            vae_temporal_rate=vae_temporal_rate,
        )
        parsed_data_infos.append(data_info)
        gpus_per_group += data_info["num_gpus"]

    if world_size % gpus_per_group != 0:
        raise RuntimeError(f"world_size ({world_size}) is not divisible by GPUs per group ({gpus_per_group})")

    num_groups = world_size // gpus_per_group

    rank_to_data_info = {}

    for group_id in range(num_groups):
        base_rank = group_id * gpus_per_group
        current_offset = 0

        for data_info in parsed_data_infos:
            num_gpus = data_info["num_gpus"]
            for r in range(base_rank + current_offset, base_rank + current_offset + num_gpus):
                rank_to_data_info[r] = data_info
            current_offset += num_gpus

    return rank_to_data_info


class DataSimulator:
    def __init__(
        self,
        data_codes: list[str],
        d_model: int = 4096,
        d_head: int = 128,
        vae_spatial_rate: float = 16.0,
        vae_temporal_rate: float = 3.4,  # 17/5 = 3.4
        min_aspect_ratio_multiplier: float = 1.0,
        max_aspect_ratio_multiplier: float = 1.0,
        min_t5_tokens: int = 0,
        max_t5_tokens: int = 392,
        random_seed: int | None = None,
    ):
        """Example data codes for 32-GPU setup:
        data_codes = [
            # Image data
            "g8b4i256f1s0",
            "g2b5i512f1s0",
            "g2b5i1024f1s0",
            "g4b1i2048f1s0",
            # Keyframe data
            "g1b10i256f4s0",
            "g3b1i512f4s0",
            # Video data
            "g8b2i256f85s1",
            "g4b1i512f85s1",
        ]
        """
        self.data_codes = data_codes
        self.d_model = d_model
        self.d_head = d_head
        self.vae_spatial_rate = vae_spatial_rate
        self.vae_temporal_rate = vae_temporal_rate
        self.min_aspect_ratio_multiplier = min_aspect_ratio_multiplier
        self.max_aspect_ratio_multiplier = max_aspect_ratio_multiplier
        self.min_t5_tokens = min_t5_tokens
        self.max_t5_tokens = max_t5_tokens

        if random_seed is None:
            self.random_seed = dist.get_rank()
        else:
            self.random_seed = random_seed
        self.rng_cpu = torch.Generator("cpu").manual_seed(self.random_seed)
        self.rng_gpu = torch.Generator("cuda").manual_seed(self.random_seed)

        self.rank_to_data_info = assign_data_codes_to_ranks(
            data_codes,
            world_size=dist.get_world_size(),
            vae_spatial_rate=vae_spatial_rate,
            vae_temporal_rate=vae_temporal_rate,
        )

        self.this_rank = dist.get_rank()
        self.this_device = torch.cuda.current_device()
        self.this_rank_data_info = self.rank_to_data_info[self.this_rank]
        self.this_batch_size = self.this_rank_data_info["batch_size"]
        self.this_visual_tokens_per_sample = self.this_rank_data_info["tokens_per_sample"]

        if self.this_rank == 0:
            logger.info(f"DataSimulator initialized for data codes: {data_codes}")
            logger.info(f"rank_to_data_info: {self.rank_to_data_info}")

    def next_batch(self):
        # Sample an aspect ratio multiplier (e.g., 0.75 ~ 1.33)
        aspect_ratio_multiplier = (
            torch.rand(1, generator=self.rng_cpu).item()
            * (self.max_aspect_ratio_multiplier - self.min_aspect_ratio_multiplier)
            + self.min_aspect_ratio_multiplier
        )
        visual_tokens_per_sample = int(self.this_visual_tokens_per_sample * aspect_ratio_multiplier)

        t5_seq_lens = [
            torch.randint(
                self.min_t5_tokens,
                self.max_t5_tokens + 1,
                (1,),
                generator=self.rng_cpu,
            ).item()
            for _ in range(self.this_batch_size)
        ]

        packed_seq_lens, packed_modality_tag = [], []
        for i in range(self.this_batch_size):
            t5_seq_len = t5_seq_lens[i]
            packed_seq_lens.append(t5_seq_len + visual_tokens_per_sample)

            packed_modality_tag.append(
                torch.cat(
                    [
                        torch.zeros(
                            (t5_seq_len, 1),
                            dtype=torch.int32,
                            device=self.this_device,
                        ),
                        torch.ones(
                            (visual_tokens_per_sample, 1),
                            dtype=torch.int32,
                            device=self.this_device,
                        ),
                    ],
                    dim=0,
                )
            )
        packed_modality_tag = torch.cat(packed_modality_tag, dim=0)

        packed_seqs = (
            torch.randn(
                sum(packed_seq_lens),
                self.d_model,
                device=self.this_device,
                generator=self.rng_gpu,
            )
            * 0.02
        )
        packed_features = [
            get_rope_cossin(packed_seq_lens, self.d_head, device=self.this_device),
            packed_modality_tag,
        ]
        return packed_seq_lens, packed_seqs, packed_features

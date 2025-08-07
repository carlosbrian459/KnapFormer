import logging

import torch
import torch.nn as nn

from knapformer import SequenceBalancer
from utils.fsdp_utils import all_gather_with_padding, common_model_setup
from utils.mmdit_utils import DoubleStreamBlock, SingleStreamBlock
from utils.transformer_utils import TransformerBlock

logger = logging.getLogger(__name__)


class Transformer(nn.Module):
    def __init__(self, d_model: int, d_head: int, n_blocks: int, causal: bool = False):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.n_blocks = n_blocks
        self.causal = causal

        self.blocks = torch.nn.ModuleDict()  # OrderedDict
        for layer_id in range(n_blocks):
            self.blocks[str(layer_id)] = TransformerBlock(d_model, d_head, causal)

    def init_weights(self):
        for block in self.blocks.values():
            block.init_weights()

    def forward(
        self,
        packed_seqs: torch.Tensor,
        packed_seq_lens: torch.Tensor,
        packed_features: torch.Tensor,
        sequence_balancer: SequenceBalancer | None = None,
    ):
        if sequence_balancer is None:
            seq_out_lens, seq_out = packed_seq_lens, packed_seqs
            pe = packed_features[0]
            for _, block in self.blocks.items():
                seq_out = block(seq_out, seq_out_lens, pe)
        else:
            sequence_balancer.plan_routing(packed_seq_lens, packed_seqs.shape[-1])

            bala_chunk_lens, bala_seqs, bala_features = sequence_balancer.route(packed_seqs, packed_features)
            bala_pe = bala_features[0]
            for _, block in self.blocks.items():
                bala_seqs = block(bala_seqs, bala_chunk_lens, bala_pe, sequence_balancer)
            seq_out_lens, seq_out, _ = sequence_balancer.reverse_route(bala_seqs)

        return seq_out_lens, seq_out


class MMDiT(nn.Module):
    def __init__(self, d_model: int, d_head: int, n_ds_blocks: int, n_ss_blocks: int):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.n_ds_blocks = n_ds_blocks
        self.n_ss_blocks = n_ss_blocks

        self.blocks = nn.ModuleDict()
        layer_id = 0
        for _ in range(n_ds_blocks):
            self.blocks[str(layer_id)] = DoubleStreamBlock(d_model, d_head)
            layer_id += 1
        for _ in range(n_ss_blocks):
            self.blocks[str(layer_id)] = SingleStreamBlock(d_model, d_head)
            layer_id += 1

    def init_weights(self):
        for block in self.blocks.values():
            block.init_weights()

    def forward(
        self,
        packed_seqs: torch.Tensor,
        packed_seq_lens: torch.Tensor,
        packed_features: torch.Tensor,
        vec: torch.Tensor,
        sequence_balancer: SequenceBalancer | None = None,
    ):
        assert vec.shape[0] == len(
            packed_seq_lens
        ), f"vec must have the same length as packed_seq_lens, but got {vec.shape[0]} and {len(packed_seq_lens)}"
        assert vec.ndim == 2, f"vec must be a 2D tensor, but got {vec.ndim}"

        if sequence_balancer is None:
            seq_ids = []
            for seq_len, seq_id in zip(packed_seq_lens, range(len(packed_seq_lens)), strict=True):
                seq_ids.extend([seq_id] * seq_len)
            seq_ids = torch.tensor(seq_ids, device=vec.device)

            packed_pe, modality_tag = packed_features
            modality_tag = modality_tag.squeeze(-1)
            txt_indices, img_indices = (torch.nonzero(modality_tag == tag_id, as_tuple=True)[0] for tag_id in [0, 1])

            seq_out_lens, seq_out = packed_seq_lens, packed_seqs
            for _, block in self.blocks.items():
                seq_out = block(
                    seq_out,
                    packed_pe,
                    vec,
                    seq_out_lens,
                    seq_ids,
                    txt_indices,
                    img_indices,
                )
        else:
            vec = all_gather_with_padding(vec, group=sequence_balancer.balance_process_group)

            sequence_balancer.plan_routing(packed_seq_lens, packed_seqs.shape[-1])

            seq_ids = []
            for seq_len, seq_id in zip(
                sequence_balancer.gpu_id2seq_lens[sequence_balancer.this_gpu_id],
                sequence_balancer.gpu_id2seq_ids[sequence_balancer.this_gpu_id],
                strict=True,
            ):
                seq_ids.extend([seq_id] * seq_len)
            seq_ids = torch.tensor(seq_ids, device=vec.device)

            bala_chunk_lens, bala_seqs, bala_features = sequence_balancer.route(
                packed_seqs, packed_features + [seq_ids.unsqueeze(-1)]
            )
            bala_pe, bala_modality_tag, bala_seq_ids = bala_features
            bala_modality_tag = bala_modality_tag.squeeze(-1)
            bala_seq_ids = bala_seq_ids.squeeze(-1)
            bala_txt_indices, bala_img_indices = (
                torch.nonzero(bala_modality_tag == tag_id, as_tuple=True)[0] for tag_id in [0, 1]
            )

            for _, block in self.blocks.items():
                bala_seqs = block(
                    bala_seqs,
                    bala_pe,
                    vec,
                    bala_chunk_lens,
                    bala_seq_ids,
                    bala_txt_indices,
                    bala_img_indices,
                    sequence_balancer,
                )
            seq_out_lens, seq_out, _ = sequence_balancer.reverse_route(bala_seqs)

        return seq_out_lens, seq_out


def create_transformer(
    d_model: int,
    d_head: int,
    n_layers: int,
    shard_size: int,
    param_dtype: torch.dtype = torch.bfloat16,
    reduce_dtype: torch.dtype = torch.float32,
    ac_freq: int = 0,
    causal: bool = False,
):
    logger.info(
        f"Creating Transformer with d_model={d_model}, d_head={d_head}, n_layers={n_layers}, shard_size={shard_size}, param_dtype={param_dtype}, reduce_dtype={reduce_dtype}, ac_freq={ac_freq}, causal={causal}"
    )

    ### meta device init ###
    with torch.device("meta"):
        model = Transformer(d_model, d_head, n_layers, causal)

    model = common_model_setup(model, shard_size, param_dtype, reduce_dtype, ac_freq)
    return model


def create_mmdit(
    d_model: int,
    d_head: int,
    n_ds_layers: int,
    n_ss_layers: int,
    shard_size: int,
    param_dtype: torch.dtype = torch.bfloat16,
    reduce_dtype: torch.dtype = torch.float32,
    ac_freq: int = 0,
):
    logger.info(
        f"Creating MMDiT with d_model={d_model}, d_head={d_head}, n_ds_layers={n_ds_layers}, n_ss_layers={n_ss_layers}, shard_size={shard_size}, param_dtype={param_dtype}, reduce_dtype={reduce_dtype}, ac_freq={ac_freq}"
    )

    ### meta device init ###
    with torch.device("meta"):
        model = MMDiT(d_model, d_head, n_ds_layers, n_ss_layers)

    model = common_model_setup(model, shard_size, param_dtype, reduce_dtype, ac_freq)
    return model

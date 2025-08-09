from dataclasses import dataclass
import itertools

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

from knapformer import SequenceBalancer
from knapformer.utils.transformer_utils import apply_attn, apply_rope


@dataclass
class ModulationOut:
    shift: torch.Tensor
    scale: torch.Tensor
    gate: torch.Tensor


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool, zero_init: bool = True) -> None:
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)
        self.zero_init = zero_init

        self.init_weights()

    def init_weights(self) -> None:
        if self.zero_init:
            nn.init.constant_(self.lin.weight, 0.0)
            nn.init.constant_(self.lin.bias, 0.0)
        else:
            nn.init.trunc_normal_(self.lin.weight, std=0.02)
            nn.init.constant_(self.lin.bias, 0.0)

    def forward(
        self, vec: torch.Tensor, seq_ids: torch.Tensor | None = None
    ) -> tuple[ModulationOut, ModulationOut | None]:
        """
        vec: (b, d) or (b1+b2+...+bn, d)
        seq_ids: one-d-tensor of indices
        -> ((b, 3d), (b, 3d)) if double else ((b, 3d), None)
        """
        assert vec.ndim == 2, f"vec must be a 2D tensor, but got {vec.ndim}"

        out = self.lin(F.silu(vec))

        if seq_ids is not None:
            assert seq_ids.ndim == 1, f"seq_ids must be a 1D tensor, but got {seq_ids.ndim}"
            assert (
                seq_ids.max() < out.shape[0]
            ), f"seq_ids.max()={seq_ids.max()} must be less than out.shape[0]={out.shape[0]}"
            out = torch.index_select(out, 0, seq_ids)

        out = out.chunk(self.multiplier, dim=-1)
        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


class FluxRMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

        self.init_weights()

    def init_weights(self) -> None:
        nn.init.constant_(self.scale, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms * self.scale).to(dtype=x_dtype)


class QKNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.query_norm = FluxRMSNorm(dim)
        self.key_norm = FluxRMSNorm(dim)

        self.init_weights()

    def init_weights(self) -> None:
        nn.init.constant_(self.query_norm.scale, 1.0)
        nn.init.constant_(self.key_norm.scale, 1.0)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q, k = self.query_norm(q), self.key_norm(k)
        return q, k


class SelfAttention(nn.Module):
    def __init__(self, d_model: int, d_head: int, flux_mode: bool = True) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.n_heads = d_model // d_head

        self.qkv = nn.Linear(d_model, d_model * 3, bias=True)
        self.norm = QKNorm(d_head)
        self.proj = nn.Linear(d_model, d_model, bias=True)

        self.flux_mode = flux_mode

        self.init_weights()

    def init_weights(self) -> None:
        nn.init.trunc_normal_(self.qkv.weight, std=0.02)
        nn.init.constant_(self.qkv.bias, 0.0)
        self.norm.init_weights()
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        nn.init.constant_(self.proj.bias, 0.0)

    def forward(
        self,
        x: torch.Tensor,
        pe: torch.Tensor,
        seq_lens: list[int],
        sequence_balancer: SequenceBalancer | None = None,
    ) -> torch.Tensor:
        """
        x: (l1+l2+...+ln, d)
        pe: (l1+l2+...+ln, d_head * 2)
        seq_lens: [l1, l2, ..., ln]
        """
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "... (nh dh) -> ... nh dh", dh=self.d_head) for t in (q, k, v))
        q, k = self.norm(q, k)
        q, k = apply_rope(q, k, pe, flux_mode=self.flux_mode)
        x = apply_attn(q, k, v, seq_lens, sequence_balancer=sequence_balancer)
        x = rearrange(x, "... nh dh -> ... (nh dh)")
        x = self.proj(x)
        return x


class DoubleStreamBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_head: int,
        mlp_ratio: float = 4.0,
        zero_init_modulation: bool = True,
        flux_mode: bool = True,
    ) -> None:
        """
        d_model: dimension of the model
        d_head: dimension of the head
        mlp_ratio: ratio of the hidden dimension to the model dimension
        zero_init_modulation: whether to initialize the modulation parameters to zero
        flux_mode: whether to use the flux mode for rope positional encoding
        """
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.flux_mode = flux_mode

        self.txt_mod = Modulation(d_model, double=True, zero_init=zero_init_modulation)
        self.txt_norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(d_model, d_head, flux_mode=flux_mode)
        self.txt_norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio), bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(d_model * mlp_ratio), d_model, bias=True),
        )

        self.img_mod = Modulation(d_model, double=True, zero_init=zero_init_modulation)
        self.img_norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(d_model, d_head, flux_mode=flux_mode)
        self.img_norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)

        self.img_mlp = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio), bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(d_model * mlp_ratio), d_model, bias=True),
        )

        self.init_weights()

    def init_weights(self) -> None:
        # Initialize txt modality components
        self.txt_mod.init_weights()
        self.txt_norm1.reset_parameters()
        self.txt_attn.init_weights()
        self.txt_norm2.reset_parameters()
        nn.init.trunc_normal_(self.txt_mlp[0].weight, std=0.02)
        nn.init.constant_(self.txt_mlp[0].bias, 0.0)
        nn.init.trunc_normal_(self.txt_mlp[2].weight, std=0.02)
        nn.init.constant_(self.txt_mlp[2].bias, 0.0)

        # Initialize img modality components
        self.img_mod.init_weights()
        self.img_norm1.reset_parameters()
        self.img_attn.init_weights()
        self.img_norm2.reset_parameters()
        nn.init.trunc_normal_(self.img_mlp[0].weight, std=0.02)
        nn.init.constant_(self.img_mlp[0].bias, 0.0)
        nn.init.trunc_normal_(self.img_mlp[2].weight, std=0.02)
        nn.init.constant_(self.img_mlp[2].bias, 0.0)

    def forward(
        self,
        packed_x: torch.Tensor,
        vec: torch.Tensor,
        packed_pe: torch.Tensor,
        seq_lens: list[int],
        seq_ids: torch.Tensor,
        txt_indices: torch.Tensor,
        img_indices: torch.Tensor,
        sequence_balancer: SequenceBalancer | None = None,
    ) -> torch.Tensor:
        """
        packed_x: (lt1+lv1+lt2+lv2+...+ltb+lvb, d)
        vec: (b, d) or (b1+b2+...+bn, d)
        packed_pe: (lt1+lv1+lt2+lv2+...+ltb+lvb, d_head*2)
        seq_lens: [l1, l2, ..., lb], where li=lti+lvi
        seq_ids: (lt1+lv1+lt2+lv2+...+ltb+lvb,)
        txt_indices: one-d-tensor of txt token indices
        img_indices: one-d-tensor of img token indices
        """
        assert img_indices.numel() > 0, "img_indices must be non-empty"

        # Due to SP, we might see zero txt tokens on some gpus
        if txt_indices.numel() > 0:
            # Process both txt and img tokens
            txt = packed_x[txt_indices]
            img = packed_x[img_indices]

            txt_mod1, txt_mod2 = self.txt_mod(vec, seq_ids[txt_indices])
            img_mod1, img_mod2 = self.img_mod(vec, seq_ids[img_indices])

            # Process txt tokens
            txt_modulated = (1 + txt_mod1.scale) * self.txt_norm1(txt) + txt_mod1.shift
            txt_qkv = self.txt_attn.qkv(txt_modulated)
            txt_q, txt_k, txt_v = rearrange(txt_qkv, "L (K H D) -> K L H D", K=3, H=self.txt_attn.n_heads)
            txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k)  # [L, H, D]

            # Process img tokens
            img_modulated = (1 + img_mod1.scale) * self.img_norm1(img) + img_mod1.shift
            img_qkv = self.img_attn.qkv(img_modulated)
            img_q, img_k, img_v = rearrange(img_qkv, "L (K H D) -> K L H D", K=3, H=self.img_attn.n_heads)
            img_q, img_k = self.img_attn.norm(img_q, img_k)  # [L, H, D]

            # Pack txt and img qkv
            q, k, v = (
                torch.empty(
                    (packed_x.shape[0], self.img_attn.n_heads, self.img_attn.d_head),
                    dtype=packed_x.dtype,
                    device=packed_x.device,
                )
                for _ in range(3)
            )
            q[txt_indices], k[txt_indices], v[txt_indices] = txt_q, txt_k, txt_v
            q[img_indices], k[img_indices], v[img_indices] = img_q, img_k, img_v

            q, k = apply_rope(q, k, packed_pe, flux_mode=self.flux_mode)
            attn = apply_attn(q, k, v, seq_lens, sequence_balancer=sequence_balancer)
            attn = rearrange(attn, "L H D -> L (H D)")

            # Apply attention to txt and img tokens
            txt_attn = attn[txt_indices]
            img_attn = attn[img_indices]

            txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
            img = img + img_mod1.gate * self.img_attn.proj(img_attn)

            # Run FFN block for txt tokens
            txt_modulated = (1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift
            txt = txt + txt_mod2.gate * self.txt_mlp(txt_modulated)

            # Run FFN block for img tokens
            img_modulated = (1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift
            img = img + img_mod2.gate * self.img_mlp(img_modulated)
            # Pack output
            output = torch.empty_like(packed_x)
            output[txt_indices] = txt
            output[img_indices] = img
        else:
            # Process only img tokens
            img = packed_x
            img_mod1, img_mod2 = self.img_mod(vec, seq_ids)

            img_modulated = (1 + img_mod1.scale) * self.img_norm1(img) + img_mod1.shift
            img_qkv = self.img_attn.qkv(img_modulated)
            img_q, img_k, img_v = rearrange(img_qkv, "L (K H D) -> K L H D", K=3, H=self.img_attn.n_heads)
            img_q, img_k = self.img_attn.norm(img_q, img_k)  # [L, H, D]

            q, k, v = img_q, img_k, img_v

            q, k = apply_rope(q, k, packed_pe, flux_mode=self.flux_mode)
            attn = apply_attn(q, k, v, seq_lens, sequence_balancer=sequence_balancer)
            attn = rearrange(attn, "L H D -> L (H D)")

            img = img + img_mod1.gate * self.img_attn.proj(attn)

            # Run FFN block
            img_modulated = (1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift
            img = img + img_mod2.gate * self.img_mlp(img_modulated)

            output = img

            # dummy op for all txt params to prevent FSDP hang
            txt_params = list(
                itertools.chain(
                    self.txt_mod.parameters(),
                    self.txt_norm1.parameters(),
                    self.txt_attn.parameters(),
                    self.txt_norm2.parameters(),
                    self.txt_mlp.parameters(),
                )
            )
            txt_params_mean = torch.cat([p.ravel() for p in txt_params]).mean()
            output = output + 0.0 * txt_params_mean

        return output


class SingleStreamBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_head: int,
        zero_init_modulation: bool = True,
        flux_mode: bool = True,
    ) -> None:
        """
        d_model: dimension of the model
        d_head: dimension of the head
        zero_init_modulation: whether to initialize the modulation parameters to zero
        flux_mode: whether to use the flux mode for rope positional encoding
        """
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.flux_mode = flux_mode

        self.modulation = Modulation(d_model, double=False, zero_init=zero_init_modulation)

        self.pre_norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)

        ffn_ratio = 4
        self.hidden_dim = ffn_ratio * d_model
        # qkv and mlp_in
        self.linear1 = nn.Linear(d_model, d_model * 3 + self.hidden_dim)
        # proj and mlp_out
        self.linear2 = nn.Linear(d_model + self.hidden_dim, d_model)

        self.norm = QKNorm(d_head)

        self.init_weights()

    def init_weights(self) -> None:
        self.modulation.init_weights()
        self.pre_norm.reset_parameters()
        nn.init.trunc_normal_(self.linear1.weight, std=0.02)
        nn.init.constant_(self.linear1.bias, 0.0)
        nn.init.trunc_normal_(self.linear2.weight, std=0.02)
        nn.init.constant_(self.linear2.bias, 0.0)
        self.norm.init_weights()

    def forward(
        self,
        packed_x: torch.Tensor,
        vec: torch.Tensor,
        packed_pe: torch.Tensor,
        seq_lens: list[int],
        seq_ids: torch.Tensor,
        txt_indices: torch.Tensor | None = None,
        img_indices: torch.Tensor | None = None,
        sequence_balancer: SequenceBalancer | None = None,
    ) -> torch.Tensor:
        """
        packed_x: (lt1+lv1+lt2+lv2+...+ltb+lvb, d)
        vec: (b, d) or (b1+b2+...+bn, d)
        packed_pe: (lt1+lv1+lt2+lv2+...+ltb+lvb, d_head * 2)
        seq_lens: [l1, l2, ..., lb], where li=lti+lvi
        seq_ids: (lt1+lv1+lt2+lv2+...+ltb+lvb,)
        txt_indices: one-d-tensor of txt token indices, or None
        img_indices: one-d-tensor of img token indices, or None
        """
        mod, _ = self.modulation(vec, seq_ids)

        # Run attention block
        x_modulated = (1 + mod.scale) * self.pre_norm(packed_x) + mod.shift
        qkv, mlp = self.linear1(x_modulated).split([3 * self.d_model, self.hidden_dim], dim=-1)
        q, k, v = rearrange(qkv, "L (K H D) -> K L H D", K=3, D=self.d_head)
        q, k = self.norm(q, k)
        q, k = apply_rope(q, k, packed_pe, flux_mode=self.flux_mode)
        attn = apply_attn(q, k, v, seq_lens, sequence_balancer=sequence_balancer)
        attn = rearrange(attn, "L H D -> L (H D)")
        packed_x = packed_x + mod.gate * self.linear2(torch.cat((attn, F.gelu(mlp, approximate="tanh")), dim=-1))
        return packed_x


class LastLayer(nn.Module):
    def __init__(self, d_model: int, d_img: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(d_model, d_img, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 2 * d_model, bias=True))

        self.init_weights()

    def init_weights(self) -> None:
        self.norm_final.reset_parameters()
        nn.init.trunc_normal_(self.linear.weight, std=0.02)
        nn.init.constant_(self.linear.bias, 0.0)
        nn.init.constant_(self.adaLN_modulation[1].weight, 0.0)
        nn.init.constant_(self.adaLN_modulation[1].bias, 0.0)

    def forward(self, packed_x: torch.Tensor, vec: torch.Tensor, seq_ids: torch.Tensor | None = None) -> torch.Tensor:
        """
        packed_x: (l1+l2+...+ln, d)
        vec: (b, d) or (b1+b2+...+bn, d)
        seq_ids: one-d-tensor of indices
        """
        assert packed_x.ndim == 2, f"packed_x must be a 2D tensor, but got {packed_x.ndim}"
        assert vec.ndim == 2, f"vec must be a 2D tensor, but got {vec.ndim}"

        mod = self.adaLN_modulation(vec)
        if seq_ids is not None:
            mod = torch.index_select(mod, 0, seq_ids)
        shift, scale = mod.chunk(2, dim=-1)
        packed_x = (1 + scale) * self.norm_final(packed_x) + shift
        packed_x = self.linear(packed_x)
        return packed_x


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def init_weights(self) -> None:
        nn.init.trunc_normal_(self.in_layer.weight, std=0.02)
        nn.init.constant_(self.in_layer.bias, 0.0)
        nn.init.trunc_normal_(self.out_layer.weight, std=0.02)
        nn.init.constant_(self.out_layer.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_layer(F.silu(self.in_layer(x)))  # type: ignore


def check_dtype(
    txt: torch.Tensor,
    txt_datum_lens: torch.Tensor,
    txt_position_ids: torch.Tensor,
    img: torch.Tensor,
    img_datum_lens: torch.Tensor,
    img_position_ids: torch.Tensor,
    t: torch.Tensor,
    vec: torch.Tensor | None = None,
    guidance: torch.Tensor | None = None,
) -> None:
    """
    Check that the dtypes of the tensors are correct.
    """
    assert txt.dtype in [
        torch.bfloat16,
        torch.float32,
    ], f"txt must be in torch.bfloat16 or torch.float32, but got {txt.dtype}"
    assert txt_datum_lens.dtype in [
        torch.int32,
        torch.int64,
    ], f"txt_datum_lens must be in torch.int32 or torch.int64, but got {txt_datum_lens.dtype}"
    assert txt_position_ids.dtype in [
        torch.int32,
        torch.int64,
    ], f"txt_position_ids must be in torch.int32 or torch.int64, but got {txt_position_ids.dtype}"
    assert img.dtype in [
        torch.bfloat16,
        torch.float32,
    ], f"img must be in torch.bfloat16 or torch.float32, but got {img.dtype}"
    assert img_datum_lens.dtype in [
        torch.int32,
        torch.int64,
    ], f"img_datum_lens must be in torch.int32 or torch.int64, but got {img_datum_lens.dtype}"
    assert img_position_ids.dtype in [
        torch.int32,
        torch.int64,
    ], f"img_position_ids must be in torch.int32 or torch.int64, but got {img_position_ids.dtype}"
    assert t.dtype == torch.float32, f"t must be in torch.float32, but got {t.dtype}"
    if vec is not None:
        assert vec.dtype in [
            torch.bfloat16,
            torch.float32,
        ], f"vec must be in torch.bfloat16 or torch.float32, but got {vec.dtype}"
    if guidance is not None:
        assert guidance.dtype in [
            torch.bfloat16,
            torch.float32,
        ], f"guidance must be in torch.bfloat16 or torch.float32, but got {guidance.dtype}"

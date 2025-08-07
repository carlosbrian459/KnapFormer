from collections.abc import Callable
import math

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

from knapformer.sequence_balancer import SequenceBalancer

if "H100" in torch.cuda.get_device_name():
    from flash_attn_interface import flash_attn_varlen_func
else:
    from flash_attn import flash_attn_varlen_func


@torch.autocast("cuda", enabled=False)  # type: ignore
def sinusoidal_encoding(position: torch.Tensor, dim: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Generate sinusoidal positional encoding for given position(s).

    This is the classic sinusoidal positional encoding from "Attention is All You Need".
    Unlike RoPE, this creates absolute positional embeddings that are added to token embeddings.
    The encoding concatenates cos and sin components: [cos_freqs, sin_freqs].

    Args:
        position: Position tensor with shape (...,) where ndim >= 1; typically in the range [0, 1000]
        dim: Dimension of the encoding (must be even)
        theta: Base frequency parameter (default: 10000.0)

    Returns:
        Tensor of shape (..., dim) with cos/sin encodings concatenated on same device as input.
        Structure: First dim//2 elements are cos values, next dim//2 are sin values.

    Examples:
        >>> # Single position
        >>> encoding = sinusoidal_encoding(torch.tensor([5]), 512)
        >>> print(encoding.shape)  # torch.Size([1, 512])

        >>> # Multiple positions
        >>> positions = torch.tensor([0, 1, 2, 3, 4])
        >>> encodings = sinusoidal_encoding(positions, 512)
        >>> print(encodings.shape)  # torch.Size([5, 512])

        >>> # Typical usage: add to token embeddings
        >>> token_embeddings = torch.randn(1, 100, 512)  # (batch, seq_len, dim)
        >>> pos_indices = torch.arange(100)  # (seq_len,)
        >>> pos_encodings = sinusoidal_encoding(pos_indices, 512)  # (seq_len, 512)
        >>> output = token_embeddings + pos_encodings.unsqueeze(0)  # Broadcast and add
    """
    if dim % 2 != 0:
        raise ValueError(f"dim must be even, got {dim}")

    if position.ndim == 0:
        raise ValueError("position must have at least 1 dimension")

    # Ensure position is float32
    position = position.to(dtype=torch.float32)

    # Create frequencies: theta^(-2i/dim) for i in [0, dim/2)
    freqs = theta ** (-torch.arange(0, dim, 2, device=position.device, dtype=torch.float32) / dim)

    # Compute position * frequencies
    pos_freqs = position.unsqueeze(-1) * freqs  # (..., dim//2)

    # Concatenate sin and cos
    pos_encoding = torch.cat(
        [
            torch.cos(pos_freqs),  # First half: cos values
            torch.sin(pos_freqs),  # Second half: sin values
        ],
        dim=-1,
    )

    return pos_encoding


@torch.autocast("cuda", enabled=False)  # type: ignore
def rope_encoding(
    position: torch.Tensor,
    d_head: int,
    theta: float = 10000,
) -> torch.Tensor:
    """
    Generate RoPE (Rotary Position Embedding) encodings.

    RoPE applies rotary transformations to position embeddings, which helps models
    better understand relative positions. The output contains cos/sin pairs that
    can be used to rotate query/key vectors in attention mechanisms.

    Args:
        position: Position tensor with shape (...,); typically in the range [0, 1000]
        d_head: Head dimension (must be even)
        theta: Base frequency parameter (default: 10000)

    Returns:
        Tensor of shape (..., d_head * 2) containing [cos_values, sin_values]
        The first d_head elements are cosines, the next d_head are sines

    Examples:
        >>> # Single position
        >>> pos = torch.tensor([5])
        >>> rope = rope_encoding(pos, 64)
        >>> print(rope.shape)  # torch.Size([1, 128])

        >>> # Multiple positions
        >>> positions = torch.tensor([0, 1, 2, 3])
        >>> ropes = rope_encoding(positions, 64)
        >>> print(ropes.shape)  # torch.Size([4, 128])

        >>> # Extract cos and sin components
        >>> cos_part = rope[:64]   # First half: cosines
        >>> sin_part = rope[64:]   # Second half: sines
    """
    if d_head % 2 != 0:
        raise ValueError(f"d_head must be even, got {d_head}")

    if position.ndim == 0:
        raise ValueError("position must have at least 1 dimension")

    # Ensure position is float32
    position = position.to(dtype=torch.float32)

    # Create frequencies: theta^(-2i/d_head) for i in [0, d_head/2)
    freqs = theta ** (-torch.arange(0, d_head, 2, device=position.device, dtype=torch.float32) / d_head)

    # Compute position * frequencies
    pos_freqs = position.unsqueeze(-1) * freqs  # (..., d_head//2)

    pos_freqs = torch.cat([pos_freqs, pos_freqs], dim=-1)  # (..., d_head), freqs for real and imaginary parts

    # Here, the cos values are organized as [cos_re, cos_im], (i.e., cos values applied to real and imaginary parts)
    # where both cos_re and cos_im have d_head//2 numbers
    # and so are the sin values.
    # So our memory layout for rope is [cos_re, cos_im, sin_re, sin_im], totaling d_head//2*4 numbers,
    # with each cos_re, cos_im, sin_re, sin_im being a chunk of d_head // 2 numbers.

    pos_encoding = torch.cat([torch.cos(pos_freqs), torch.sin(pos_freqs)], dim=-1)  # (..., d_head * 2)
    return pos_encoding


@torch.autocast("cuda", enabled=False)  # type: ignore
def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    rope_cossin: torch.Tensor,
    flux_mode: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE (Rotary Position Embedding) to query and key tensors.

    This function implements the core RoPE transformation by treating the d_head dimension
    as complex numbers and applying rotary transformations. Elements at index i and
    index i+d_head//2 form a complex number pair (real, imaginary) for i=0,1,...,d_head//2-1.
    The rotation is applied using complex multiplication: z * e^(iθ) = z * (cos(θ) + i*sin(θ)).

    Args:
        q: Query tensor with shape (..., num_heads, d_head)
        k: Key tensor with shape (..., num_heads, d_head)
        rope_cossin: RoPE cos/sin tensor with shape (..., d_head * 2) from rope_encoding()
                    First d_head elements are cosines, next d_head are sines
        flux_mode: Whether to use mode compatible with Flux denoiser

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Rotated (q, k) tensors with same shapes as input

    Shape Details:
        - Input: q, k have shape (..., num_heads, d_head)
        - Input: rope_cossin has shape (..., d_head * 2)
        - Output: q, k have shape (..., num_heads, d_head)

        Mathematical Operation:
        For each complex pair formed by elements at positions i and i+d_head//2:
        Let a = element[i], b = element[i+d_head//2]

        After rotation:
        - Real part:      a * cos(θ) - b * sin(θ)
        - Imaginary part: a * sin(θ) + b * cos(θ)

        Rotation trick used:
        (a, b) * cos + (-b, a) * sin = (a*cos - b*sin, a*sin + b*cos)

        Implementation splits d_head into two halves:
        - First half [0:d_head//2] contains real parts (a values)
        - Second half [d_head//2:d_head] contains imaginary parts (b values)

        Code implementation:
        - q_a, q_b = q.chunk(2, dim=-1)  # Split into real and imaginary parts
        - rotated = q * rope_cos + torch.cat([-q_b, q_a], dim=-1) * rope_sin
        - This uses the trick: (q_a, q_b) * cos + (-q_b, q_a) * sin

    Examples:
        >>> # Generate RoPE encodings for sequence
        >>> positions = torch.arange(10)  # sequence length 10
        >>> rope_cossin = rope_encoding(positions, d_head=64)  # (10, 128)
        >>>
        >>> # Query and key tensors from attention mechanism
        >>> q = torch.randn(1, 10, 8, 64)  # (batch, seq_len, num_heads, d_head)
        >>> k = torch.randn(1, 10, 8, 64)
        >>>
        >>> # Apply rotary position embedding
        >>> q_rotated, k_rotated = apply_rope(q, k, rope_cossin)
        >>> print(q_rotated.shape)  # (1, 10, 8, 64) - same as input

        >>> # Multi-dimensional case (e.g., 2D positions)
        >>> rope_2d = rope_encoding(torch.randn(32, 32, 2), d_head=64)  # (32, 32, 128)
        >>> q_2d = torch.randn(1, 32, 32, 8, 64)  # (batch, height, width, heads, d_head)
        >>> k_2d = torch.randn(1, 32, 32, 8, 64)
        >>> q_rot, k_rot = apply_rope(q_2d, k_2d, rope_2d)  # Preserves all dimensions

    Notes:
        - d_head must be even (split into two halves for complex number representation)
        - First half and second half of d_head form complex pairs: (element[i], element[i+d_head//2])
        - Dtype precision is preserved (casts to float for computation, then back)
        - Only q and k are rotated; values (v) in attention typically don't use RoPE
        - The rotation creates relative position awareness in attention weights
    """
    assert (
        len(rope_cossin.shape) == len(q.shape) - 1
        and rope_cossin.shape[:-1] == q.shape[:-2]
        and rope_cossin.shape[-1] == q.shape[-1] * 2
    ), f"invalid rope_cossin shape: {rope_cossin.shape} and q shape: {q.shape}"

    # a rotation is represented as cos(theta)+i*sin(theta)
    rope_cos, rope_sin = rope_cossin.float().unsqueeze(-2).chunk(2, dim=-1)

    if flux_mode:
        # Flux model takes [a0, b0, a1, b1, a2, b2, ...] layout
        # So here we change [a0, b0, a1, b1, a2, b2, ...] to [a0, a1, a2, ..., b0, b1, b2] layout
        # before applying RoPE
        q = rearrange(q, "... (nab dab) -> ... (dab nab)", dab=2)
        k = rearrange(k, "... (nab dab) -> ... (dab nab)", dab=2)

    # a complex number is represented as a+ib
    q_dtype = q.dtype
    q = q.float()
    q_a, q_b = q.chunk(2, dim=-1)
    q = q * rope_cos + torch.cat([-q_b, q_a], dim=-1) * rope_sin
    q = q.type(q_dtype)

    k_dtype = k.dtype
    k = k.float()
    k_a, k_b = k.chunk(2, dim=-1)
    k = k * rope_cos + torch.cat([-k_b, k_a], dim=-1) * rope_sin
    k = k.type(k_dtype)

    if flux_mode:
        # Change [a_0, a_1, a_2, ..., b_0, b_1, b_2] back to [a_0, b_0, a_1, b_1, a_2, b_2, ...] format
        # after applying RoPE
        q = rearrange(q, "... (dab nab) -> ... (nab dab)", dab=2)
        k = rearrange(k, "... (dab nab) -> ... (nab dab)", dab=2)

    return q, k


class MultiAxisPositionalEncoding:
    """
    Multi-axis positional encoding for handling multi-dimensional data.

    This class enables positional encoding along multiple axes (e.g., width, height, depth)
    using any base positional encoding function. Each axis can have different dimensions
    and frequency parameters. The final encoding concatenates cos/sin components from all axes.

    Args:
        pos_enc_fn: Base positional encoding function (e.g., sinusoidal_encoding, rope_encoding)
        n_axes: Number of spatial axes to encode
        axis_dim: Dimension(s) for each axis. If int, same dimension used for all axes.
                  If list, must have length n_axes
        theta: Frequency parameter(s). If float, same theta used for all axes.
               If list, must have length n_axes
        flux_mode: Whether to use mode compatible with Flux denoiser, default is False

    Examples:
        >>> # 2D positional encoding for images (width=256, height=256)
        >>> encoder_2d = MultiAxisPositionalEncoding(
        ...     pos_enc_fn=sinusoidal_encoding,
        ...     n_axes=2,
        ...     axis_dim=[128, 128],  # dims for width, height
        ...     theta=[10000.0, 10000.0]
        ... )
        >>>
        >>> # Position tensor: (batch, height, width, 2) where last dim is [x, y]
        >>> positions = torch.rand(4, 32, 32, 2)  # Random 2D positions
        >>> encodings = encoder_2d(positions)
        >>> print(encodings.shape)  # (4, 32, 32, 256) # 128*2 for each axis

        >>> # 1D case (equivalent to direct function call)
        >>> encoder_1d = MultiAxisPositionalEncoding(
        ...     pos_enc_fn=rope_encoding,
        ...     n_axes=1,
        ...     axis_dim=64
        ... )
        >>> pos_1d = torch.tensor([[0], [1], [2]])
        >>> enc_1d = encoder_1d(pos_1d)
        >>> print(enc_1d.shape)  # (3, 128)  # 64*2 from RoPE
    """

    def __init__(
        self,
        pos_enc_fn: Callable,
        n_axes: int,
        axis_dim: int | list[int],
        theta: float | list[float] = 10000.0,
        pos_scales: float | list[float] = 1.0,
        flux_mode: bool = False,
    ):
        self.pos_enc_fn = pos_enc_fn
        self.n_axes = n_axes
        if isinstance(axis_dim, int):
            axis_dim = [axis_dim] * n_axes
        if len(axis_dim) != n_axes:
            raise ValueError(f"axis_dim must be a list of length {n_axes}, got {len(axis_dim)}")
        self.axis_dim = axis_dim
        if isinstance(theta, float):
            theta = [theta] * n_axes
        if len(theta) != n_axes:
            raise ValueError(f"theta must be a list of length {n_axes}, got {len(theta)}")
        self.theta = theta
        if isinstance(pos_scales, float):
            pos_scales = [pos_scales] * n_axes
        if len(pos_scales) != n_axes:
            raise ValueError(f"pos_scales must be a list of length {n_axes}, got {len(pos_scales)}")
        self.pos_scales = pos_scales
        self.flux_mode = flux_mode

    @torch.autocast("cuda", enabled=False)  # type: ignore
    def __call__(self, position: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-axis positional encoding to position tensor.

        Args:
            position: Position tensor with shape (..., n_axes) where the last dimension
                     contains coordinates for each axis (e.g., [x, y] for 2D)

        Returns:
            Tensor with shape (..., total_dim) where total_dim is the sum of
            2 * axis_dim[i] for all axes. The output is structured as:
            [sin_axis0, sin_axis1, ..., cos_axis0, cos_axis1, ...]
            For 2D: [sin_x, sin_y, cos_x, cos_y]

        Raises:
            AssertionError: If position.shape[-1] != n_axes
        """
        if self.n_axes == 1:
            return self.pos_enc_fn(
                position * self.pos_scales[0], self.axis_dim[0], self.theta[0], flux_mode=self.flux_mode
            )  # type: ignore
        else:
            assert (
                position.shape[-1] == self.n_axes
            ), f"position must have {self.n_axes} dimensions, got {position.shape[-1]}"
            cos_list, sin_list = [], []
            for i in range(self.n_axes):
                cos_values, sin_values = self.pos_enc_fn(
                    position[..., i] * self.pos_scales[i],
                    self.axis_dim[i],
                    self.theta[i],
                ).chunk(2, dim=-1)

                # Here, each axis i's rope encoding is organized as
                # cos_values: [cos_re_axis_i, cos_im_axis_i] and sin_values: [sin_re_axis_i, sin_im_axis_i]
                # where i is the axis index, cos_re and cos_im are
                # the cosine values applied to real (a) and imaginary (b) parts (so are sin_re and sin_im).
                cos_list.append(cos_values)
                sin_list.append(sin_values)

            if not self.flux_mode:
                # We directly concatenate cos and sin list,
                # making the layout as
                # [cos_re_axis_0, cos_im_axis_0, cos_re_axis_1, cos_im_axis_1, ...,
                #  sin_re_axis_0, sin_im_axis_0, sin_re_axis_1, sin_im_axis_1, ...]
                return torch.cat(cos_list + sin_list, dim=-1)
            else:
                # If flux_mode is True, we need to rearrange the layout of cos and sin list
                # to be layout as
                # [cos_re_axis_0, cos_re_axis_1, ..., cos_im_axis_0, cos_im_axis_1, ...,
                #  sin_re_axis_0, sin_re_axis_1, ..., sin_im_axis_0, sin_im_axis_1, ...]
                # where want to put different axes' cos_re/im and sin_re/im in contiguous to each other.

                # ri: real/imag, na: number of elements
                cos_list = [rearrange(cos, "... (ri ne) -> ... ri ne", ri=2) for cos in cos_list]
                all_cos = rearrange(torch.cat(cos_list, dim=-1), "... ri ne -> ... (ri ne)")
                sin_list = [rearrange(sin, "... (ri ne) -> ... ri ne", ri=2) for sin in sin_list]
                all_sin = rearrange(torch.cat(sin_list, dim=-1), "... ri ne -> ... (ri ne)")
                return torch.cat([all_cos, all_sin], dim=-1)


def get_cu_seqlens(seq_lens: list[int] | torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(seq_lens, list):
        seq_lens = torch.tensor(seq_lens, device=device, dtype=torch.int32)
    max_seqlen = torch.max(seq_lens)
    cu_seqlens = F.pad(torch.cumsum(seq_lens, dim=0, dtype=torch.int32), (1, 0))
    return cu_seqlens, max_seqlen


def apply_attn(
    packed_q: torch.Tensor,
    packed_k: torch.Tensor,
    packed_v: torch.Tensor,
    seq_lens: list[int] | torch.Tensor,
    causal: bool = False,
    sequence_balancer: SequenceBalancer | None = None,
) -> torch.Tensor:
    """
    packed_q, packed_k, packed_v: (l1+l2+...+ln, h, d)
    seq_lens: [l1, l2, ..., ln]
    """
    if sequence_balancer is not None:
        seq_lens, packed_q, packed_k, packed_v = sequence_balancer.pre_attn(packed_q, packed_k, packed_v)

    # Tested with FA3
    # https://github.com/Dao-AILab/flash-attention/blob/34a3656b70711aed2383c4d486186e68ac1a2619/hopper/flash_attn_interface.py#L581
    cu_seqlens, max_seqlen = get_cu_seqlens(seq_lens, packed_q.device)

    packed_x: torch.Tensor = flash_attn_varlen_func(
        q=packed_q,
        k=packed_k,
        v=packed_v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        causal=causal,
    )

    if sequence_balancer is not None:
        seq_lens, packed_x = sequence_balancer.post_attn(packed_x)

    return packed_x


class Attention(nn.Module):
    def __init__(self, d_model: int, d_head: int, causal: bool = False):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.causal = causal

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.q_norm, self.k_norm = nn.RMSNorm(d_head), nn.RMSNorm(d_head)
        self.attn_out = nn.Linear(d_model, d_model)

        self.init_weights()

    def init_weights(self) -> None:
        nn.init.trunc_normal_(self.qkv_proj.weight, std=0.02)
        nn.init.constant_(self.qkv_proj.bias, 0.0)
        nn.init.constant_(self.q_norm.weight, 1.0)
        nn.init.constant_(self.k_norm.weight, 1.0)
        # nn.init.trunc_normal_(self.attn_out.weight, std=0.02)
        nn.init.constant_(self.attn_out.weight, 0.0)
        nn.init.constant_(self.attn_out.bias, 0.0)

    def forward(
        self,
        x: torch.Tensor,
        seq_lens: list[int],
        rope_cossin: torch.Tensor,
        sequence_balancer: SequenceBalancer | None = None,
    ) -> torch.Tensor:
        """
        x: (l1+l2+...+ln, d)
        seq_lens: [l1, l2, ..., ln]
        rope_cossin: (l1+l2+...+ln, d_head * 2)
        """
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "... (nh dh) -> ... nh dh", dh=self.d_head) for t in (q, k, v))
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, k, rope_cossin)
        x = apply_attn(q, k, v, seq_lens, self.causal, sequence_balancer)
        x = rearrange(x, "... nh dh -> ... (nh dh)")
        x = self.attn_out(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

        ffn_ratio, multiple_of = 8 / 3, 1024
        self.hidden_dim = int(math.ceil(ffn_ratio * d_model / multiple_of)) * multiple_of
        self.ffn_gateup_proj = nn.Linear(d_model, 2 * self.hidden_dim)
        self.ffn_down_proj = nn.Linear(self.hidden_dim, d_model)

        self.init_weights()

    def init_weights(self) -> None:
        nn.init.trunc_normal_(self.ffn_gateup_proj.weight, std=0.02)
        nn.init.constant_(self.ffn_gateup_proj.bias, 0.0)
        # nn.init.trunc_normal_(self.ffn_down_proj.weight, std=0.02)
        nn.init.constant_(self.ffn_down_proj.weight, 0.0)
        nn.init.constant_(self.ffn_down_proj.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.ffn_gateup_proj(x).chunk(2, dim=-1)
        x = self.ffn_down_proj(F.silu(gate) * up)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, d_head: int, causal: bool = False):
        super().__init__()
        self.attn_norm = nn.RMSNorm(d_model)
        self.attn = Attention(d_model, d_head, causal)

        self.ffn_norm = nn.RMSNorm(d_model)
        self.ffn = FeedForward(d_model)

        self.init_weights()

    def init_weights(self) -> None:
        nn.init.constant_(self.attn_norm.weight, 1.0)
        self.attn.init_weights()
        nn.init.constant_(self.ffn_norm.weight, 1.0)
        self.ffn.init_weights()

    def forward(
        self,
        x: torch.Tensor,
        seq_lens: list[int],
        rope_cossin: torch.Tensor,
        sequence_balancer: SequenceBalancer | None = None,
    ) -> torch.Tensor:
        """
        x: (l1+l2+...+ln, d)
        seq_lens: [l1, l2, ..., ln]
        rope_cossin: (l1+l2+...+ln, d_head * 2)
        """
        attn_in = self.attn_norm(x)
        x = x + self.attn(attn_in, seq_lens, rope_cossin, sequence_balancer)

        ffn_in = self.ffn_norm(x)
        x = x + self.ffn(ffn_in)
        return x


def get_rope_cossin(seq_lens: list[int], d_head: int, device: torch.device):
    """
    seq_lens: [l1, l2, ..., ln]
    d_head: d_head
    """
    position = torch.cat([torch.arange(seq_len, device=device) for seq_len in seq_lens], dim=0)
    rope_cossin = rope_encoding(position, d_head)

    return rope_cossin

import torch
from typing import Optional, Union

def apply_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    interleaved: bool = False,
    inplace: bool = False,
    conjugate: bool = False,
) -> torch.Tensor:
    """
    Applies rotary embeddings to the first part of the head dimension.
    
    Args:
        x: Tensor of shape
             - (batch, seqlen, nheads, headdim) if cu_seqlens is None, or
             - (total_seqlen, nheads, headdim) if using variable lengths.
        cos: Precomputed cosine table of shape (seqlen_ro, rotary_dim/2)
        sin: Precomputed sine table of shape (seqlen_ro, rotary_dim/2)
        seqlen_offsets: Either an int or a tensor of shape (batch,) specifying the offset(s)
                        to add to the time index.
        cu_seqlens: If provided, a tensor of shape (batch+1,) with cumulative sequence lengths.
        max_seqlen: Maximum sequence length (required if cu_seqlens is provided).
        interleaved: If True, assumes that the rotary dimensions are interleaved.
        inplace: Whether to apply the transformation in-place.
        conjugate: If True, applies the conjugate version (i.e. negates sin).
        
    Returns:
        Tensor with rotary embeddings applied:
            - (batch, seqlen, nheads, headdim) if cu_seqlens is None,
            - (batch, max_seqlen, nheads, headdim) otherwise.
    """
    # Determine rotary dimension (rotary_dim = 2 * (cos.shape[1]))
    d_half = cos.shape[1]
    rotary_dim = d_half * 2

    if conjugate:
        sin = -sin

    if cu_seqlens is not None:
        # ---- Variable-length case ----
        # x is (total_seqlen, nheads, headdim)
        # We need to build a padded output tensor of shape (B, max_seqlen, nheads, headdim)
        assert max_seqlen is not None, "max_seqlen must be provided when using cu_seqlens."
        B = cu_seqlens.shape[0] - 1
        nheads = x.shape[1]
        headdim = x.shape[2]
        out = x.new_empty((B, max_seqlen, nheads, headdim))
        for b in range(B):
            start = int(cu_seqlens[b].item())
            end = int(cu_seqlens[b + 1].item())
            L = end - start  # actual sequence length for batch b
            # Extract the b-th batch slice
            xb = x[start:end]  # shape: (L, nheads, headdim)
            # Respect inplace: if not inplace, work on a clone
            if not inplace:
                xb = xb.clone()
            # Use the helper below; note that we pass a per-batch offset if seqlen_offsets is a tensor.
            offset = seqlen_offsets[b] if not isinstance(seqlen_offsets, int) else seqlen_offsets
            xb = _apply_rotary_batched(xb.unsqueeze(0), cos, sin, offset, interleaved, inplace)
            # Remove the extra batch dim (since _apply_rotary_batched always works on 4D tensors)
            out[b, :L] = xb[0]
            if L < max_seqlen:
                out[b, L:] = x.new_zeros((max_seqlen - L, nheads, headdim))
        return out
    else:
        # ---- Fixed-length (batched) case ----
        # x is (B, L, nheads, headdim)
        if not inplace:
            x = x.clone()
        return _apply_rotary_batched(x, cos, sin, seqlen_offsets, interleaved, inplace)

def _apply_rotary_batched(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seqlen_offset: Union[int, torch.Tensor],
    interleaved: bool,
    inplace: bool = False,
) -> torch.Tensor:
    """
    Applies the rotary transformation to a batched tensor.
    
    x: Tensor of shape (B, L, nheads, headdim)
    cos, sin: Tensors of shape (seqlen_ro, rotary_dim/2)
    seqlen_offset: Either an int or (if x is batched) a tensor of shape (B,)
    interleaved: Whether the rotary dimensions are interleaved.
    inplace: If True, modify x in-place.
    """
    B, L, H, D = x.shape
    d_half = cos.shape[1]
    rotary_dim = d_half * 2
    device = x.device

    # Create a time index for the sequence dimension.
    t = torch.arange(L, device=device)  # shape: (L,)
    
    if isinstance(seqlen_offset, int):
        t_idx = t + seqlen_offset  # (L,)
        valid = t_idx < cos.shape[0]
        t_idx_clamped = t_idx.clamp(max=cos.shape[0] - 1)
        cos_t = cos[t_idx_clamped].unsqueeze(0).unsqueeze(2)  # (1, L, 1, d_half)
        sin_t = sin[t_idx_clamped].unsqueeze(0).unsqueeze(2)
        cos_t = torch.where(valid.unsqueeze(-1), cos_t, torch.ones_like(cos_t))
        sin_t = torch.where(valid.unsqueeze(-1), sin_t, torch.zeros_like(sin_t))
    else:
        seqlen_offset = seqlen_offset.to(device)
        t_exp = t.unsqueeze(0).expand(B, L)  # (B, L)
        t_idx = t_exp + seqlen_offset.unsqueeze(1)  # (B, L)
        valid = t_idx < cos.shape[0]
        t_idx_clamped = t_idx.clamp(max=cos.shape[0] - 1)
        cos_t = cos[t_idx_clamped]  # (B, L, d_half)
        sin_t = sin[t_idx_clamped]  # (B, L, d_half)
        cos_t = torch.where(valid.unsqueeze(-1), cos_t, torch.ones_like(cos_t))
        sin_t = torch.where(valid.unsqueeze(-1), sin_t, torch.zeros_like(sin_t))
        cos_t = cos_t.unsqueeze(2)  # (B, L, 1, d_half)
        sin_t = sin_t.unsqueeze(2)
    
    # Process only the first rotary_dim components.
    x_rot = x[..., :rotary_dim]  # (B, L, H, rotary_dim)
    
    if not interleaved:
        # Non-interleaved: the first half and second half are contiguous.
        x1 = x_rot[..., :d_half]  # (B, L, H, d_half)
        x2 = x_rot[..., d_half:rotary_dim]  # (B, L, H, d_half)
        out1 = x1 * cos_t - x2 * sin_t
        out2 = x1 * sin_t + x2 * cos_t
        rotated = torch.cat([out1, out2], dim=-1)  # (B, L, H, rotary_dim)
    else:
        # Interleaved: assume x_rot is arranged as (x0, x1, x2, x3, ...)
        # Reshape to group pairs.
        x_rot = x_rot.view(B, L, H, d_half, 2)  # (B, L, H, d_half, 2)
        x1 = x_rot[..., 0]  # (B, L, H, d_half)
        x2 = x_rot[..., 1]  # (B, L, H, d_half)
        out1 = x1 * cos_t - x2 * sin_t
        out2 = x1 * sin_t + x2 * cos_t
        rotated = torch.stack([out1, out2], dim=-1).view(B, L, H, rotary_dim)
    
    # Write back the rotated values.
    # If inplace, update x directly; otherwise, x is already a clone.
    x[..., :rotary_dim] = rotated
    return x
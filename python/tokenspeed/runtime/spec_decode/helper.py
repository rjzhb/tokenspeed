# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Speculative decoding helpers for the draft head's first-step active-row slice.

The draft head emits a hidden state for every input token but downstream only
consumes the per-request last token (lm_head + next draft step). These helpers
drop the dead-position rows so MLP / norm / collective ops only touch live
rows.

Usage in a single layer:

    attn_output = self.attn(...)
    attn_output = apply_draft_active_row_slice_pre_oproj(attn_output, ctx)    # before o_proj
    output = self.o_proj(attn_output)
    ...
    hidden_states, residual = apply_draft_active_row_slice_post_attn(
        hidden_states, residual, ctx,
    )                                                                # once per layer
"""

from __future__ import annotations

import torch

from tokenspeed.runtime.execution.context import ForwardContext


def apply_draft_active_row_slice_pre_oproj(
    tensor: torch.Tensor, ctx: ForwardContext
) -> torch.Tensor:
    """Gather ``tensor`` to one row per request; no-op if not requested or idle."""
    if not ctx.draft_active_row_slice or ctx.gather_ids is None:
        return tensor
    return tensor.index_select(0, ctx.gather_ids)


def apply_draft_active_row_slice_post_attn(
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    ctx: ForwardContext,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Finalize the active-row slice after the layer's self-attention.

    Active rank: gather ``residual`` to match the already-sliced ``attn_output``
    and update ``ctx.input_num_tokens``.

    All ranks (active + idle): switch ``ctx.global_num_tokens`` to
    ``ctx.global_bs`` so cross-rank MoE / RSAG see consistent scatter sizes,
    then clear ``gather_ids`` / ``draft_active_row_slice`` so downstream
    layernorms, MLP, and final-norm don't double-slice.
    """
    if not ctx.draft_active_row_slice:
        return hidden_states, residual

    # Active rank: attn module already sliced its output via
    # apply_draft_active_row_slice_pre_oproj; line up residual + per-rank token count.
    gather_ids = ctx.gather_ids
    if gather_ids is not None:
        assert hidden_states.size(0) == gather_ids.size(0), (
            "attn module must call apply_draft_active_row_slice_pre_oproj before this"
        )
        if residual is not None and residual.size(0) != gather_ids.size(0):
            residual = residual.index_select(0, gather_ids)
        ctx.input_num_tokens = gather_ids.size(0)

    # All ranks: switch DP-global counts so collectives agree across the world,
    # then clear flags so the layer state is coherent for downstream ops.
    if ctx.global_bs is not None:
        ctx.global_num_tokens = ctx.global_bs
    ctx.gather_ids = None
    ctx.draft_active_row_slice = False

    return hidden_states, residual

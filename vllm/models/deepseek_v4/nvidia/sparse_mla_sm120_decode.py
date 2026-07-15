# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer ``BatchSparseMLAPagedAttentionWrapper`` decode adapter (SM120/121).

The raw ``trtllm_batch_decode_sparse_mla_dsv4`` entry point dispatches only
single-query decode (``q_len == 1``) and prefill (``num_tokens > 64``). The MTP
speculative *verify* step needs the multi-query band (``1 < q_len <= 64``),
which only ``BatchSparseMLAPagedAttentionWrapper`` (FlashInfer's sparse-MLA
SM120 wrapper module) dispatches, via ``sparse_mla_sm120_decode_dsv4``. This
adapter routes the DSV4 SM120 decode through that wrapper so speculative
decode works on GB10 (SM121).

Enabled by default; set ``VLLM_DSV4_SM120_WRAPPER_DECODE=0`` to fall back to
the raw single-query path (speculative decode will then fail its warmup with
``num_tokens > 64``).

CUDA-graph safety (hard-learned on 2x DGX Spark, TP=2 over RoCE):

* The shared workspace arena used for the decode split-K scratch is reserved
  ONCE at its lifetime maximum, before any capture (``_reserve_decode_scratch``).
  Per-call requests vary by layer/phase (padded topk 128/512, extra_topk
  0/512/1664); if a later, larger request arrived after FULL-graph capture the
  arena would reallocate and every captured graph's kernels would point at
  freed memory.
* The wrapper call passes NO ``topk_length``/``extra_topk_length``. Both index
  tensors are -1-padded beyond their valid lengths by the metadata builders
  (``sparse_swa.py`` fill kernels, ``cache_utils`` c128a kernels), and the
  decode kernel masks ``idx < 0`` to -inf, so the length tensors are redundant
  for correctness. They are not harmless: the kernel derives its chunk counts
  (mbarrier producer/consumer loop ranges) from ``topk_length_ptr`` at runtime,
  which under FULL-graph replay is a live re-read of a shared metadata buffer
  on every replay. Without the length pointers the chunk counts come only from
  the index WIDTHS -- launch constants -- making the kernel control flow
  replay-proof, matching the fork backend's ``forward_mqa`` call shape.
"""
from __future__ import annotations

import os

import torch

from vllm.logger import init_logger
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_logger(__name__)

_DECODE_SPLIT_TILE = 64

# Decode kernel hard cap on num_tokens (multi-query band).
_DECODE_MAX_TOKENS = 64

# Upper bound on split-K count for the init-time workspace reservation:
# swa topk<=1024 (16 splits) + compressed extra_topk (observed up to 1664 ->
# 26 splits) with headroom. Override with VLLM_DSV4_SM120_MAX_SPLITS.
_MAX_RESERVED_SPLITS = int(os.getenv("VLLM_DSV4_SM120_MAX_SPLITS", "64") or "64")

# The decode_dsv4 kernel only dispatches these swa topk widths
# (see flashinfer's _DECODE_DSV4_DISPATCH).
_SUPPORTED_TOPK = (128, 512, 1024)

_scratch_reserved = False


def _cdiv(x: int, y: int) -> int:
    return (int(x) + int(y) - 1) // int(y)


def _decode_num_splits(topk: int, extra_topk: int = 0) -> int:
    return _cdiv(topk, _DECODE_SPLIT_TILE) + _cdiv(extra_topk, _DECODE_SPLIT_TILE)


def use_sm120_wrapper_decode() -> bool:
    v = os.getenv("VLLM_DSV4_SM120_WRAPPER_DECODE", "1").strip().lower()
    return v not in ("0", "false", "no", "off", "")


def _resolve_max_num_tokens() -> int:
    # The vLLM config context is not set during worker execution, so fall back
    # to the env/default when unavailable.
    try:
        from vllm.config import get_current_vllm_config

        n = int(get_current_vllm_config().scheduler_config.max_num_batched_tokens)
        if n > 0:
            return n
    except Exception:
        pass
    return int(os.getenv("MAX_NUM_BATCHED_TOKENS", "8192") or "8192")


def _reserve_decode_scratch(padded_heads: int, d_v: int, max_num_tokens: int) -> None:
    """Pre-size the shared workspace arena to its lifetime maximum, once,
    before any CUDA-graph capture (the first call happens during the eager
    warmup/profile run). Mirrors the fork backend's __init__ reservation."""
    global _scratch_reserved
    if _scratch_reserved:
        return
    max_tok = min(int(max_num_tokens), _DECODE_MAX_TOKENS)
    current_workspace_manager().get_simultaneous(
        ((max_tok, padded_heads, _MAX_RESERVED_SPLITS, d_v), torch.bfloat16),
        ((max_tok, padded_heads, _MAX_RESERVED_SPLITS), torch.float32),
    )
    _scratch_reserved = True
    logger.info(
        "SM120 sparse-MLA decode scratch reserved: tokens<=%d heads=%d "
        "max_splits=%d d_v=%d (arena fixed before cudagraph capture)",
        max_tok, padded_heads, _MAX_RESERVED_SPLITS, d_v,
    )


def _get_wrapper(layer, *, padded_heads: int):
    ws = getattr(layer, "_sparse_mla_sm120_wrapper", None)
    if (
        ws is not None
        and getattr(layer, "_sparse_mla_sm120_wrapper_heads", 0) == padded_heads
    ):
        return ws
    from flashinfer.sparse_mla_sm120 import BatchSparseMLAPagedAttentionWrapper

    # max_num_tokens is only a pre-allocation bound for the wrapper's out_lse;
    # decode num_tokens never exceeds max_num_batched_tokens.
    max_num_tokens = _resolve_max_num_tokens()
    ws = BatchSparseMLAPagedAttentionWrapper(
        max_num_tokens=max_num_tokens,
        max_num_heads=int(padded_heads),
        d_v=512,
    )
    _reserve_decode_scratch(int(padded_heads), 512, max_num_tokens)
    layer._sparse_mla_sm120_wrapper = ws
    layer._sparse_mla_sm120_wrapper_heads = int(padded_heads)
    logger.info_once(
        "DeepSeek V4 SM120 sparse-MLA WRAPPER decode enabled "
        "(multi-query capable; replaces raw trtllm_batch_decode_sparse_mla_dsv4)."
    )
    return ws


def sm120_wrapper_decode(
    layer,
    *,
    q: torch.Tensor,
    swa_kv_cache: torch.Tensor,
    indexed_kv_cache: torch.Tensor | None,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    extra_sparse_indices: torch.Tensor | None,
    extra_sparse_lengths: torch.Tensor | None,
    output: torch.Tensor,
) -> None:
    """Decode via the sparse-MLA SM120 wrapper. ``q`` is pre-padded to
    ``output.shape[1]`` heads by the caller (``_prepare_query``).

    ``swa_lens``/``extra_sparse_lengths`` are accepted for call-site symmetry
    with the raw path but deliberately NOT forwarded -- see the module
    docstring (replay-proof kernel control flow; -1 index padding already
    provides masking)."""
    del swa_lens, extra_sparse_lengths  # correctness carried by -1 padding
    num_decode_tokens = q.shape[0]
    padded_heads = output.shape[1]
    topk = int(swa_indices.shape[-1])
    # Pad the swa indices up to the next kernel-supported width with -1
    # (invalid slots the kernel masks). The spec-verify step produces topk=256,
    # which is not in the dispatch set and would otherwise fall through to the
    # paged kernel (asserts num_tokens > 64).
    if topk not in _SUPPORTED_TOPK:
        target = next((t for t in _SUPPORTED_TOPK if t >= topk), None)
        if target is not None and target > topk:
            pad = swa_indices.new_full((*swa_indices.shape[:-1], target - topk), -1)
            swa_indices = torch.cat([swa_indices, pad], dim=-1)
            topk = target
    extra_topk = (
        int(extra_sparse_indices.shape[-1])
        if extra_sparse_indices is not None
        else 0
    )
    num_splits = _decode_num_splits(topk, extra_topk)
    if num_splits > _MAX_RESERVED_SPLITS:
        # A request beyond the init-time reservation would grow the arena and
        # invalidate every captured graph.
        logger.warning_once(
            "SM120 decode num_splits=%d exceeds reserved max %d; the workspace "
            "arena will REALLOCATE and captured cudagraphs become invalid. "
            "Raise VLLM_DSV4_SM120_MAX_SPLITS.",
            num_splits, _MAX_RESERVED_SPLITS,
        )
    mid_out, mid_lse = current_workspace_manager().get_simultaneous(
        (
            (num_decode_tokens, padded_heads, num_splits, output.shape[-1]),
            torch.bfloat16,
        ),
        ((num_decode_tokens, padded_heads, num_splits), torch.float32),
    )

    wrapper = _get_wrapper(layer, padded_heads=padded_heads)
    # The wrapper wants the raw cache with a singleton kv-head dim:
    # [num_pages, page_block_size, 1, kv_bytes_per_token].
    swa_cache = swa_kv_cache.unsqueeze(-2)
    extra_cache = (
        indexed_kv_cache.unsqueeze(-2) if indexed_kv_cache is not None else None
    )
    wrapper.run(
        q=q,
        kv_cache=swa_cache,
        indices=swa_indices,
        output=output,
        sm_scale=layer.scale,
        attn_sink=layer.attn_sink,
        extra_kv_cache=extra_cache,
        extra_indices=extra_sparse_indices,
        mid_out=mid_out,
        mid_lse=mid_lse,
    )

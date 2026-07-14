# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for DSpark TP draft-loop optimizations.

Covers:
- ``top_tokens_from_local_logits`` pair-reduce vs a plain gathered ``argmax``
  (single- and simulated multi-rank, padding masking, tie-breaking).
- ``LogitsProcessor.get_top_tokens`` composing ``compute_local_logits`` with
  the pair reduce (scale/soft-cap parity with ``forward``).
- ``DSparkMarkovHead(replicate_w1=...)`` module selection.
- ``DSparkSpeculator._validate_local_argmax_reduction`` startup validation.
"""

import types
from unittest import mock

import pytest
import torch

from vllm.model_executor.layers.logits_processor import (
    LogitsProcessor,
    top_tokens_from_local_logits,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)

_LP_MODULE = "vllm.model_executor.layers.logits_processor"


def _shard_indices(
    num_org_vocab_padding: int = 0, org_vocab_start_index: int = 0
) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        num_org_vocab_padding=num_org_vocab_padding,
        org_vocab_start_index=org_vocab_start_index,
    )


class _FakeLmHead:
    # cpu_linear mirrors VocabParallelEmbedding's attribute so the
    # CPU-platform UnquantizedEmbeddingMethod dispatch works on the fake.
    cpu_linear = staticmethod(torch.nn.functional.linear)

    def __init__(self, weight: torch.Tensor, shard_indices: types.SimpleNamespace):
        self.weight = weight
        self.quant_method = UnquantizedEmbeddingMethod()
        self.shard_indices = shard_indices


def test_top_tokens_from_local_logits_tp1_matches_argmax():
    torch.manual_seed(0)
    logits = torch.randn(8, 64)
    lm_head = types.SimpleNamespace(shard_indices=_shard_indices())
    with mock.patch(
        f"{_LP_MODULE}.get_tensor_model_parallel_world_size", return_value=1
    ):
        top = top_tokens_from_local_logits(logits.clone(), lm_head)
    assert torch.equal(top, logits.argmax(dim=-1))


def test_top_tokens_from_local_logits_masks_vocab_padding():
    torch.manual_seed(0)
    num_pad = 8
    logits = torch.randn(4, 64)
    # Plant maxima inside the padding region; they must be ignored.
    logits[:, -num_pad:] = 100.0
    lm_head = types.SimpleNamespace(
        shard_indices=_shard_indices(num_org_vocab_padding=num_pad)
    )
    with mock.patch(
        f"{_LP_MODULE}.get_tensor_model_parallel_world_size", return_value=1
    ):
        top = top_tokens_from_local_logits(logits.clone(), lm_head)
    assert torch.equal(top, logits[:, :-num_pad].argmax(dim=-1))
    assert (top < 64 - num_pad).all()


def _simulated_two_rank_top_tokens(
    shard0: torch.Tensor, shard1: torch.Tensor
) -> torch.Tensor:
    """Run the pair reduce as rank 0 of a simulated TP=2 group."""
    shard_size = shard0.shape[-1]
    head0 = types.SimpleNamespace(
        shard_indices=_shard_indices(org_vocab_start_index=0)
    )

    def fake_all_gather(local_pair: torch.Tensor, dim: int) -> torch.Tensor:
        # Build rank 1's (value, global index) pair the same way the real
        # function does, then concatenate as a gloo/nccl all-gather would.
        vals1, idx1 = shard1.max(dim=-1)
        pair1 = torch.stack([vals1.float(), (idx1 + shard_size).float()], dim=-1)
        return torch.cat([local_pair, pair1], dim=dim)

    with (
        mock.patch(
            f"{_LP_MODULE}.get_tensor_model_parallel_world_size", return_value=2
        ),
        mock.patch(
            f"{_LP_MODULE}.tensor_model_parallel_all_gather",
            side_effect=fake_all_gather,
        ),
    ):
        return top_tokens_from_local_logits(shard0.clone(), head0)


def test_top_tokens_from_local_logits_pair_reduce_two_ranks():
    torch.manual_seed(0)
    full = torch.randn(8, 64)
    shard0, shard1 = full[:, :32], full[:, 32:]
    top = _simulated_two_rank_top_tokens(shard0, shard1)
    assert torch.equal(top, full.argmax(dim=-1))


def test_top_tokens_from_local_logits_tie_breaks_to_lowest_index():
    # Exact same max value on both shards: the reduce must pick the lowest
    # global index, matching a gathered argmax.
    full = torch.zeros(2, 8)
    full[0, 1] = full[0, 6] = 5.0  # tie across shards -> expect 1
    full[1, 5] = full[1, 7] = 3.0  # tie within shard 1 -> expect 5
    shard0, shard1 = full[:, :4], full[:, 4:]
    top = _simulated_two_rank_top_tokens(shard0, shard1)
    assert torch.equal(top, full.argmax(dim=-1))
    assert torch.equal(top, torch.tensor([1, 5]))


def test_get_top_tokens_matches_forward_argmax_with_scale_and_soft_cap(
    default_vllm_config,
):
    torch.manual_seed(0)
    vocab_size, hidden_size = 64, 16
    lp = LogitsProcessor(vocab_size, scale=0.5, soft_cap=30.0)
    lp._gather_logits = lambda logits: logits  # TP gather is orthogonal here.
    hidden = torch.randn(4, hidden_size)
    lm_head = _FakeLmHead(torch.randn(vocab_size, hidden_size), _shard_indices())

    with mock.patch(
        f"{_LP_MODULE}.get_tensor_model_parallel_world_size", return_value=1
    ):
        top = lp.get_top_tokens(lm_head, hidden)
        forward_logits = lp.forward(lm_head, hidden, None)

    assert torch.equal(top, forward_logits.argmax(dim=-1))


def test_markov_head_replicate_w1_selects_plain_embedding():
    import vllm.model_executor.models.qwen3_dspark as qd

    class _StubParallelLMHead(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    with (
        mock.patch.object(qd, "ParallelLMHead", _StubParallelLMHead),
        mock.patch.object(qd, "VocabParallelEmbedding", _StubParallelLMHead),
    ):
        replicated = qd.DSparkMarkovHead(64, 64, 8, prefix="m", replicate_w1=True)
        default = qd.DSparkMarkovHead(64, 64, 8, prefix="m")

    assert isinstance(replicated.markov_w1, torch.nn.Embedding)
    assert replicated.markov_w1.weight.shape == (64, 8)
    assert not replicated.markov_w1.weight.requires_grad
    # embed() is a local lookup on the full table.
    ids = torch.tensor([3, 7])
    assert torch.equal(
        replicated.embed(ids), replicated.markov_w1.weight[ids]
    )
    # Default path keeps the vocab-parallel module (stubbed here).
    assert isinstance(default.markov_w1, _StubParallelLMHead)


def _make_validator_speculator(
    *,
    use_local: bool,
    model: object,
    draft_sample_method: str = "greedy",
):
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    spec = DSparkSpeculator.__new__(DSparkSpeculator)
    spec.use_local_argmax_reduction = use_local
    spec.speculative_config = types.SimpleNamespace(
        draft_sample_method=draft_sample_method
    )
    spec.model = model
    return spec


def _full_protocol_model(draft_id_to_target_id=None) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        get_top_tokens=lambda h: h,
        compute_local_draft_logits=lambda h: h,
        local_markov_bias=lambda e: e,
        local_draft_top_tokens=lambda l: l,
        draft_id_to_target_id=draft_id_to_target_id,
    )


def test_dspark_validator_accepts_full_protocol():
    spec = _make_validator_speculator(use_local=True, model=_full_protocol_model())
    spec._validate_local_argmax_reduction()  # must not raise


def test_dspark_validator_noop_when_flag_off():
    spec = _make_validator_speculator(
        use_local=False, model=types.SimpleNamespace()
    )
    spec._validate_local_argmax_reduction()  # must not raise


def test_dspark_validator_rejects_missing_local_protocol():
    model = types.SimpleNamespace(
        get_top_tokens=lambda h: h, draft_id_to_target_id=None
    )
    spec = _make_validator_speculator(use_local=True, model=model)
    with pytest.raises(ValueError, match="compute_local_draft_logits"):
        spec._validate_local_argmax_reduction()


def test_dspark_validator_rejects_reduced_draft_vocab():
    model = _full_protocol_model(draft_id_to_target_id=torch.tensor([0, 2]))
    spec = _make_validator_speculator(use_local=True, model=model)
    with pytest.raises(ValueError, match="full-vocab"):
        spec._validate_local_argmax_reduction()


def test_dspark_validator_rejects_probabilistic():
    spec = _make_validator_speculator(
        use_local=True,
        model=_full_protocol_model(),
        draft_sample_method="probabilistic",
    )
    with pytest.raises(ValueError, match="probabilistic"):
        spec._validate_local_argmax_reduction()

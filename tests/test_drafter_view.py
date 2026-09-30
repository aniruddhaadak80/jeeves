"""build_anchor_layout rejects block sizes that cannot describe a prediction target."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drafter.view import build_anchor_layout


def layout(L=16, block=4, n_blocks=3, seed=0):
    return build_anchor_layout(L, block, n_blocks, torch.Generator().manual_seed(seed), "cpu")


def test_a_valid_layout_is_unchanged():
    mask_pos, attn_mask = layout()
    assert mask_pos.numel() == 3 * 3
    assert mask_pos.dtype == torch.long
    assert tuple(attn_mask.shape) == (1, 1, 3 * 3, 16 + 3 * 3)
    assert int(mask_pos.min()) >= 1


def test_block_one_has_no_mask_position_and_is_refused():
    with pytest.raises(ValueError) as excinfo:
        layout(block=1)
    assert ">= 2" in str(excinfo.value)


def test_block_zero_is_refused():
    with pytest.raises(ValueError):
        layout(block=0)


def test_a_block_longer_than_the_row_is_refused():
    # block == L is the tightest legal case (anchor at 0, predicting L-1 positions); only a block
    # past the row has nowhere to anchor.
    mask_pos, _ = layout(L=8, block=8, n_blocks=1)
    assert mask_pos.numel() == 7
    with pytest.raises(ValueError) as excinfo:
        layout(L=8, block=9)
    assert "does not fit" in str(excinfo.value)


def test_zero_anchor_blocks_is_refused():
    with pytest.raises(ValueError) as excinfo:
        layout(n_blocks=0)
    assert "n_blocks" in str(excinfo.value)


def test_n_blocks_is_still_clamped_to_what_the_row_holds():
    # 16 tokens, block 4 -> at most 13 anchors fit; asking for more is capped, not an error.
    mask_pos, _ = layout(L=16, block=4, n_blocks=100)
    assert mask_pos.numel() == 13 * 3

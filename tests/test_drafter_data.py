"""Chain loading and packing fail loudly instead of handing the trainer an empty stream."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drafter.data import Batches, load_streams, pack


class FakeEncoder:
    think_end_id = 2


def test_a_pattern_matching_nothing_raises_instead_of_returning_no_streams(tmp_path):
    pattern = str(tmp_path / "chains.rank*.jsonl")
    with pytest.raises(FileNotFoundError) as excinfo:
        load_streams(pattern, encoder=FakeEncoder())
    assert "chains.rank" in str(excinfo.value)


def test_a_comma_separated_list_loads_the_shards_that_do_match(tmp_path):
    # One absent rank shard is normal, so a comma list only fails when nothing at all matched.
    good = tmp_path / "chains.rank0.jsonl"
    good.write_text(json.dumps({"prompt": [1], "chain": [2]}) + "\n", encoding="utf-8")
    pattern = f"{good},{tmp_path / 'chains.rank1.jsonl'}"
    assert len(load_streams(pattern, encoder=FakeEncoder())) == 1


def test_a_comma_separated_list_matching_nothing_raises(tmp_path):
    pattern = f"{tmp_path / 'a*.jsonl'},{tmp_path / 'b*.jsonl'}"
    with pytest.raises(FileNotFoundError):
        load_streams(pattern, encoder=FakeEncoder())


def test_matching_shards_still_load_and_respect_the_limit(tmp_path):
    p = tmp_path / "chains.rank0.jsonl"
    p.write_text("\n".join(json.dumps({"prompt": [i], "chain": [2]}) for i in range(5)) + "\n", encoding="utf-8")
    assert len(load_streams(str(tmp_path / "chains.rank*.jsonl"), encoder=FakeEncoder())) == 5
    assert len(load_streams(str(tmp_path / "chains.rank*.jsonl"), encoder=FakeEncoder(), limit=2)) == 2


def test_packing_tokens_shorter_than_one_row_raises():
    with pytest.raises(ValueError) as excinfo:
        pack([[1, 2, 3]], length=8, sep=0)
    assert "8-token row" in str(excinfo.value)


def test_a_row_that_fits_packs_to_one_row():
    packed = pack([[1, 2, 3, 4, 5]], length=4, sep=0)
    assert tuple(packed.shape) == (1, 4)
    assert packed.dtype == torch.long


def test_batches_over_a_real_pack_still_iterate():
    packed = pack([list(range(16))], length=8, sep=0)
    batches = Batches(packed, batch_size=1)
    assert len(batches) == 2
    assert sum(b.shape[0] for b in batches) == 2

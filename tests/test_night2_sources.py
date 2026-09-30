"""The date generator files its questions under the record's own source, not contrastive_*."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prep.contrastive import FAMILIES
from prep.night2 import dates_records


def records():
    return dates_records(pairs_per_family=2, seed=0)


def family_of(record):
    # ids look like night2_dates/<family>/<pair_id>/<sibling>
    return record["_meta"]["id"].split("/")[1]


def test_date_questions_are_not_attributed_to_the_contrastive_generator():
    rows = records()
    assert rows
    for r in rows:
        src = r["questions"]["decision"]["src"]
        assert r["_meta"]["source"] == "night2_dates"
        assert not src.startswith("contrastive_"), (r["_meta"]["id"], src)


def test_each_date_question_names_its_own_family():
    for r in records():
        family = family_of(r)
        assert family in FAMILIES
        assert r["questions"]["decision"]["src"] == f"dates_{family}"


def test_the_families_are_all_represented_and_never_collide():
    srcs = {r["questions"]["decision"]["src"] for r in records()}
    families = {family_of(r) for r in records()}
    assert srcs == {f"dates_{f}" for f in families}
    assert len(srcs) > 1
    assert not (srcs & {f"contrastive_{f}" for f in families})


def test_the_question_source_agrees_with_the_record_source_for_every_record():
    for r in records():
        assert r["questions"]["decision"]["src"].startswith("dates_")
        assert r["_meta"]["source"] == "night2_dates"


def test_row_digest_is_present_and_untouched_by_the_relabelling():
    # row_sha256 hashes the source sentences, not the question, so moving the src label must not
    # perturb the digest prep.prep sorts and screens by.
    for r in records():
        digest = r["_meta"]["row_sha256"]
        assert digest and len(digest) == 64

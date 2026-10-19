"""Physical content, not task IDs, defines independent evaluation instances."""
from dataclasses import replace

import pytest

from coimagine.random_start import layout


@pytest.mark.parametrize("map_name", ["L0", "L1"])
def test_full_panels_disjoint_and_train_unchanged(map_name):
    panels = {split: layout.build_split(map_name, split)[0]
              for split in ("train", "validation", "test")}
    assert layout.audit_split_content(panels)["ok"]
    legacy_train, _ = layout.build_split(map_name, "train",
                                        split_version=layout.LEGACY_SPLIT_VERSION)
    assert [x.as_dict() for x in panels["train"]] == [x.as_dict() for x in legacy_train]
    assert len({x.seed for p in panels.values() for x in p}) == 384
    again, metadata = layout.build_split(map_name, "validation")
    assert [x.as_dict() for x in again] == [x.as_dict() for x in panels["validation"]]
    assert metadata["split_version"] == layout.SPLIT_VERSION
    assert all("_validation_v2_" in x.name for x in again)


def test_duplicate_check_ignores_metadata_and_reproduces_legacy_leak():
    train = layout.build_split("L0", "train", count=3)[0]
    legacy = layout.build_split("L0", "validation", count=3,
                                split_version=layout.LEGACY_SPLIT_VERSION)[0]
    report = layout.audit_split_content({"train": train, "legacy": legacy})
    assert len(report["duplicates"]) == 3
    renamed = replace(train[0], name="different", seed=999, index=999, split="test")
    assert layout.content_hash(train[0]) == layout.content_hash(renamed)
    assert not layout.audit_split_content({"one": [train[0], renamed]})["ok"]
    changed = replace(renamed, goal_centre=(0.1, 0.2))
    assert layout.content_hash(changed) != layout.content_hash(train[0])


def test_custom_base_seed_still_separates_splits():
    panels = {s: layout.build_split("L0", s, count=4, base_seed=12)[0]
              for s in ("train", "validation", "test")}
    assert layout.audit_split_content(panels)["ok"]
    with pytest.raises(ValueError):
        layout.build_split("L0", "validation", split_version="typo")

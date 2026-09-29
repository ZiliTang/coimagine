"""Documentation guards.

Two things rot silently in a long project: the image links in the reports, and the claim that the
figures a reader is shown are the current ones. Both are checked here.
"""

from __future__ import annotations

import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
DOCS = sorted((REPO / "docs").glob("*.md"))

#: Figures that a reader is pointed at. If one of these disappears the reports show a broken image.
REQUIRED_FIGURES = (
    "outputs/grid/world_model/branches_showcase_t00.png",
    "outputs/grid/world_model/imagined_vs_real_t00.png",
    "outputs/grid/world_model/rollout_tree_showcase.png",
    "outputs/grid/world_model/episode_showcase.gif",
    "outputs/grid/world_model/belief_probe.png",
    "outputs/grid/world_model/prepare_probe.png",
)


@pytest.mark.parametrize("doc", DOCS, ids=[d.name for d in DOCS])
def test_image_links_resolve(doc: pathlib.Path) -> None:
    missing = []
    for match in re.finditer(r"!\[[^\]]*\]\(([^)]+)\)", doc.read_text()):
        target = (doc.parent / match.group(1)).resolve()
        if not target.exists():
            missing.append(match.group(1))
    assert not missing, f"{doc.name} links to missing images: {missing}"


@pytest.mark.parametrize("relative", REQUIRED_FIGURES)
def test_referenced_figures_exist(relative: str) -> None:
    assert (REPO / relative).exists(), relative


def test_the_final_report_shows_the_headline_figures() -> None:
    """The conclusions document must actually display the two figures for its two headline claims,
    not merely mention them."""

    text = (REPO / "docs" / "CoImagine_2D_最终结论_v1.md").read_text()
    for name in ("belief_probe.png", "prepare_probe.png", "branches_showcase_t00.png"):
        assert f"]({ '../outputs/grid/world_model/' + name })" in text, name


def test_superseded_figures_are_archived_not_referenced() -> None:
    """Anything under a `__old__` directory is explicitly not to be cited."""

    for doc in DOCS:
        assert "__old__" not in doc.read_text().split("## 8.1")[0], doc.name

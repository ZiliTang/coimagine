"""Tests for the limited message channel.

The message must be (a) computable by the sender from its own private state, (b) a
declared discrete codebook of at most B bits, and (c) actually informative -- i.e. it must
change the receiver's features, otherwise it is not a channel at all.
"""

from __future__ import annotations

import pytest

from coimagine.grid.state import HiddenPhysics
from coimagine.learning.messages import (
    MESSAGE_SCHEMES,
    SCHEME_ORDER,
    message_dim,
    message_symbol,
    message_vector,
    payload_bits,
)


def test_schemes_declare_bits_and_codebooks() -> None:
    for scheme, (bits, codebook) in MESSAGE_SCHEMES.items():
        # the payload must be able to distinguish every symbol in the codebook
        assert 2 ** bits >= len(codebook), scheme
        assert len(set(codebook)) == len(codebook), scheme
    assert payload_bits("none") == 0
    assert payload_bits("traction") == 1
    assert payload_bits("feedback") == 2
    assert SCHEME_ORDER == ("none", "traction", "feedback")


def test_message_is_computed_from_the_senders_private_state() -> None:
    grip = HiddenPhysics(1, "free", ("grip", "grip"))
    slip = HiddenPhysics(1, "free", ("grip", "slip"))
    assert message_symbol("traction", grip, ()) == "grip"
    assert message_symbol("traction", slip, ()) == "slip"
    assert message_symbol("none", grip, ()) == "silent"
    # the feedback scheme reports the sender's own last private observation
    assert message_symbol("feedback", grip, ("strained", "slipped")) == "slipped"


def test_message_vector_is_one_hot_of_the_declared_length() -> None:
    physics = HiddenPhysics(2, "constrained", ("slip", "grip"))
    for scheme in SCHEME_ORDER:
        vector = message_vector(scheme, physics, ("drove",))
        assert len(vector) == message_dim(scheme)
        assert sum(vector) == pytest.approx(1.0)
        assert set(vector) <= {0.0, 1.0}


def test_messages_change_the_receiver_features() -> None:
    """A channel that does not change the input is not a channel."""

    from coimagine.learning.data import generate_situations
    from coimagine.learning.features import sample_vector

    situation = generate_situations(n_scenes=3, lead_in=2, seed=0)[0]
    base = sample_vector(
        situation.scan, situation.scene.robot_pose(0),
        situation.history_a, situation.public_history, "push:0:E",
    )
    silent = base + message_vector("none", situation.physics, situation.history_b)
    spoken = base + message_vector("feedback", situation.physics, situation.history_b)
    assert silent != spoken


@pytest.mark.slow
def test_joint_message_comparison_runs() -> None:
    from coimagine.diagnostics.threshold_probe import (
        format_joint_comparison_markdown,
        run_joint_message_comparison,
    )

    summaries = run_joint_message_comparison(
        n_train_situations=20, n_test_situations=12, epochs=20, seed=0
    )
    assert len(summaries) == len(SCHEME_ORDER)
    assert [s.bits for s in summaries] == [0, 1, 2]
    assert "mean regret" in format_joint_comparison_markdown(summaries)

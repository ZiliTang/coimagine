"""Replay geometry and CAP memory at EVERY actual option boundary."""
from contextlib import contextmanager
from .library_features import LibraryFeatures


class HandoffLibraryFeatures(LibraryFeatures):
    @contextmanager
    def decision(self,history):
        with super().decision(history):
            # Normal decisions previously did this in panel(); forced-prefix
            # replay has no scoring panel. advance() is idempotent per boundary.
            for member in self.members:member.geometry.advance(history)
            yield

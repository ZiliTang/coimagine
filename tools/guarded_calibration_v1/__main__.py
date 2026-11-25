"""Reproduce the frozen v1 loop with its original collector, in its own CLI process."""
from tools import cap_guarded_calibration_loop as loop
from . import cap_guarded_calibration_forks as frozen

if __name__ == "__main__":
    loop.forks = frozen
    loop.ROOT = frozen.OUT
    loop.run()

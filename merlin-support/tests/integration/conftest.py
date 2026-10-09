"""Historical integration checks require explicit fixtures and execution opt-in."""

import os


def pytest_ignore_collect(collection_path, config):
    # Some historical tests execute Spike/RTL when tools are installed. Merely
    # installing a tool must not turn a lightweight relocation check into a run.
    return os.environ.get("GEMMINI_RUN_INTEGRATION") != "1"

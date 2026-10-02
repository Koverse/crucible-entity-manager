"""``python -m crucible_entity_manager``: the same as ``src/run_entity_manager.py``."""

from crucible_entity_manager.cli import main
from crucible_entity_manager.runtime.process import exit_process

if __name__ == "__main__":
    exit_process(main())

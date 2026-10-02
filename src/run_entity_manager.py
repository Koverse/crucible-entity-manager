"""Executor entrypoint: ``python3 /work/repo/src/run_entity_manager.py <component> ...``.

Running this file puts ``src/`` on ``sys.path``, so the package imports without
being installed.
"""

from crucible_entity_manager.cli import main
from crucible_entity_manager.runtime.process import exit_process

if __name__ == "__main__":
    exit_process(main())

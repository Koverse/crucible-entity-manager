# crucible-entity-manager

The Crucible entity correlation pipeline (transformer, tracker, track fuser and
duplicate identifier) rebuilt to run on the Executor platform.

| Document | Read it for |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | How the pipeline and a pod work, in brief. |
| [`docs/runbook.md`](docs/runbook.md) | Running, observing and recovering it on Executor. |
| [`deploy/executions/`](deploy/executions/README.md) | The Execution definitions and how to start partitions. |
| [`docs/DESIGN.md`](docs/DESIGN.md) | The full design, the decisions behind it, and what changed from crucible-streamlit. |

## Running

Each Execution runs one component as one partition:

```sh
python3 src/run_entity_manager.py transformer LIVE_POV --partition 0 --partitions 2
python3 src/run_entity_manager.py transformer LIVE_POV --partition 1 --partitions 2
python3 src/run_entity_manager.py tracker LIVE_POV
python3 src/run_entity_manager.py fuser LIVE_POV
python3 src/run_entity_manager.py duplicates LIVE_POV
```

`python3 src/run_entity_manager.py <component> --help` lists the options.

## Development setup

You need Python 3.12, a pyenv virtualenv, and the cruciblelib wheel. cruciblelib
isn't on PyPI; it ships inside the `cruciblelib-*.tar.gz` asset of a
[crucible-analytics release](https://github.com/Koverse/crucible-analytics/releases).

```sh
pyenv virtualenv 3.12.9 entity-manager
pyenv local entity-manager

mkdir -p wheels
gh release download v3.0.10 -R Koverse/crucible-analytics -p 'cruciblelib-*.tar.gz' -D wheels
tar -xzf wheels/cruciblelib-*.tar.gz -C wheels

uv pip sync requirements/dev.lock --find-links wheels
uv pip install --no-deps -e .
```

Run the checks that CI runs:

```sh
ruff check . && ruff format --check . && ty check && pytest --cov
```

## Dependencies

Dependencies are declared in `pyproject.toml`, and the resolved versions are
pinned in `requirements/`:

| File | Purpose |
|---|---|
| `requirements/dev.lock` | The development and CI environment, with hashes. |
| `requirements/runtime.txt` | What the Executor profile image must provide at runtime, with hashes. Hand this file to the image build together with the cruciblelib wheel from the release named above. The wheel's hash is pinned in the file, so `pip install --require-hashes --find-links <wheel-dir> -r requirements/runtime.txt` verifies it. |

To regenerate both after changing `pyproject.toml`:

```sh
uv pip compile pyproject.toml --extra dev --python-version 3.12 --find-links wheels \
    --no-emit-find-links --generate-hashes -o requirements/dev.lock
uv pip compile pyproject.toml --python-version 3.12 --find-links wheels \
    --no-emit-find-links --generate-hashes -o requirements/runtime.txt
```

## Parity fixtures

`tests/fixtures/parity` holds outputs of the baseline (crucible-streamlit
`1b534df`) that the parity tests compare against. Regenerate them, after a
deliberate change to the generator, with a local clone of crucible-streamlit:

```sh
python tests/parity/generate.py --streamlit-repo ~/path/to/crucible-streamlit
```

The output is deterministic, so unchanged fixtures regenerate byte for byte.

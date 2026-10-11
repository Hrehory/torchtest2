# Contributing

Contributions (issues and pull requests) are very welcome.

## Setting up

1. Clone the repository (or your fork of it) and enter it.

2. Create a virtual environment and install the package in editable mode with the development
   tools:

   ```bash
   python -m venv .venv
   source .venv/bin/activate          # .venv\Scripts\activate on Windows
   pip install -e ".[dev]"            # add ",integrations" to also test the built-in plugins
   ```

   On a machine without a GPU you can install the much smaller CPU build of PyTorch first:
   `pip install torch --index-url https://download.pytorch.org/whl/cpu`.

3. Install the pre-commit hooks. They lint and format the code with
   [ruff](https://docs.astral.sh/ruff/) on every commit:

   ```bash
   pre-commit install
   ```

## Making a change

Add tests for any new functionality. In particular:

- **A new check** needs a function `_check_<name>` in `torchtest2/checks.py`, registered in
  `CHECKS` and `_CHECK_FUNCTIONS`; at least one test with a model that has the bug (`FAIL`/`WARN`)
  and one without it (`PASS`) in `tests/test_check_model.py`; a row in the "What is checked"
  table in `README.md` (same order as `CHECKS`); and ideally a bug-gallery example.
- **A new built-in plugin** goes into `torchtest2/plugins/` as a `Plugin` subclass with a `name`,
  is added to `BUILTIN_PLUGINS` in `torchtest2/plugins/__init__.py`, must not import its
  framework at module level (torchtest2 depends only on PyTorch), and needs a test in
  `tests/test_plugins.py` that uses `pytest.importorskip` for the framework, plus a row and an
  example in the "Built-in plugins" section of the README. Add the framework to the
  `integrations` extra in `pyproject.toml`.
- **README examples are tests.** `tests/test_readme.py` runs every Python block of `README.md` in
  order and checks that every report line quoted in a following `text` block is really
  produced. If you change a message, update the README. Mark purely illustrative blocks (e.g.
  with placeholder imports) with `<!-- readme-test: skip -->` on the line before the block.

Before opening a pull request:

```bash
ruff check . && ruff format --check .
mypy
pytest
```

Describe what changed and why in the commit message or the pull request.

## Releasing

Releases are published to [PyPI](https://pypi.org/p/torchtest2) by the `Publish` workflow
(`.github/workflows/publish.yml`) through trusted publishing, so no API token is needed.

1. Bump `__version__` in `torchtest2/__init__.py` and merge the change into `master`.
2. Optionally, run the `Publish` workflow by hand (Actions → Publish → Run workflow): it uploads
   the build to [TestPyPI](https://test.pypi.org/p/torchtest2), where you can check the page and
   `pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ torchtest2`.
   TestPyPI refuses a version it already has, so this works once per version.
3. Create a GitHub release with the tag `vX.Y.Z` (matching `__version__`) and the release notes.
   Publishing it builds the package, checks it with `twine check --strict` and uploads it to PyPI.

One-time setup: on pypi.org and test.pypi.org, add a (pending) trusted publisher with owner
`Hrehory`, repository `torchtest2`, workflow `publish.yml` and environment `pypi` (`testpypi` on
TestPyPI).

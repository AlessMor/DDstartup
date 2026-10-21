<!-- ai-instructions: managed; source-sha256=01f8856f681842544f9c4f383fb66db80ec52f0e3d60f25c7607c02134bfe7cc; scope=project; source=projects/ddstartup/instructions.md -->
<!-- Edit the canonical source in ~/.agents; run ai-instructions sync. -->

# DDstartup project instructions

Shared by the `DDstartup` and `dd_startup` checkouts of the same remote; keep
it branch-neutral.

## Environment and gate

- Use the `ddstartupenv` conda environment and run `python -m pytest` from the
  checkout root. `pytest.ini` enables coverage by default; add `--no-cov` for
  quick iterations.
- Branches differ in layout (`src/` versus a top-level `ddstartup/` package).
  `tests/conftest.py` puts the checkout root first on `sys.path`, so tests use
  the local tree. The editable `ddstartup` package in `ddstartupenv` points at
  the `dd_startup` checkout, so code run outside the tests may import that
  tree instead.

## Scientific scope

This file makes no claim about physics or numerical results. Validate such
claims with this project's tests, documented methods, and scientific review;
do not import conclusions from `multispecies_fuelcycle`.

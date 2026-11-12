# Contributing to MVIS

Thanks for helping. Bug reports, dataset reports and pull requests are welcome.

## Reporting a problem
Open an issue with: the config YAML, the full log, `mvis --version`, your OS and
Python version, and (if you can share it) a link to the data.

Security problems: see [SECURITY.md](https://github.com/openmvis/openmvis/blob/main/SECURITY.md)
and report them privately to contact@openmvis.com, not in a public issue.

## Development setup
```bash
python -m venv .venv && source .venv/bin/activate     # Python 3.11 or 3.12, Linux or macOS
pip install -e ".[apriltag,aruco,ros1,dev]"
scripts/test.sh unit
```
`scripts/test.sh` runs the tests by tier (`unit`, `solver`, `integration`,
`full`); see `doc/testing.md`. Tests that need real recordings skip when the
data is missing; point `MVIS_DATASETS` at your dataset folder (default
`~/Datasets`) to run them.

## Pull requests
- One topic per PR, with a test that fails before and passes after.
- Keep `scripts/test.sh unit` and `scripts/test.sh solver` green (what CI runs).
- New source files start with the same licence header as the files in `src/`: the
  OpenMVIS GPL-3.0-only notice block, then the `SPDX-FileCopyrightText` and
  `SPDX-License-Identifier: GPL-3.0-only` lines (copy the top of any `.py` file). YAML
  configs carry the two SPDX lines only. `tests/test_spdx_headers.py` checks it.

## Releases
A `v*` tag triggers `release.yml`, which builds the wheel and publishes it to
PyPI through the protected `pypi` environment (required reviewer: the owner).
Tags are protected by a ruleset, so only the maintainers can create them.

## Contributor licence agreement
By submitting a pull request, you confirm that you wrote the contribution (or
have the right to submit it), and you grant Yulin Yang a perpetual, worldwide,
royalty-free, irrevocable licence to use, modify, sublicense, and relicense
your contribution, including under commercial licences, in addition to
GPL-3.0. You keep the copyright to your contribution.

This lets MVIS stay dual-licensed (GPL-3.0 plus a commercial licence; see
[COMMERCIAL-LICENSE.md](https://github.com/openmvis/openmvis/blob/main/COMMERCIAL-LICENSE.md)).

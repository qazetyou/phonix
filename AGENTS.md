# Project Instructions

These instructions apply to this standalone project. If a parent workspace
provides additional instructions, follow those too; this file does not require
any files outside the source archive.

Requirements are recorded in `README.md` and model assumptions in `docs/model.md`.

## Shared working defaults

- Follow the user's current request. Keep changes inside the requested project
  and preserve unrelated work.
- Explain unfamiliar commands and significant design choices briefly in plain
  language. Identify commands the user must run and their working directory.
- Act directly on simple tasks. For complex work, inspect the project, propose
  a concrete brief plan, and ask for confirmation before implementing. Carry an
  approved plan through verification; make routine reversible choices yourself.
- Ask about major decisions when intent is unclear. Prefer the simplest solution
  and check project files or authoritative documentation before consequential
  assumptions. Reproduce reported bugs when practical.
- Keep dependencies minimal and follow this project's layout. Document verified
  prerequisites, installation, build and test commands in the README.
- Use descriptive filenames, no trailing whitespace, and final newlines.
- Run appropriate checks; add meaningful tests for executable behavior and error
  cases. Documentation-only edits need diff and whitespace review, not tests.
  Finish with a concise report of changes, checks and unresolved limitations.
- Inspect Git status before editing and review the final diff. Keep changes
  unstaged and uncommitted unless requested; preserve already staged work. Push
  only when requested. Review staged files before a focused imperative commit.
- Keep credentials and local `.env` files out of Git. Use placeholder values in
  shared configuration and ignore generated output. Do not expose secrets.

## Project conventions

Runtime code is in `phonix/`, immutable reference assets in `phonix/data/`, silent
tests in `tests/`, development helpers in `tools/`, and documentation in `docs/`.
Install from source with `python3 -m venv .venv` and
`.venv/bin/python -m pip install -e .`. Build deliveries with
`.venv/bin/python tools/build_release.py` using setuptools 77+. Release outputs
belong in ignored `dist/`; personal profiles and exports must never be packaged.

Keep dependencies minimal and document verified setup, run, and test commands in `README.md` once implemented.

Use Python 3.11+ with NumPy as the only Python runtime dependency and PipeWire's `pw-play` for playback. Run `python3 -m unittest discover -s tests -v` from this project directory. Keep tests silent; do not play live audio as part of automated verification.

Never describe listening consistency or numerical filter-fitting error as measured acoustic accuracy against diffuse field. Keep the population baseline fixed and versioned across profiles. Preserve incomplete-session data, and never export prior-only estimates for frequencies that have not been connected by actual comparisons.

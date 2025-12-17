# Release runbook (repo-local)

Canonical changelog: `CHANGELOG.md` (Keep a Changelog + SemVer).

## What “release-ready” means here

- `make test` passes
- `make smoke` passes (hard requirements)
- hard requirements verified (see `docs/CURSOR_SMOKE_TEST.md`)
- `README.md` + `CHANGELOG.md` updated and consistent

## Tag-based release SOP (short)

1. Make sure `main` is clean and up to date:

```bash
git checkout main
git pull --ff-only
make test
make smoke
```

2. Update version (SemVer) in `pyproject.toml`.

3. Update `CHANGELOG.md` under a new version heading.

4. Commit:

```bash
git commit -am "chore(release): vX.Y.Z"
```

5. Create tag:

```bash
git tag -a vX.Y.Z -m "vX.Y.Z"
git push origin main --tags
```

6. GitHub Release:
- Create a release for tag `vX.Y.Z`
- Paste the `CHANGELOG.md` section for `X.Y.Z`

## Pre-push automation (recommended)

Install pre-commit hooks so you catch CI failures before pushing:

```bash
pre-commit install
pre-commit install --hook-type pre-push
```

What it runs on `git push`:
- `make test`
- `make smoke`

## Canonical SOP (recommended)

If you use the multi-layer memory system, the canonical release SOP lives in Pepper:
- `~/memo/global-memories/release-sop-tag-based.md`



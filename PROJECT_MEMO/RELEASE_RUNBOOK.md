# Release runbook (repo-local)

Canonical changelog: `CHANGELOG.md` (Keep a Changelog + SemVer).

## What “release-ready” means here

- `make test` passes
- `make smoke` passes (hard requirements)
- hard requirements verified (see `docs/CURSOR_SMOKE_TEST.md`)
- `README.md` + `CHANGELOG.md` updated and consistent
- PRs/branches reviewed and cleaned (see "PR + branch hygiene" below)

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

## PR + branch hygiene (required before release)

1. Review open PRs:
   - Merge anything that is release-related and ready.
   - Close PRs that are outdated or tied to previous releases.
   - Keep PRs that are clearly a new feature, POC, or unrelated track.

2. Review branches:
   - Delete merged or stale branches that belong to prior releases.
   - Keep branches that represent active or unrelated work (POC, new feature track).

3. Keep a short audit trail:
   - Note any kept branches/PRs with a short reason in the release notes or tracker.

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



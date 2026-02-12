---
name: memo-kb-sync
description: Performs bi-directional sync for memo and global-kb repos and updates release artifacts. Use when the user requests memory sync, after releases, or before planning next release.
---
# Memo and KB Sync

## When to Use
- User requests memory sync
- After releases or major updates
- Before planning next release

## Instructions
1. Pull latest from both repos:
   - `~/memo` (Pepper Memory Bank)
   - `~/Code/global-kb` (permanent archive)
2. Apply updates:
   - Pepper memory updates in `~/memo/global-memories/`
   - Release notes and architecture docs in `~/Code/global-kb/`
3. Commit with conventional messages:
   - `sync:` or `update:` or `archive:`
4. Push both repos.

## Output Format
- Sync status for memo and global-kb
- Files updated
- Commit hashes (if available)

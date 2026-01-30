---
name: memo-kb-sync
description: Perform bi-directional sync for memo and global-kb repos and update release artifacts. Use when the user requests memory sync or after releases.
---
# Memo and KB Sync

## When to Use
- User requests memory sync
- After releases or major updates
- Before planning next release

## Instructions
1. Pull latest from both repos:
   - `~/memo`
   - `~/Code/zendesk/global-kb`
2. Apply updates:
   - Pepper memory updates in `~/memo/global-memories/`
   - Release notes in `~/Code/zendesk/global-kb/architecture/`
3. Commit with conventional messages:
   - `sync:` or `update:` or `archive:`
4. Push both repos.

## Output Format
- Sync status for memo and global-kb
- Files updated
- Commit hashes (if available)

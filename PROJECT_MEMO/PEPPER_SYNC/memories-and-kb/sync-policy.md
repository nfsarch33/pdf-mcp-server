# Memory Sync Policy v9.0

## Three-Layer Architecture

### Layer 1: .cursor/rules (Auto-Injected)

- Memory strategy and self-regulation
- Development standards
- Status: Always in context

### Layer 2: Pepper Memory Bank (MCP-Accessed)

- ~/memo/global-memories/
- Detailed procedures and patterns
- Status: On-demand via MCP

### Layer 3: global-kb (Archived)

- ~/Code/global-kb/
- Completed investigations and decisions
- Status: Permanent record with git history

## Memory Actions

| Trigger | Action |
| --- | --- |
| User contradicts | DELETE Cursor memory |
| New pattern | CREATE Cursor + UPDATE Pepper |
| Procedure | UPDATE Pepper only |
| Investigation | WRITE to global-kb/ |

## GitHub Backup

**Repositories**:

- Pepper: `git@github.com:nfsarch33/cursor-memory-bank.git`
- KB: `git@github.com:nfsarch33/cursor-global-kb.git`

**Sync Commands**:

```bash
# Push to GitHub
cd ~/memo && git add -A && git commit -m "sync: update" && git push origin main
cd ~/Code/global-kb && git add -A && git commit -m "sync: update" && git push origin main

# Pull from GitHub
cd ~/memo && git pull origin main
cd ~/Code/global-kb && git pull origin main
```

## Cross-Machine Sync

When switching machines:

1. Configure SSH key
2. Clone from GitHub (fast)
3. Configure MCP
4. Verify with "health check"

# Memory Sync Policy v12.1

## Canonical Paths (Both Machines)

| Repo | Path | Remote |
|------|------|--------|
| Pepper (memo) | ~/memo | github.com:nfsarch33/cursor-memory-bank |
| global-kb | ~/Code/global-kb | github.com:nfsarch33/cursor-global-kb |

No Zendesk-specific nesting. Both repos serve all projects (personal and work).

## Manual sync commands

```bash
# push
cd ~/memo && git add -A && git commit -m "sync: update" && git push origin main
cd ~/Code/global-kb && git add -A && git commit -m "sync: update" && git push origin main

# pull
cd ~/memo && git pull --ff-only origin main
cd ~/Code/global-kb && git pull --ff-only origin main
```

## Automated sync

`~/memo/tools/daily_refresh.sh` handles pull/commit/push for both repos.
Also syncs workspace rules: global-kb/cursor-config/rules/ -> known workspace targets.

## global-kb-path.txt

Machine-specific. In .gitignore (not tracked in git).
Each machine sets once after cloning:

```bash
echo "$HOME/Code/global-kb" > ~/memo/tools/global-kb-path.txt
```

The `detect_global_kb()` function in daily_refresh.sh has a fallback search order if the file is missing.

## Cross-machine setup

See Agent Skill SKILL.md "Cross-Machine Bootstrap" section.

## WSL scheduling

Use Windows Task Scheduler. See `~/memo/tools/install-wsl-task-scheduler.ps1`.

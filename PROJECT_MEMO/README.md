# Project Memo (Portable)

This folder exists to make it easy to **pick up this repo on a new computer** with minimal context loss.

## Directory Structure

```
PROJECT_MEMO/
├── README.md                              # This file
├── GLOBAL_CURSOR_INSTRUCTIONS.md          # Cursor AI system prompt
├── MEMORY_AND_RULES.md                    # Development rules and memory system
├── PROJECT_STATUS_PROMPT.md               # Current project status for AI
├── MIGRATION_RUNBOOK.md                   # How to migrate to new environments
├── RELEASE_RUNBOOK.md                     # Release process documentation
├── TROUBLESHOOTING.md                     # Common issues and solutions
├──
├── ROADMAPS/                              # Release roadmaps & feature planning
│  └── v0.7.0-v1.0.0-roadmap.md            # Current 4-release roadmap
│  └── [future roadmaps...]              # Add new release roadmaps here
├──
├── STRATEGY/                              # Business & strategic planning
│  └── next-steps-sprint-plan.md           # 4-week sprint plan + business context
│  └── [future strategy docs...]           # Commercial models, GTM, etc.
├──
├── OPERATIONS/                            # Operational/execution documents
│  └── github-milestone-v0.7.0-template.md # GitHub milestone + issue templates
│  └── [future operations docs...]          # Processes, checklists, templates
├──
├── PEPPER_SYNC/                           # Pepper sync configuration
│  └── [sync rules and policies]
└── cursor-ai-memory-system-complete-setup.md  # Cursor memory setup guide
```

## Content Categories

### Core Runbooks (Read First)
- **MIGRATION_RUNBOOK.md** - How to set up on new machines
- **RELEASE_RUNBOOK.md** - Release process and versioning
- **TROUBLESHOOTING.md** - Common issues and solutions

### AI System Documents
- **GLOBAL_CURSOR_INSTRUCTIONS.md** - System prompt for Cursor AI
- **MEMORY_AND_RULES.md** - Development rules and memory integration
- **PROJECT_STATUS_PROMPT.md** - Current project status snapshot

### ROADMAPS/ - Release Planning
Store all release roadmaps, feature planning, and product vision here:
- `v0.7.0-v1.0.0-roadmap.md` - Comprehensive 4-release roadmap
- Future: `v1.1-v1.5-roadmap.md`, feature planning docs, etc.

**Purpose:** Track product evolution, feature prioritization, and architectural decisions

### STRATEGY/ - Business & Strategic Planning
Store business strategy, market positioning, and high-level planning:
- `next-steps-sprint-plan.md` - 4-week action plan with deliverables
- Future: Commercial licensing strategies, GTM plans, partnership approaches, etc.

**Purpose:** Align technical roadmap with business goals and market opportunities

### OPERATIONS/ - Execution & Process Documents
Store templates, checklists, and operational processes:
- `github-milestone-v0.7.0-template.md` - Ready-to-use GitHub issues/milestone
- Future: Release checklists, deployment procedures, CI/CD configurations, etc.

**Purpose:** Standardize processes and enable fast execution

---

## What Lives Here

- ✅ Portable, repo-scoped notes only (no machine paths required, no secrets).
- ✅ "How to run / test / release / troubleshoot" for PDF MCP Server.
- ✅ Pointers to the **canonical** multi-layer memory system (Pepper/global-kb) when available.
- ✅ Canonical copies of onboarding docs (GLOBAL_CURSOR_INSTRUCTIONS, etc.).
- ✅ Strategic roadmaps, sprint plans, and business context.
- ✅ Operational templates and execution guides.

## What Must NOT Live Here

- ❌ Secrets/tokens/passwords (Cursor `mcp.json` credentials, API keys, etc.).
- ❌ Personal/academic notes (keep those untracked and/or in private repos).
- ❌ Code (this is docs-only; code lives in `/pdf_tools`, `/tests`, etc.).
- ❌ Machine-specific paths or configurations.

---

## Quick Start (Read in Order)

1. **New environment?** Start with `MIGRATION_RUNBOOK.md`
2. **Ready to release?** Read `RELEASE_RUNBOOK.md`
3. **Something broken?** Check `TROUBLESHOOTING.md`
4. **AI context?** Review `PROJECT_STATUS_PROMPT.md`
5. **Need roadmap?** See `ROADMAPS/v0.7.0-v1.0.0-roadmap.md`
6. **Sprint planning?** See `STRATEGY/next-steps-sprint-plan.md`
7. **Creating GitHub issues?** See `OPERATIONS/github-milestone-v0.7.0-template.md`

---

## Canonical Onboarding Docs

These are maintained here and referenced throughout the repo:
- `GLOBAL_CURSOR_INSTRUCTIONS.md` - Cursor AI system prompt
- `MEMORY_AND_RULES.md` - Development memory and rules
- `PROJECT_STATUS_PROMPT.md` - Current project snapshot

Repo root keeps short "moved to ..." stubs for backward compatibility.

---

## Adding New Documents

### For Release Roadmaps:
1. Create file in `ROADMAPS/` (e.g., `ROADMAPS/v1.1-v2.0-roadmap.md`)
2. Include: phase breakdown, tool inventory, release plans, success criteria
3. Link from `PROJECT_MEMO/README.md`

### For Strategic Plans:
1. Create file in `STRATEGY/` (e.g., `STRATEGY/commercial-licensing-model.md`)
2. Include: context, options, recommendations, decision factors
3. Link from `PROJECT_MEMO/README.md`

### For Operational Docs:
1. Create file in `OPERATIONS/` (e.g., `OPERATIONS/deployment-checklist.md`)
2. Include: steps, requirements, validation criteria
3. Link from `PROJECT_MEMO/README.md`

---

## Last Updated

**2026-01-28**: Added ROADMAPS/, STRATEGY/, OPERATIONS/ directories with comprehensive planning documents.

**Document Version:** 2.0

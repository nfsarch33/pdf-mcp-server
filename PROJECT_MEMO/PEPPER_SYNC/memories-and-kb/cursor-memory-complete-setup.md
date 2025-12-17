# Cursor AI: Complete Self-Regulating Memory System Setup

**Version**: 9.0  
**Date**: 2025-12-16  
**Purpose**: One-file complete setup for new Cursor IDE instances  
**Principles**: KISS, DRY, SOLID, CLEAN CODE  
**Status**: Production-Ready  

---

## TL;DR: Quick Start (5 minutes)

```bash
# 1. Run setup script (see Section 8)
bash ~/setup-cursor-memory.sh

# 2. Restart Cursor

# 3. Say "health check" in any conversation

# Done. No daily prompts needed.
```

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Core Concepts](#2-core-concepts)
3. [Prerequisites](#3-prerequisites)
4. [Directory Structure](#4-directory-structure)
5. [MCP Configuration](#5-mcp-configuration)
6. [Memory Bank Setup](#6-memory-bank-setup)
7. [Workspace Rules](#7-workspace-rules)
8. [Automated Setup Script](#8-automated-setup-script)
9. [Verification](#9-verification)
10. [Maintenance](#10-maintenance)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. System Architecture

### Three-Layer Memory Model

```
┌────────────────────────────────────────────────┐
│ LAYER 1: Auto-Injected (Every Conversation)   │
├────────────────────────────────────────────────┤
│ .cursor/rules (83 lines)                       │
│ ├─ Memory strategy                             │
│ ├─ Self-regulation logic                       │
│ └─ Dev standards                               │
└────────────────────────────────────────────────┘
            ↓
┌────────────────────────────────────────────────┐
│ LAYER 2: On-Demand (MCP Tools)                 │
├────────────────────────────────────────────────┤
│ Pepper Memory Bank (~/memo/global-memories)   │
│ ├─ Procedures (18 files max)                   │
│ ├─ Guides                                      │
│ └─ Project-specific context                    │
│                                                │
│ Memory MCP                                      │
│ └─ Knowledge graph (entities + relations)      │
└────────────────────────────────────────────────┘
            ↓
┌────────────────────────────────────────────────┐
│ LAYER 3: Archive (Git-Controlled)              │
├────────────────────────────────────────────────┤
│ global-kb/ (~/Code/global-kb)                  │
│ ├─ Investigations                              │
│ ├─ Archived procedures                         │
│ └─ Historical context                          │
└────────────────────────────────────────────────┘
```

### Data Flow

```
New Knowledge
    ↓
Decide: Needed every conversation?
    ├─ YES → Store in .cursor/rules
    └─ NO → Store in Pepper Memory Bank
            ↓
            Fits 1 paragraph?
            ├─ NO → Pepper Memory Bank
            └─ YES (+ daily use) → .cursor/rules
                    ↓
                    Archive old version to global-kb/
```

---

## 2. Core Concepts

### Self-Regulation (Built-In, Zero Daily Prompt)

| Trigger | Action | Where |
|---------|--------|-------|
| User contradicts memory | DELETE from Cursor memories | Auto |
| New pattern detected | CREATE in rules + UPDATE Pepper | Auto |
| Procedure discovered | UPDATE Pepper only | On-demand |
| Investigation needed | WRITE to global-kb/ | Manual |

### Quality Gates

Before creating any memory, Cursor asks:

1. **Frequency**: Needed every conversation? (No → Pepper)
2. **Size**: Fits 1 paragraph? (No → Pepper)
3. **Type**: Rule/pattern, not task? (Task → don't memorize)

### Reliability Hierarchy

- **Local Rules** (.cursor/rules) — Always available
- **Pepper Memory Bank** — Fallback if MCP fails
- **Global KB** — Archive only (source control)

---

## 3. Prerequisites

### Required

- **Cursor IDE** (latest)
- **Node.js** v18+ (for MCP servers)
- **Git** (for version control)
- **~5 GB disk space** (for all three layers)

### Optional

- **Docker** (for advanced MCP servers)
- **Perplexity API key** (for research tools)

### Verify

```bash
node --version      # Should be 18+
npm --version
git --version
curl -I https://api.perplexity.com  # If using Perplexity
```

---

## 4. Directory Structure

### Final Layout

```
$HOME/
├── .cursor/
│   └── mcp.json                          ← MCP config
│
├── memo/                                 ← Pepper Memory Bank (git repo)
│   ├── .git/
│   └── global-memories/
│       ├── sync-policy.md               ← System documentation
│       ├── error-patterns.md            ← Error solutions
│       ├── coding-standards.md          ← Code patterns
│       └── ... (up to 18 files)
│
└── Code/
    ├── global-kb/                        ← Knowledge Base (git repo)
    │   ├── .git/
    │   ├── architecture/
    │   │   └── cursor-memory-system-v9.md
    │   ├── archive/
    │   │   └── procedure-name-timestamp/
    │   └── cursor-config/
    │       └── rules/
    │           └── <workspace>.rules   ← Backup
    │
    └── <workspace>/                      ← Your project
        ├── .cursor/
        │   └── rules                    ← Workspace rules (auto-injected)
        └── ... (your code)
```

---

## 5. MCP Configuration

### Create ~/.cursor/mcp.json

```json
{
  "mcpServers": {
    "allPepper-memory-bank": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@allpepper/memory-bank-mcp@latest"],
      "env": {
        "MEMORY_BANK_ROOT": "$HOME/memo"
      }
    },
    "memory": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"]
    },
    "context7": {
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp"]
    },
    "sequential-thinking": {
      "command": "npx",
      "args": ["-y", "sequential-thinking-mcp"]
    },
    "fetch": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-fetch"]
    }
  }
}
```

### MCP Servers Explained

| Server | Purpose | Required |
|--------|---------|----------|
| **allPepper-memory-bank** | Procedure storage | ✅ Yes |
| **memory** | Knowledge graph | ✅ Yes |
| context7 | Library documentation | Recommended |
| sequential-thinking | Complex planning | Recommended |
| fetch | Web page fetching | Optional |

### Optional: Add Perplexity (Research)

```json
{
  "perplexity-ask": {
    "command": "npx",
    "args": ["-y", "server-perplexity-ask"],
    "env": {
      "PERPLEXITY_API_KEY": "pplx-xxxxx"
    }
  }
}
```

**Note**: Replace `$HOME` with actual path if needed. Cursor reads this on startup.

---

## 6. Memory Bank Setup

### Initialize Pepper

```bash
mkdir -p ~/memo/global-memories
cd ~/memo
git init
git config user.name "Your Name"
git config user.email "your@email.com"
```

### Create Core Files

#### File 1: sync-policy.md

```markdown
# Memory Sync Policy v9.0 (Self-Regulating)

## How It Works

| Layer | Auto-Injected | Purpose |
|-------|---|---------|
| `.cursor/rules` | Yes | Patterns (1 para max) |
| Cursor memories | Yes | Quick facts |
| Pepper Memory Bank | On demand | Procedures |
| global-kb/ | Manual | Archive only |

## Memory Actions (Rules Auto-Execute)

| Trigger | Action |
|---------|--------|
| User contradicts | DELETE memory |
| New pattern | CREATE rule + UPDATE Pepper |
| Procedure discovered | UPDATE Pepper only |
| Investigation | WRITE to global-kb/ |

## Quality Gates

1. Needed every conversation? → Yes = Rules, No = Pepper
2. Fits 1 paragraph? → Yes = Rules, No = Pepper
3. Rule not task? → Yes = Memorize, No = Don't

## Health Check

Say "health check" to verify:
- read_file(".cursor/rules")
- memory_bank_read("global-memories", "sync-policy.md")

## Backup Locations

- Pepper: ~/memo (git repo)
- Rules: Code/global-kb/cursor-config/rules/
- Archive: Code/global-kb/archive/

## Auto-Recovery on Error

| Failure | Fallback |
|---------|----------|
| Memory MCP unavailable | Use Pepper Memory Bank |
| Pepper fails | Use .cursor/rules only |
| All systems | Alert user |
```

#### File 2: error-patterns.md

```markdown
# Error Patterns & Solutions

## Common Issues

### MongoDB Connection Timeout
**Pattern**: operation=mongo_connect timeout
**Solution**: 
1. Check MONGO_URL environment variable
2. Verify secret-sidecar running
3. Test AWS IAM roles

### OpenSearch Cluster Yellow
**Pattern**: cluster health = yellow
**Solution**:
1. Check DataDog dashboard
2. Run: POST /_cluster/reroute?retry_failed=true
3. Verify disk space > 15%

### MCP Server Not Available
**Pattern**: mcp_error: server_not_found
**Solution**:
1. Check ~/.cursor/mcp.json
2. Verify Node.js installed
3. Restart Cursor

## Prevention Checklist

- [ ] Run "health check" daily
- [ ] Commit Pepper changes: git -C ~/memo commit -am "update"
- [ ] Review rules monthly
- [ ] Archive old procedures quarterly
```

#### File 3: coding-standards.md

```markdown
# Coding Standards & Patterns

## Language-Specific

### Go
- Use MustWithRetry() for production operations
- Implement retry with exponential backoff
- Test error paths
- Log at INFO level for operations

### Python
- Use type hints
- Implement comprehensive error handling
- Add docstrings to all functions
- Test async patterns

### JavaScript/TypeScript
- Use ES6+ syntax
- Implement proper error boundaries
- Add JSDoc comments
- Test edge cases

## General Rules

- DRY: Extract common patterns into functions
- SOLID: Single responsibility per function
- CLEAN CODE: Meaningful names, small functions
- Error handling: Fail fast, log clearly
```

#### File 4: Init and Commit

```bash
cd ~/memo

# Create all files (copy content above)
cat > global-memories/sync-policy.md << 'EOF'
[paste sync-policy.md content]
EOF

cat > global-memories/error-patterns.md << 'EOF'
[paste error-patterns.md content]
EOF

cat > global-memories/coding-standards.md << 'EOF'
[paste coding-standards.md content]
EOF

# Commit
git add -A
git commit -m "Initial: sync policy, error patterns, coding standards"
```

---

## 7. Workspace Rules

### Create .cursor/rules

Create `<workspace>/.cursor/rules`:

```yaml
---
description: "Cursor AI Self-Regulating Memory System v9.0"
globs: ["**/*"]
alwaysApply: true
---

# MEMORY STRATEGY: Three-Layer System

## Layer 1: .cursor/rules (This File)
Auto-injected every conversation. Max 1 paragraph per memory.

## Layer 2: Pepper Memory Bank (~/memo/global-memories)
- Read: Use `memory_bank_read("global-memories", "filename.md")`
- Update: Use `memory_bank_update("global-memories", "filename.md", content)`

## Layer 3: global-kb/ Archive
Git-controlled. Source of truth for historical context only.

---

# SELF-REGULATION: Auto Memory Management

## When User Contradicts Memory
DELETE from Cursor memories. Always trust user input.

## When New Pattern Emerges
1. CREATE memory in .cursor/rules
2. UPDATE corresponding Pepper file

## When Procedure Discovered
UPDATE Pepper Memory Bank only (too detailed for rules).

## When Investigation Needed
WRITE findings to global-kb/archive/

---

# QUALITY GATES: Before Creating Memory

1. Needed EVERY conversation?
   - No → Use Pepper Memory Bank instead
   - Yes → OK for .cursor/rules

2. Fits ONE paragraph (max)?
   - No → Use Pepper Memory Bank instead
   - Yes → OK for .cursor/rules

3. Rule or pattern (not task)?
   - No → Don't memorize tasks, users repeat them
   - Yes → OK to memorize

---

# DEV STANDARDS

## Code Organization
- DRY: No repeated code patterns
- SOLID: Single responsibility principle
- CLEAN: Meaningful names, <20 lines per function

## Error Handling
- Log errors with context
- Implement retry logic for transient failures
- Fail fast, don't hide errors

## Testing
- Test happy path AND error cases
- Use meaningful assertion messages
- Include edge case coverage

---

# MCP TOOL ROUTING

Use MCP tools strategically:

| Need | Tool | Command |
|------|------|---------|
| Read procedure | Pepper | memory_bank_read(project, file) |
| Update procedure | Pepper | memory_bank_update(project, file, content) |
| Search knowledge | Memory MCP | search_nodes(query) |
| Create entity | Memory MCP | create_entities([...]) |
| Understand library | context7 | get-library-docs(id, topic) |
| Complex planning | sequential-thinking | problem_breakdown(problem) |
| Research question | perplexity-ask | perplexity_ask(messages) |

---

# HEALTH CHECK (Say "health check")

Runs:
1. read_file(".cursor/rules") — Verify rules exist
2. memory_bank_read("global-memories", "sync-policy.md") — Verify Pepper
3. Report any errors

---

# ON-ERROR RECOVERY

| Error | Recovery |
|-------|----------|
| Memory MCP unavailable | Skip, use Pepper Memory Bank |
| Pepper unavailable | Use .cursor/rules only |
| Rules missing | Alert user, cannot proceed |

---

# MEMORY EXAMPLES

## Example 1: Pattern (1 para, fits rules)
**Pattern**: Error X happens when Y. Solution: Z.
→ Store in .cursor/rules

## Example 2: Procedure (Too detailed, use Pepper)
**Procedure**: 10-step deployment with configs.
→ Store in Pepper Memory Bank

## Example 3: Archive (Historical only)
**Investigation**: Root cause analysis from incident.
→ Store in global-kb/archive/
```

### After Creating Rules

```bash
# Backup to global-kb
cp <workspace>/.cursor/rules \
   ~/Code/global-kb/cursor-config/rules/<workspace>.rules

# Initialize workspace git repo
cd ~/Code/global-kb
git init
git add -A
git commit -m "Initial: workspace rules backup"
```

---

## 8. Automated Setup Script

### Create ~/setup-cursor-memory.sh

```bash
#!/bin/bash

# Cursor AI Memory System: Automated Setup
# Version: 9.0
# Purpose: Complete setup in one command

set -e  # Exit on error

echo "🚀 Cursor AI Memory System Setup v9.0"
echo "======================================"

# Configuration
export HOME_DIR="$HOME"
export CODE_DIR="$HOME_DIR/Code"
export MEMO_DIR="$HOME_DIR/memo"
export KB_DIR="$CODE_DIR/global-kb"

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Helper functions
log_step() {
    echo -e "${BLUE}▸ $1${NC}"
}

log_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

log_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Step 1: Check prerequisites
log_step "Checking prerequisites..."
if ! command -v node &> /dev/null; then
    log_error "Node.js not found. Install Node.js 18+ first."
    exit 1
fi
if ! command -v git &> /dev/null; then
    log_error "Git not found. Install Git first."
    exit 1
fi
log_success "Prerequisites verified"

# Step 2: Create directories
log_step "Creating directories..."
mkdir -p "$MEMO_DIR/global-memories"
mkdir -p "$KB_DIR/archive"
mkdir -p "$KB_DIR/cursor-config/rules"
mkdir -p "$KB_DIR/architecture"
log_success "Directories created"

# Step 3: Initialize git repos
log_step "Initializing git repositories..."
if [ ! -d "$MEMO_DIR/.git" ]; then
    cd "$MEMO_DIR"
    git init
    git config user.name "Cursor AI"
    git config user.email "cursor@local"
    log_success "Pepper repo initialized"
fi

if [ ! -d "$KB_DIR/.git" ]; then
    cd "$KB_DIR"
    git init
    git config user.name "Cursor AI"
    git config user.email "cursor@local"
    log_success "Global KB repo initialized"
fi

# Step 4: Create Pepper Memory files
log_step "Creating Pepper Memory files..."

cat > "$MEMO_DIR/global-memories/sync-policy.md" << 'SYNC_EOF'
# Memory Sync Policy v9.0 (Self-Regulating)

## How It Works

| Layer | Auto-Injected | Purpose |
|-------|---|---------|
| `.cursor/rules` | Yes | Patterns (1 para max) |
| Cursor memories | Yes | Quick facts |
| Pepper Memory Bank | On demand | Procedures |
| global-kb/ | Manual | Archive only |

## Memory Actions (Rules Auto-Execute)

| Trigger | Action |
|---------|--------|
| User contradicts | DELETE memory |
| New pattern | CREATE rule + UPDATE Pepper |
| Procedure discovered | UPDATE Pepper only |
| Investigation | WRITE to global-kb/ |

## Quality Gates

1. Needed every conversation? → Yes = Rules, No = Pepper
2. Fits 1 paragraph? → Yes = Rules, No = Pepper
3. Rule not task? → Yes = Memorize, No = Don't

## Health Check

Say "health check" to verify system.

## Backup Locations

- Pepper: ~/memo (git repo)
- Rules: Code/global-kb/cursor-config/rules/
- Archive: Code/global-kb/archive/
SYNC_EOF

cat > "$MEMO_DIR/global-memories/error-patterns.md" << 'ERROR_EOF'
# Error Patterns & Solutions

## Common Issues

### MCP Server Not Available
**Pattern**: mcp_error: server_not_found
**Solution**:
1. Check ~/.cursor/mcp.json
2. Verify Node.js installed
3. Restart Cursor

### Memory Bank Read Failed
**Pattern**: Failed to read Pepper memory
**Solution**:
1. Check ~/memo/global-memories/ exists
2. Run: git -C ~/memo status
3. Verify git initialized

### Rules Not Loading
**Pattern**: .cursor/rules not in context
**Solution**:
1. Verify .cursor/rules exists in workspace
2. Check syntax (YAML format)
3. Restart Cursor
4. Say "health check"

## Prevention

- [ ] Run "health check" regularly
- [ ] Commit changes: git -C ~/memo commit -am "update"
- [ ] Monitor MCP status in Cursor logs
ERROR_EOF

cat > "$MEMO_DIR/global-memories/coding-standards.md" << 'CODING_EOF'
# Coding Standards & Patterns

## General Rules

- DRY: Extract common patterns
- SOLID: Single responsibility per function
- CLEAN CODE: Meaningful names, small functions
- Error handling: Fail fast, log clearly

## Testing

- Test happy path AND error cases
- Include edge case coverage
- Use meaningful assertion messages

## Documentation

- Add comments for non-obvious logic
- Document function parameters
- Include usage examples for public APIs
CODING_EOF

log_success "Pepper Memory files created"

# Step 5: Create MCP Configuration
log_step "Creating MCP configuration..."

# Escape $HOME for JSON
HOME_ESCAPED=$(echo "$HOME" | sed 's/\\/\\\\/g')

cat > "$HOME/.cursor/mcp.json" << MCP_EOF
{
  "mcpServers": {
    "allPepper-memory-bank": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@allpepper/memory-bank-mcp@latest"],
      "env": {
        "MEMORY_BANK_ROOT": "$HOME_ESCAPED/memo"
      }
    },
    "memory": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"]
    },
    "context7": {
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp"]
    },
    "sequential-thinking": {
      "command": "npx",
      "args": ["-y", "sequential-thinking-mcp"]
    },
    "fetch": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-fetch"]
    }
  }
}
MCP_EOF

log_success "MCP configuration created at ~/.cursor/mcp.json"

# Step 6: Create workspace rules template
log_step "Creating workspace rules template..."

cat > "$KB_DIR/cursor-config/rules/template.rules" << 'RULES_EOF'
---
description: "Cursor AI Self-Regulating Memory System v9.0"
globs: ["**/*"]
alwaysApply: true
---

# MEMORY STRATEGY: Three-Layer System

## Layer 1: .cursor/rules (This File)
Auto-injected every conversation. Max 1 paragraph per memory.

## Layer 2: Pepper Memory Bank (~/memo/global-memories)
- Read: Use `memory_bank_read("global-memories", "filename.md")`
- Update: Use `memory_bank_update("global-memories", "filename.md", content)`

## Layer 3: global-kb/ Archive
Git-controlled. Source of truth for historical context only.

---

# SELF-REGULATION: Auto Memory Management

## When User Contradicts Memory
DELETE from Cursor memories. Always trust user input.

## When New Pattern Emerges
1. CREATE memory in .cursor/rules
2. UPDATE corresponding Pepper file

## When Procedure Discovered
UPDATE Pepper Memory Bank only (too detailed for rules).

## When Investigation Needed
WRITE findings to global-kb/archive/

---

# QUALITY GATES: Before Creating Memory

1. Needed EVERY conversation?
   - No → Use Pepper Memory Bank instead
   - Yes → OK for .cursor/rules

2. Fits ONE paragraph (max)?
   - No → Use Pepper Memory Bank instead
   - Yes → OK for .cursor/rules

3. Rule or pattern (not task)?
   - No → Don't memorize tasks
   - Yes → OK to memorize

---

# DEV STANDARDS

## Code Organization
- DRY: No repeated patterns
- SOLID: Single responsibility
- CLEAN: Meaningful names, <20 lines per function

## Error Handling
- Log with context
- Retry transient failures
- Fail fast

## Testing
- Test happy path AND errors
- Include edge cases
- Use meaningful assertions

---

# MCP TOOL ROUTING

| Need | Tool |
|------|------|
| Read procedure | memory_bank_read(project, file) |
| Update procedure | memory_bank_update(project, file, content) |
| Search knowledge | search_nodes(query) |
| Complex planning | problem_breakdown(problem) |
| Research | perplexity_ask(messages) |

---

# HEALTH CHECK

Say "health check" to verify:
1. read_file(".cursor/rules")
2. memory_bank_read("global-memories", "sync-policy.md")

---

# ON-ERROR RECOVERY

| Error | Recovery |
|-------|----------|
| Memory MCP unavailable | Use Pepper Memory Bank |
| Pepper unavailable | Use .cursor/rules only |
| Rules missing | Alert user |
RULES_EOF

log_success "Workspace rules template created"

# Step 7: Commit to git
log_step "Committing to git..."
cd "$MEMO_DIR"
git add -A
git commit -m "Initial: Cursor memory system setup (sync-policy, error-patterns, coding-standards)" || true

cd "$KB_DIR"
git add -A
git commit -m "Initial: Cursor memory system setup (rules template)" || true

log_success "Changes committed to git"

# Step 8: Show summary
echo ""
echo "======================================"
echo -e "${GREEN}✓ Setup Complete!${NC}"
echo "======================================"
echo ""
echo "📍 Key Locations:"
echo "  • MCP Config: $HOME/.cursor/mcp.json"
echo "  • Pepper Memory: $MEMO_DIR/global-memories/"
echo "  • Global KB: $KB_DIR/"
echo ""
echo "📋 Next Steps:"
echo "  1. Restart Cursor IDE"
echo "  2. Copy template rules to your workspace:"
echo "     cp $KB_DIR/cursor-config/rules/template.rules <workspace>/.cursor/rules"
echo "  3. Say 'health check' in any conversation"
echo ""
echo "📖 Documentation: $KB_DIR/architecture/cursor-memory-system-v9.md"
echo ""
```

### Run Setup

```bash
# Make executable
chmod +x ~/setup-cursor-memory.sh

# Run
~/setup-cursor-memory.sh

# Expected output:
# ✓ Setup Complete!
# 
# 📍 Key Locations:
# • MCP Config: /Users/xxx/.cursor/mcp.json
# • Pepper Memory: /Users/xxx/memo/global-memories/
# • Global KB: /Users/xxx/Code/global-kb/
```

---

## 9. Verification

### Quick Verification (30 seconds)

In Cursor, say:

```
health check
```

Cursor will automatically:
1. Read .cursor/rules
2. Read sync-policy.md from Pepper
3. Report any errors

### Full Test Suite (5 minutes)

```bash
# Test 1-4: Rules
test -f .cursor/rules && echo "✓ Rules exist"
grep -q "Memory Strategy" .cursor/rules && echo "✓ Rules content valid"
test -d ~/Code/global-kb && echo "✓ Global KB exists"
test -f ~/Code/global-kb/cursor-config/rules/template.rules && echo "✓ Rules backup exists"

# Test 5-7: Pepper and git
test -f ~/memo/global-memories/sync-policy.md && echo "✓ Sync policy exists"
git -C ~/memo status > /dev/null 2>&1 && echo "✓ Pepper git repo valid"
git -C ~/Code/global-kb status > /dev/null 2>&1 && echo "✓ KB git repo valid"

# Test 8-10: MCP
test -f ~/.cursor/mcp.json && echo "✓ MCP config exists"
grep -q "allPepper-memory-bank" ~/.cursor/mcp.json && echo "✓ Pepper MCP configured"
grep -q '"memory"' ~/.cursor/mcp.json && echo "✓ Memory MCP configured"
```

### Manual Verification

```bash
# 1. Check MCP servers are recognized
cat ~/.cursor/mcp.json | jq '.mcpServers | keys'

# 2. Verify Pepper git status
git -C ~/memo log --oneline | head -3

# 3. Verify rules backup
ls -la ~/Code/global-kb/cursor-config/rules/

# 4. Check directory structure
tree -L 2 ~/memo/
tree -L 2 ~/Code/global-kb/
```

---

## 10. Maintenance

### Daily (Automatic)

- System self-regulates
- No daily prompts needed
- Say "health check" if uncertain

### Weekly (Manual)

```bash
# Update Pepper after edits
git -C ~/memo add -A
git -C ~/memo commit -m "update: description of change"

# Verify git status
git -C ~/memo status
git -C ~/Code/global-kb status
```

### Monthly (Review)

```bash
# 1. Archive old procedures
mv ~/memo/global-memories/old-procedure.md ~/Code/global-kb/archive/
git -C ~/memo add -A
git -C ~/memo commit -m "archive: old-procedure.md"

# 2. Update coding standards if needed
# Edit ~/memo/global-memories/coding-standards.md

# 3. Review error patterns
# Edit ~/memo/global-memories/error-patterns.md
```

### Backup Locations

- **Pepper Memory**: ~/memo/ (git repo)
- **Rules Backup**: ~/Code/global-kb/cursor-config/rules/
- **Archive**: ~/Code/global-kb/archive/

### Rollback

```bash
# Rollback Pepper to previous version
git -C ~/memo log --oneline | head -5
git -C ~/memo reset --hard <commit-hash>

# Rollback Global KB
git -C ~/Code/global-kb reset --hard <commit-hash>
```

---

## 11. Troubleshooting

### Issue: "health check" fails to read rules

**Diagnosis**:
```bash
# Check rules file exists
test -f .cursor/rules && echo "exists" || echo "missing"

# Check rules syntax (YAML)
cat .cursor/rules | head -10
```

**Solution**:
1. Copy template: `cp ~/Code/global-kb/cursor-config/rules/template.rules .cursor/rules`
2. Verify YAML syntax (no tab characters)
3. Restart Cursor

### Issue: Pepper Memory Bank not accessible

**Diagnosis**:
```bash
# Check directory exists
test -d ~/memo/global-memories && echo "exists" || echo "missing"

# Check git status
git -C ~/memo status

# Check file permissions
ls -la ~/memo/global-memories/
```

**Solution**:
1. Run setup script again: `bash ~/setup-cursor-memory.sh`
2. Verify git initialized: `git -C ~/memo log`
3. Restart Cursor

### Issue: MCP server errors

**Diagnosis**:
```bash
# Check MCP config syntax
cat ~/.cursor/mcp.json | jq . > /dev/null && echo "valid JSON" || echo "invalid JSON"

# Verify Node.js
node --version

# Check if npm packages can be run
npx -y @allpepper/memory-bank-mcp@latest --help
```

**Solution**:
1. Fix ~/.cursor/mcp.json syntax
2. Upgrade Node.js: `node --version` should be 18+
3. Restart Cursor
4. Check Cursor logs: Help > Show Logs

### Issue: Can't commit to git

**Diagnosis**:
```bash
# Check git initialized
git -C ~/memo log --oneline

# Check git config
git -C ~/memo config --list
```

**Solution**:
```bash
# Reinitialize
cd ~/memo
git init
git config user.name "Your Name"
git config user.email "your@email.com"
git add -A
git commit -m "resync after error"
```

---

## Quick Reference

### Essential Commands

```bash
# Health check (in Cursor)
"health check"

# Commit Pepper changes
git -C ~/memo add -A && git -C ~/memo commit -m "update"

# View Pepper history
git -C ~/memo log --oneline

# Archive a file
mv ~/memo/global-memories/file.md ~/Code/global-kb/archive/
git -C ~/memo add -A && git -C ~/memo commit -m "archive: file.md"

# Rollback
git -C ~/memo reset --hard HEAD~1

# Verify setup
bash ~/setup-cursor-memory.sh  # Idempotent - safe to run again
```

### File Locations Cheat Sheet

| Item | Location |
|------|----------|
| MCP Config | ~/.cursor/mcp.json |
| Pepper Memory | ~/memo/global-memories/ |
| Global KB | ~/Code/global-kb/ |
| Rules Backup | ~/Code/global-kb/cursor-config/rules/ |
| Rules (Workspace) | <workspace>/.cursor/rules |
| This Guide | ~/Code/global-kb/architecture/cursor-memory-system-v9.md |

### MCP Tools Quick Reference

```bash
# Read Pepper file
memory_bank_read("global-memories", "sync-policy.md")

# Update Pepper file
memory_bank_update("global-memories", "filename.md", "content")

# Search knowledge graph
search_nodes("query terms")

# Complex planning
problem_breakdown("describe problem")
```

---

## System Principles

### KISS (Keep It Simple, Stupid)

- Three layers only (rules, Pepper, KB)
- No complex configurations
- One file per concept
- Automatic self-regulation

### DRY (Don't Repeat Yourself)

- Shared sync-policy for all instances
- Template rules for all workspaces
- One git config across system
- Reusable error patterns

### SOLID

- **Single Responsibility**: Each layer has one purpose
- **Open/Closed**: Extensible via MCP tools
- **Liskov Substitution**: Layers interchangeable on failure
- **Interface Segregation**: Clean MCP interfaces
- **Dependency Inversion**: Depends on abstractions (git, MCP)

### CLEAN CODE

- Meaningful file names
- Self-documenting structure
- Clear error messages
- Automated testing (health check)

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 9.0 | 2025-12-16 | Single consolidated file, automated setup script, KISS/DRY/SOLID principles |
| 8.1 | 2025-12-12 | Portable guide, 12 tests, MCP documentation |
| 8.0 | 2025-12-12 | Self-regulation framework |
| 7.0 | 2025-12-12 | KISS simplification |

---

## Support

### Need Help?

1. **Quick Issues**: Say "health check" (auto-diagnosis)
2. **File Issues**: Check ~/memo/global-memories/error-patterns.md
3. **Setup Issues**: Re-run `~/setup-cursor-memory.sh`
4. **Git Issues**: Review git status and logs

### Contributing

To improve this system:

1. Update relevant .md files in ~/memo/global-memories/
2. Test with `health check`
3. Commit: `git -C ~/memo commit -am "improve: description"`
4. Archive old versions to ~/Code/global-kb/archive/

---

**Created**: 2025-12-16  
**Status**: Production-Ready  
**License**: MIT (Use freely, improve continuously)  
**Last Updated**: 2025-12-16

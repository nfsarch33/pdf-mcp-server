# Cursor AI Memory System - Complete Cross-Platform Setup Guide

**Version**: 3.0 - Combined & Improved  
**Updated**: January 27, 2026  
**Target Platforms**: macOS + Windows 11 (WSL2 Ubuntu)  
**Status**: ✅ Production Ready with GitHub Backup Integration  

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [GitHub Private Repositories](#2-github-private-repositories)
3. [Prerequisites](#3-prerequisites)
4. [Quick Setup (Recommended - 5 minutes)](#4-quick-setup-recommended--5-minutes)
5. [macOS Setup (Fresh Install)](#5-macos-setup-fresh-install)
6. [Windows 11 + WSL2 Ubuntu Setup (Fresh Install)](#6-windows-11--wsl2-ubuntu-setup-fresh-install)
7. [Path Mappings & MCP Configuration](#7-path-mappings--mcp-configuration)
8. [Workspace Rules Setup](#8-workspace-rules-setup)
9. [Cross-Platform Synchronization](#9-cross-platform-synchronization)
10. [Health Check & Verification](#10-health-check--verification)
11. [Regular Maintenance](#11-regular-maintenance)
12. [Troubleshooting](#12-troubleshooting)
13. [Quick Reference Card](#13-quick-reference-card)

---

## 1. Architecture Overview

### Three-Layer Memory System

```
┌──────────────────────────────────────────────────────────────┐
│         LAYER 1: AUTO-INJECTED (Every Conversation)          │
├───────────────────────────────┬──────────────────────────────┤
│ .cursor/rules                 │ Cursor Built-in Memories     │
│ • Memory strategy (50-100 ln) │ • Code patterns              │
│ • Self-regulation logic       │ • Standards (max 1 para)     │
│ • Development standards       │ • Auto-injected always       │
├───────────────────────────────┴──────────────────────────────┤
│ Status: ALWAYS LOADED - System health check via visibility   │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│      LAYER 2: ON-DEMAND (Via MCP Tools - MCP-Accessed)      │
├───────────────────────────────┬──────────────────────────────┤
│ Pepper Memory Bank            │ Memory MCP Servers           │
│ ~/memo/global-memories/       │ • Knowledge graph            │
│ • Detailed procedures         │ • Entity relationships       │
│ • Framework patterns          │ • Context linking            │
│ • Error solutions             │ • Automatic learning         │
├───────────────────────────────┴──────────────────────────────┤
│ Status: ON-DEMAND - Synced to GitHub (private repo)          │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│    LAYER 3: ARCHIVE (Git-Controlled + GitHub Backup)        │
├──────────────────────────────────────────────────────────────┤
│ global-kb/ (~Code/global-kb)                                 │
│ • Completed investigations & research                        │
│ • Architecture decision records (ADRs)                       │
│ • Historical documentation                                   │
│ • Rules backups & configuration                             │
│ • Lessons learned & patterns                                │
├──────────────────────────────────────────────────────────────┤
│ Status: PERMANENT RECORD - Full git history + GitHub sync    │
└──────────────────────────────────────────────────────────────┘
```

### Memory Actions Decision Tree

| Trigger | Action | Layer | Example |
|---------|--------|-------|---------|
| User contradicts info | DELETE Cursor memory immediately | Layer 1 | Remove outdated pattern |
| New pattern needed EVERY chat | CREATE Cursor (1 para) + UPDATE Pepper | Both | Development standard |
| Procedure/guide discovered | UPDATE Pepper only | Layer 2 | Error fix walkthrough |
| Investigation completed | WRITE to global-kb/, git commit | Layer 3 | Research findings |

### Quality Gates (Before Creating Cursor Memory)

- ✅ Needed EVERY conversation? (No → Pepper)
- ✅ Fits 1 paragraph? (No → Pepper)
- ✅ Is it a rule/pattern, not a task? (Task → Don't memorize)

---

## 2. GitHub Private Repositories

### Remote Backup Repositories (Private)

Your GitHub account has two private repositories for backup and cross-device sync:

| Repository | Purpose | URL | Local Path |
|------------|---------|-----|------------|
| **cursor-memory-bank** | Pepper Memory Bank backup | `git@github.com:<org-or-user>/cursor-memory-bank.git` | `~/memo` |
| **cursor-global-kb** | Global KB archive backup | `git@github.com:<org-or-user>/cursor-global-kb.git` | `~/Code/global-kb` |

### Repository Structure

**cursor-memory-bank** (Pepper):
```
├── README.md                                    # System overview
├── SETUP.md                                     # Detailed setup guide
├── ARCHITECTURE.md                              # System design
├── global-memories/
│   ├── sync-policy.md                          # v9.0 - Memory sync rules
│   ├── error-patterns.md                        # Common error solutions
│   ├── architecture-decisions.md                # Design decisions
│   └── [other memory files]
└── [templates and configuration]
```

**cursor-global-kb** (Archive):
```
├── README.md                                    # Archive guidelines
├── architecture/                                # System design docs
├── investigations/                              # Research & analysis
├── decisions/                                   # ADRs (Architecture Decision Records)
├── patterns/                                    # Code patterns
├── lessons-learned/                             # Historical insights
├── archive/                                     # Moved from Pepper
├── cursor-config/rules/                         # Rules backups
├── timelines/                                   # Progress records
└── research/                                    # General research
```

### SSH Configuration

```bash
# Your SSH key for GitHub
# Location: ~/.ssh/<key-name>
# Type: ed25519

# SSH config (~/.ssh/config):
Host github.com
  HostName github.com
  User git
  IdentityFile ~/.ssh/<key-name>
```

---

## 3. Prerequisites

### macOS Prerequisites

```bash
# Using Homebrew (recommended)
brew install node                           # Node.js 20+
brew install git                            # Git 2.34+

# Install uv (for uvx commands)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to shell profile if needed
source ~/.bashrc  # or ~/.zshrc

# Verify installation
node --version                              # Should be v20+
npm --version
git --version
uv --version
```

### Windows 11 + WSL2 Ubuntu Prerequisites

**On Windows (PowerShell as Administrator)**:

```powershell
# 1. Enable WSL2 if not already enabled
wsl --install -d Ubuntu

# 2. Install Docker Desktop (optional but recommended)
# Download from: https://www.docker.com/products/docker-desktop/
# Enable WSL2 integration: Settings > Resources > WSL Integration

# 3. Cursor IDE
# Download from: https://cursor.sh

# Verify WSL2 installation
wsl --list --verbose
# Should show Ubuntu with VERSION 2
```

**Inside WSL2 Ubuntu Terminal**:

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Node.js 20
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs

# Install git (usually pre-installed)
sudo apt install -y git

# Install uv (for uvx commands)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Verify installation
node --version                              # Should be v20+
npm --version
git --version
uv --version
```

### Prerequisite Verification Script

Save and run this on your target platform:

```bash
#!/bin/bash
echo "=== Cursor Memory System Prerequisites Check ==="
echo ""
echo -n "✓ Node.js:    "; node --version 2>/dev/null || echo "❌ NOT INSTALLED"
echo -n "✓ npm:        "; npm --version 2>/dev/null || echo "❌ NOT INSTALLED"
echo -n "✓ Git:        "; git --version 2>/dev/null || echo "❌ NOT INSTALLED"
echo -n "✓ uv:         "; uv --version 2>/dev/null || echo "❌ NOT INSTALLED"
echo -n "✓ Docker:     "; docker --version 2>/dev/null || echo "⚠ OPTIONAL - not installed"
echo ""
echo "============================================="
echo "Result: Ready to proceed if all show versions"
echo "============================================="
```

---

## 4. Quick Setup (Recommended - 5 minutes)

### FASTEST PATH: Clone from GitHub

This is the fastest way to set up on any new machine. Requires GitHub SSH key configured.

#### Step 1: Configure SSH Key (One-time)

```bash
# Generate SSH key (if you don't have one)
ssh-keygen -t ed25519 -f ~/.ssh/<key-name> -C "your-email@example.com"

# Start SSH agent and add key
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/<key-name>

# Display public key to add to GitHub
cat ~/.ssh/<key-name>.pub

# Copy the output and add to GitHub:
# Settings > SSH and GPG keys > New SSH key
# Paste the output and save
```

#### Step 2: Configure SSH Config

```bash
# Edit or create ~/.ssh/config
cat >> ~/.ssh/config << 'EOF'
Host github.com
  HostName github.com
  User git
  IdentityFile ~/.ssh/<key-name>
  AddKeysToAgent yes
EOF

# Test SSH connection
ssh -T git@github.com
# Should output: "Hi nfsarch33! You've successfully authenticated..."
```

#### Step 3: Clone Repositories (Both Platforms)

**macOS**:
```bash
# Clone Pepper Memory Bank
git clone git@github.com:<org-or-user>/cursor-memory-bank.git ~/memo

# Clone Global KB
git clone git@github.com:<org-or-user>/cursor-global-kb.git ~/Code/global-kb

# Create workspace directory
mkdir -p ~/Code/workspace/.cursor

# Copy rules from global-kb
cp ~/Code/global-kb/cursor-config/rules/zendesk-workspace.rules \
   ~/Code/workspace/.cursor/rules

# Configure git for each repo
git -C ~/memo config user.name "Your Name"
git -C ~/memo config user.email "your-email@example.com"
git -C ~/Code/global-kb config user.name "Your Name"
git -C ~/Code/global-kb config user.email "your-email@example.com"

# Verify clones
ls -la ~/memo/global-memories/sync-policy.md
ls -la ~/Code/global-kb/README.md
```

**WSL Ubuntu** (Same commands):
```bash
# Clone Pepper Memory Bank
git clone git@github.com:<org-or-user>/cursor-memory-bank.git ~/memo

# Clone Global KB
git clone git@github.com:<org-or-user>/cursor-global-kb.git ~/Code/global-kb

# Create workspace directory
mkdir -p ~/Code/workspace/.cursor

# Copy rules from global-kb
cp ~/Code/global-kb/cursor-config/rules/zendesk-workspace.rules \
   ~/Code/workspace/.cursor/rules

# Configure git for each repo
git -C ~/memo config user.name "Your Name"
git -C ~/memo config user.email "your-email@example.com"
git -C ~/Code/global-kb config user.name "Your Name"
git -C ~/Code/global-kb config user.email "your-email@example.com"

# Verify clones
ls -la ~/memo/global-memories/sync-policy.md
ls -la ~/Code/global-kb/README.md
```

#### Step 4: Configure MCP

See [Section 7: MCP Configuration](#7-path-mappings--mcp-configuration)

#### Step 5: Verify System

In Cursor, open any conversation and type: **"health check"**

Expected output:
```
✅ Rules file readable (X lines, auto-injected)
✅ Pepper Memory Bank accessible
✅ sync-policy.md v9.0 loaded
✅ All MCP servers responsive
✅ System: HEALTHY
```

**Total time: ~5 minutes** ⏱️

---

## 5. macOS Setup (Fresh Install)

Use this path if you prefer manual setup or don't have GitHub SSH configured.

### Step 1: Install Prerequisites

```bash
# Install Node.js with Homebrew
brew install node

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify
node --version    # v20+
npm --version
uv --version
```

### Step 2: Create Directory Structure

```bash
# Create all required directories
mkdir -p ~/memo/global-memories
mkdir -p ~/Code/global-kb/architecture
mkdir -p ~/Code/global-kb/archive
mkdir -p ~/Code/global-kb/cursor-config/rules
mkdir -p ~/Code/workspace/.cursor

# Verify
echo "Created directories:"
ls -d ~/memo ~/Code/global-kb ~/Code/workspace
```

### Step 3: Initialize Local Git Repositories

```bash
# Initialize Pepper Memory Bank
cd ~/memo
git init
git config user.email "your-email@example.com"
git config user.name "Your Name"

# Initialize global-kb
cd ~/Code/global-kb
git init
git config user.email "your-email@example.com"
git config user.name "Your Name"

# Create initial commit for both
echo "# Pepper Memory Bank" > ~/memo/README.md
cd ~/memo && git add README.md && git commit -m "Initial commit: Pepper Memory Bank setup"

echo "# Global Knowledge Base" > ~/Code/global-kb/README.md
cd ~/Code/global-kb && git add README.md && git commit -m "Initial commit: Global KB setup"
```

### Step 4: Create Memory Bank Files

```bash
# Create sync-policy.md (v9.0)
cat > ~/memo/global-memories/sync-policy.md << 'EOF'
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
|---------|--------|
| User contradicts | DELETE Cursor memory |
| New pattern | CREATE Cursor + UPDATE Pepper |
| Procedure | UPDATE Pepper only |
| Investigation | WRITE to global-kb/ |

## GitHub Backup

**Repositories**:
- Pepper: git@github.com:<org-or-user>/cursor-memory-bank.git
- KB: git@github.com:<org-or-user>/cursor-global-kb.git

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
EOF

# Create error-patterns.md
cat > ~/memo/global-memories/error-patterns.md << 'EOF'
# Common Error Patterns and Solutions

## Go Development

### err113: Dynamic errors
**Problem**: `err113: do not define dynamic errors`
**Solution**: Use package-level sentinel errors

### errorlint: Error wrapping
**Problem**: Type assertions on errors
**Solution**: Use `errors.As` or `errors.Is`

## General Patterns

### Long lines (lll)
- Break function calls across multiple lines
- Max 120 characters per line
- Use variables for long expressions
EOF

# Commit new files
cd ~/memo
git add -A
git commit -m "add: sync-policy.md v9.0 and error-patterns.md"
```

### Step 5: Configure MCP

See [Section 7: MCP Configuration](#7-path-mappings--mcp-configuration)

### Step 6: Create Workspace Rules

```bash
# Create .cursor/rules file
mkdir -p ~/Code/workspace/.cursor

cat > ~/Code/workspace/.cursor/rules << 'EOF'
---
description: "Workspace Development Rules"
globs: ["**/*"]
alwaysApply: true
---

# Memory Strategy

## Three-Layer Architecture
1. **Cursor memories** = Auto-injected (in context always)
2. **Pepper Memory Bank** = On-demand (search with MCP)
3. **global-kb/** = Archive (git-controlled)

## Memory Actions
| Trigger | Action |
|---------|--------|
| User contradicts | DELETE Cursor memory |
| New pattern | CREATE Cursor + UPDATE Pepper |
| Procedure | UPDATE Pepper only |
| Investigation | WRITE to global-kb/ |

## Self-Regulation (Built-In)

### Implicit Health Check
Rules visible in context = System healthy. No explicit test needed.

### On-Error Recovery
| Failure | Auto-Recovery |
|---------|---------------|
| Memory MCP error | Skip it, use Pepper fallback |
| Pepper read fails | Use rules only |
| Rules missing | Alert user |

### Explicit Health Check
Say "health check" to verify system status
EOF

# Commit rules
cd ~/Code/workspace
git init
git config user.email "your-email@example.com"
git config user.name "Your Name"
git add .cursor/rules
git commit -m "Initial workspace rules setup"

# Also backup to global-kb
cp ~/Code/workspace/.cursor/rules ~/Code/global-kb/cursor-config/rules/zendesk-workspace.rules
cd ~/Code/global-kb
git add cursor-config/rules/zendesk-workspace.rules
git commit -m "backup: workspace rules"
```

### Step 7: Verify macOS Setup

```bash
# Check all files exist
echo "=== macOS Setup Verification ==="
echo -n "✓ Memo exists: "; [ -d ~/memo ] && echo "OK" || echo "MISSING"
echo -n "✓ global-kb exists: "; [ -d ~/Code/global-kb ] && echo "OK" || echo "MISSING"
echo -n "✓ Workspace .cursor exists: "; [ -d ~/Code/workspace/.cursor ] && echo "OK" || echo "MISSING"
echo -n "✓ sync-policy.md: "; [ -f ~/memo/global-memories/sync-policy.md ] && echo "OK" || echo "MISSING"
echo -n "✓ Rules file: "; [ -f ~/Code/workspace/.cursor/rules ] && echo "OK" || echo "MISSING"
echo -n "✓ Node.js: "; node --version
echo -n "✓ npm: "; npm --version
echo -n "✓ Git: "; git --version
echo "=============================="
```

---

## 6. Windows 11 + WSL2 Ubuntu Setup (Fresh Install)

### Step 1: Enable WSL2 (Windows Side)

```powershell
# PowerShell as Administrator
wsl --install -d Ubuntu

# Wait for installation to complete, then verify
wsl --list --verbose
# Output should show: Ubuntu   Running   2
```

### Step 2: Install Prerequisites (WSL Ubuntu Side)

```bash
# Inside WSL Ubuntu terminal
sudo apt update && sudo apt upgrade -y

# Install Node.js 20
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Verify
node --version    # v20+
npm --version
git --version
uv --version
```

### Step 3: Create Directory Structure (WSL Side)

```bash
# Inside WSL Ubuntu terminal
mkdir -p ~/memo/global-memories
mkdir -p ~/Code/global-kb/architecture
mkdir -p ~/Code/global-kb/archive
mkdir -p ~/Code/global-kb/cursor-config/rules
mkdir -p ~/Code/workspace/.cursor

# Verify
ls -d ~/memo ~/Code/global-kb ~/Code/workspace/.cursor
```

### Step 4: Initialize Git Repositories (WSL Side)

```bash
# Initialize Pepper Memory Bank
cd ~/memo
git init
git config user.email "your-email@example.com"
git config user.name "Your Name"

# Initialize global-kb
cd ~/Code/global-kb
git init
git config user.email "your-email@example.com"
git config user.name "Your Name"

# Create initial commits
echo "# Pepper Memory Bank" > ~/memo/README.md
cd ~/memo && git add README.md && git commit -m "Initial commit: Pepper Memory Bank setup"

echo "# Global Knowledge Base" > ~/Code/global-kb/README.md
cd ~/Code/global-kb && git add README.md && git commit -m "Initial commit: Global KB setup"
```

### Step 5: Create Memory Bank Files (WSL Side)

Same as [Step 4 in macOS Setup](#step-4-create-memory-bank-files) - copy those commands exactly.

### Step 6: Configure MCP (Windows Side - NOT WSL)

**Important**: MCP config goes on Windows, not in WSL.

```powershell
# PowerShell - NOT WSL
cd $env:USERPROFILE

# Create .cursor directory if needed
New-Item -ItemType Directory -Force -Path .\.cursor

# Create MCP config
$mcp_config = @"
{
  "mcpServers": {
    "allPepper-memory-bank": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@allpepper/memory-bank-mcp@latest"],
      "env": {
        "MEMORY_BANK_ROOT": "/home/[YOUR_WSL_USERNAME]/memo"
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
    "git-mcp-server": {
      "command": "npx",
      "args": ["@cyanheads/git-mcp-server"],
      "env": {
        "MCP_LOG_LEVEL": "info",
        "GIT_SIGN_COMMITS": "false"
      }
    }
  }
}
"@

# Save to file
$mcp_config | Out-File -FilePath ".\.cursor\mcp.json" -Encoding UTF8

# Open in editor to update YOUR_WSL_USERNAME
notepad ".\.cursor\mcp.json"
```

**Replace `[YOUR_WSL_USERNAME]` with your actual WSL username:**

```bash
# Find your WSL username:
# In WSL terminal, run:
whoami
# Output: your-wsl-username

# Then update mcp.json:
# Change: "/home/[YOUR_WSL_USERNAME]/memo"
# To:     "/home/your-wsl-username/memo"
```

### Step 7: Create Workspace Rules (WSL Side)

Same as [Step 6 in macOS Setup](#step-6-create-workspace-rules).

### Step 8: Verify WSL Setup

```bash
# In WSL Ubuntu terminal
echo "=== WSL Setup Verification ==="
echo -n "✓ Memo exists: "; [ -d ~/memo ] && echo "OK" || echo "MISSING"
echo -n "✓ global-kb exists: "; [ -d ~/Code/global-kb ] && echo "OK" || echo "MISSING"
echo -n "✓ Workspace exists: "; [ -d ~/Code/workspace/.cursor ] && echo "OK" || echo "MISSING"
echo -n "✓ sync-policy.md: "; [ -f ~/memo/global-memories/sync-policy.md ] && echo "OK" || echo "MISSING"
echo -n "✓ Rules file: "; [ -f ~/Code/workspace/.cursor/rules ] && echo "OK" || echo "MISSING"
echo -n "✓ Node.js: "; node --version
echo -n "✓ npm: "; npm --version
echo ""
echo "Check Windows side:"
echo "✓ MCP config: C:\\Users\\$USERNAME\\.cursor\\mcp.json"
echo "  (Verify MEMORY_BANK_ROOT path matches your username)"
echo "=============================="
```

---

## 7. Path Mappings & MCP Configuration

### Path Reference by Platform

| Component | macOS | WSL Ubuntu |
|-----------|-------|-----------|
| Home directory | `/Users/<username>` | `/home/<username>` |
| Pepper Memory | `~/memo` | `~/memo` |
| global-kb | `~/Code/global-kb` | `~/Code/global-kb` |
| Workspace | `~/Code/workspace` | `~/Code/workspace` |
| MCP config | `~/.cursor/mcp.json` | Windows: `C:\Users\<user>\.cursor\mcp.json` |
| Rules file | `~/.cursor/rules` | `~/.cursor/rules` |

### Accessing WSL Files from Windows

```
\\wsl$\Ubuntu\home\<username>\memo
\\wsl$\Ubuntu\home\<username>\Code\global-kb
```

### MCP Configuration Details

**macOS mcp.json**:

```json
{
  "mcpServers": {
    "allPepper-memory-bank": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@allpepper/memory-bank-mcp@latest"],
      "env": {
        "MEMORY_BANK_ROOT": "/Users/your-username/memo"
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
    "git-mcp-server": {
      "command": "npx",
      "args": ["@cyanheads/git-mcp-server"],
      "env": {
        "MCP_LOG_LEVEL": "info",
        "GIT_SIGN_COMMITS": "false"
      }
    }
  }
}
```

**Windows WSL mcp.json** (on Windows, NOT in WSL):

```json
{
  "mcpServers": {
    "allPepper-memory-bank": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@allpepper/memory-bank-mcp@latest"],
      "env": {
        "MEMORY_BANK_ROOT": "/home/your-wsl-username/memo"
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
    "git-mcp-server": {
      "command": "npx",
      "args": ["@cyanheads/git-mcp-server"],
      "env": {
        "MCP_LOG_LEVEL": "info",
        "GIT_SIGN_COMMITS": "false"
      }
    }
  }
}
```

### MCP Server Reference

| Server | Purpose | Status |
|--------|---------|--------|
| allPepper-memory-bank | Pepper Memory Bank access | REQUIRED |
| memory | Knowledge graph | REQUIRED |
| context7 | External documentation | RECOMMENDED |
| sequential-thinking | Complex reasoning | RECOMMENDED |
| git-mcp-server | Git automation | OPTIONAL |

**After creating/updating mcp.json**: Restart Cursor IDE and check for green MCP status indicator.

---

## 8. Workspace Rules Setup

### Workspace Rules File

Located at: `~/Code/workspace/.cursor/rules`

```markdown
---
description: "Cursor Memory System - Workspace Rules"
globs: ["**/*"]
alwaysApply: true
---

# Memory Strategy

## Three-Layer Architecture

Your system has three memory layers that work together automatically:

### Layer 1: .cursor/rules (Always in Context)
- Memory strategy and self-regulation
- Development standards and patterns
- This file is auto-injected into every conversation
- System health check: If this file is visible → System is healthy

### Layer 2: Pepper Memory Bank (On-Demand via MCP)
- Location: ~/memo/global-memories/
- Detailed procedures, guides, and error solutions
- Framework-specific patterns and best practices
- Search and access via MCP memory tools

### Layer 3: global-kb (Permanent Archive)
- Location: ~/Code/global-kb/
- Completed investigations and research
- Architecture decision records (ADRs)
- Historical documentation
- Synced to GitHub private repo

## Memory Actions (Follow This Decision Tree)

### When You Discover New Information

| Situation | Action | Where |
|-----------|--------|-------|
| User contradicts existing info | DELETE from Cursor memory | Layer 1 |
| New pattern needed EVERY chat | CREATE in Cursor + UPDATE Pepper | Layer 1 + 2 |
| Procedure/guide discovered | UPDATE in Pepper | Layer 2 |
| Investigation completed | WRITE to global-kb/ | Layer 3 |

### Quality Gates (Before Creating Cursor Memory)

1. **Is this needed EVERY conversation?**
   - YES: Continue to next gate
   - NO: Put in Pepper Memory Bank only

2. **Does it fit in 1 paragraph?**
   - YES: Create Cursor memory
   - NO: Put in Pepper Memory Bank

3. **Is it a rule/pattern, not a task?**
   - YES: Can memorize
   - NO: Don't memorize (tasks are temporary)

## Self-Regulation (Automatic)

### Implicit Health Check
- Rules file is visible in this context → System is HEALTHY
- If not visible → Check .cursor/rules file exists
- If MCP servers fail → System falls back to rules only

### Explicit Health Check
Say "**health check**" in any conversation to verify:
1. Rules file loads correctly
2. Pepper Memory Bank accessible
3. All MCP servers responsive
4. System status: HEALTHY or issues

## Professional Standards

- No emojis in code, configs, commits, documentation
- Lowercase log messages with structured fields
- Conventional commits: `type: description`
- Type examples: `add`, `update`, `fix`, `archive`, `sync`

## GitHub Synchronization

Your repositories sync with GitHub:

```bash
# Push your local changes
cd ~/memo && git add -A && git commit -m "update: description" && git push
cd ~/Code/global-kb && git add -A && git commit -m "archive: description" && git push

# Pull from other machines
cd ~/memo && git pull
cd ~/Code/global-kb && git pull
```
```

---

## 9. Cross-Platform Synchronization

### Syncing Between macOS and WSL2

#### Method 1: GitHub Sync (Recommended)

```bash
# On First Machine (macOS)
# Push all changes
cd ~/memo && git add -A && git commit -m "sync: current state" && git push origin main
cd ~/Code/global-kb && git add -A && git commit -m "sync: current state" && git push origin main

# On Second Machine (WSL2)
# Pull all changes
cd ~/memo && git pull origin main
cd ~/Code/global-kb && git pull origin main
```

#### Method 2: Manual Sync via Cloud Storage

```bash
# Create sync archive on macOS
tar czf ~/cursor-memories-backup.tar.gz ~/memo ~/Code/global-kb

# Copy to cloud storage (Dropbox, OneDrive, etc.)
cp ~/cursor-memories-backup.tar.gz /Volumes/cloud/

# Extract on WSL Ubuntu
tar xzf ~/cursor-memories-backup.tar.gz -C ~/
```

### Regular Sync Workflow

**Weekly Push** (Friday):
```bash
cd ~/memo
git add -A
git commit -m "sync: weekly update - $(date +%Y-%m-%d)"
git push origin main

cd ~/Code/global-kb
git add -A
git commit -m "sync: weekly update - $(date +%Y-%m-%d)"
git push origin main
```

**Daily Sync** (Before switching machines):
```bash
# Push from current machine
cd ~/memo && git push origin main
cd ~/Code/global-kb && git push origin main

# Pull on next machine
cd ~/memo && git pull origin main
cd ~/Code/global-kb && git pull origin main
```

### Resolving Sync Conflicts

```bash
# If conflicts occur on pull
cd ~/memo
git status

# See what changed
git diff

# If you want remote version
git checkout --theirs global-memories/[filename]
git add [filename]
git commit -m "resolve: accept remote version"
git push origin main

# If you want local version
git checkout --ours global-memories/[filename]
git add [filename]
git commit -m "resolve: accept local version"
git push origin main
```

---

## 10. Health Check & Verification

### Quick Health Check (In Cursor)

**Say**: "health check"

**Expected Response**:
```
✅ Rules file readable (83 lines, auto-injected)
✅ Pepper Memory Bank accessible
✅ sync-policy.md v9.0 loaded
✅ All MCP servers responsive
✅ System: HEALTHY
```

### Detailed Health Check Script

**macOS**:
```bash
#!/bin/bash
echo "=== Cursor Memory System Health Check ==="
echo ""

PASS=0
FAIL=0

# Check 1: Rules file
echo -n "T1: Rules file exists... "
if [ -f ~/Code/workspace/.cursor/rules ]; then
    echo "✅ PASS"; ((PASS++))
else
    echo "❌ FAIL"; ((FAIL++))
fi

# Check 2: Pepper exists
echo -n "T2: Pepper Memory Bank exists... "
if [ -d ~/memo/global-memories ]; then
    echo "✅ PASS"; ((PASS++))
else
    echo "❌ FAIL"; ((FAIL++))
fi

# Check 3: sync-policy.md
echo -n "T3: sync-policy.md exists... "
if [ -f ~/memo/global-memories/sync-policy.md ]; then
    echo "✅ PASS"; ((PASS++))
else
    echo "❌ FAIL"; ((FAIL++))
fi

# Check 4: global-kb
echo -n "T4: global-kb exists... "
if [ -d ~/Code/global-kb ]; then
    echo "✅ PASS"; ((PASS++))
else
    echo "❌ FAIL"; ((FAIL++))
fi

# Check 5: MCP config
echo -n "T5: MCP config exists... "
if [ -f ~/.cursor/mcp.json ]; then
    echo "✅ PASS"; ((PASS++))
else
    echo "❌ FAIL"; ((FAIL++))
fi

# Check 6: Node.js
echo -n "T6: Node.js available... "
if command -v node &> /dev/null; then
    echo "✅ PASS ($(node --version))"; ((PASS++))
else
    echo "❌ FAIL"; ((FAIL++))
fi

# Check 7: Git in ~/memo
echo -n "T7: Memo is git repository... "
if [ -d ~/memo/.git ]; then
    echo "✅ PASS"; ((PASS++))
else
    echo "❌ FAIL"; ((FAIL++))
fi

# Check 8: Git in global-kb
echo -n "T8: global-kb is git repository... "
if [ -d ~/Code/global-kb/.git ]; then
    echo "✅ PASS"; ((PASS++))
else
    echo "❌ FAIL"; ((FAIL++))
fi

echo ""
echo "=== Results: $PASS/8 PASS, $FAIL/8 FAIL ==="
if [ $FAIL -eq 0 ]; then
    echo "✅ SYSTEM HEALTHY"
else
    echo "⚠️ Some components need attention"
fi
```

**WSL Ubuntu** (Same script works in WSL)

---

## 11. Regular Maintenance

### Daily (0 minutes)
- System auto-injects rules
- No manual action needed

### Weekly (5-10 minutes)
```bash
# Optional: Push to GitHub
cd ~/memo && git push origin main
cd ~/Code/global-kb && git push origin main
```

### Monthly (15-20 minutes)

```bash
# Archive old Pepper files to global-kb
cd ~/memo/global-memories
ls -lt | head -20  # View recent files

# Move completed investigations to archive
mv investigation-topic.md ~/Code/global-kb/investigations/

# Commit archive
cd ~/Code/global-kb
git add investigations/investigation-topic.md
git commit -m "archive: investigation-topic completed"
git push origin main

# Update Pepper reference
cd ~/memo
echo "[Archived to /global-kb/investigations/investigation-topic.md]" >> investigation-topic.md
git add investigation-topic.md
git commit -m "archive reference: investigation-topic"
git push origin main
```

### Quarterly (30-60 minutes)

```bash
# Review and consolidate
cd ~/Code/global-kb

# Create quarterly summary
cat > timelines/2026-q1-summary.md << 'EOF'
# Q1 2026 Summary

## Completed
- [Investigation 1]
- [Architecture decision 1]

## Patterns Documented
- [Pattern 1]

## Lessons Learned
- [Learning 1]
EOF

git add timelines/2026-q1-summary.md
git commit -m "add: Q1 2026 quarterly summary"
git tag -a "2026-q1" -m "Q1 2026 completion"
git push origin main
git push origin 2026-q1
```

---

## 12. Troubleshooting

### Issue: "MCP config not found"

**Solution**:
- macOS: Check `~/.cursor/mcp.json` exists
- WSL: Check `C:\Users\<username>\.cursor\mcp.json` exists
- **Action**: Restart Cursor after creating/editing

### Issue: "Rules not injecting"

**Solution**:
1. Verify file exists: `ls ~/Code/workspace/.cursor/rules`
2. Check YAML frontmatter has `alwaysApply: true`
3. Restart Cursor completely
4. Check Cursor output panel for errors

### Issue: "Pepper Memory Bank not accessible"

**Solution**:
1. Verify path in mcp.json matches your setup
2. Check directory exists: `ls -la ~/memo/global-memories/`
3. Verify file permissions: `chmod 755 ~/memo/`
4. **WSL only**: Ensure using `/home/username/`, not Windows path
5. Restart Cursor

### Issue: "Permission denied" on git operations

**Solution**:
```bash
# Check file permissions
ls -la ~/memo/
ls -la ~/Code/global-kb/

# Fix if needed
chmod 755 ~/memo
chmod 755 ~/Code/global-kb

# Try git operation again
cd ~/memo && git status
```

### Issue: "SSH key not found" when cloning

**Solution**:
```bash
# Verify SSH key exists
ls -la ~/.ssh/<key-name>

# Verify SSH config
cat ~/.ssh/config

# Test SSH connection
ssh -T git@github.com
# Should output: "Hi nfsarch33! You've successfully authenticated..."
```

### Issue: "Node.js/npx not found"

**macOS**:
```bash
brew install node@20
# Or verify path:
which node
```

**WSL Ubuntu**:
```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
```

### Issue: Green MCP indicator turns red

**Steps**:
1. Check Cursor output panel for error message
2. Verify mcp.json syntax: `jq . ~/.cursor/mcp.json` (macOS) or `Get-Content .cursor\mcp.json | ConvertFrom-Json` (Windows)
3. Verify all paths in mcp.json still exist
4. Check Node.js still accessible: `node --version`
5. Restart Cursor completely
6. If persists, temporarily remove problematic MCP server from config

---

## 13. Quick Reference Card

### Essential Paths

```
macOS:
  Memo:        ~/memo
  global-kb:   ~/Code/global-kb
  Workspace:   ~/Code/workspace
  MCP config:  ~/.cursor/mcp.json
  Rules:       ~/Code/workspace/.cursor/rules

WSL Ubuntu:
  Memo:        ~/memo
  global-kb:   ~/Code/global-kb
  Workspace:   ~/Code/workspace
  Rules:       ~/Code/workspace/.cursor/rules
  MCP config:  C:\Users\<user>\.cursor\mcp.json (Windows)
```

### Memory Decision Tree (One-liner)

```
Needed every chat? → Yes → Fits 1 para? → Yes → Cursor memory
                   ↓                      ↓
                   No → Pepper memory    No → Pepper memory
                   
Archive when done → global-kb/
```

### Essential Git Commands

```bash
# Add and commit changes
cd ~/memo && git add -A && git commit -m "update: description"

# Push to GitHub
git push origin main

# Pull from GitHub
git pull origin main

# View recent commits
git log --oneline | head -10

# Undo last commit (careful!)
git revert HEAD
```

### Health Check

**In Cursor**: Say "health check"

**Terminal**:
```bash
# Quick verification
echo "✓ Rules: $([ -f ~/Code/workspace/.cursor/rules ] && echo OK || echo MISSING)"
echo "✓ Pepper: $([ -d ~/memo/global-memories ] && echo OK || echo MISSING)"
echo "✓ KB: $([ -d ~/Code/global-kb ] && echo OK || echo MISSING)"
echo "✓ Node: $(node --version 2>/dev/null || echo MISSING)"
```

### MCP Status

- Green: All MCP servers working
- Red: One or more MCP servers down
- ⚠️ **Gray**: MCP not connected (rules-only fallback)

If green → System fully operational  
If red → Check errors in Cursor output  
If gray → Restart Cursor

---

## Setup Completion Checklist

- [ ] Prerequisites installed (Node.js, Git, uv)
- [ ] Directories created (memo, global-kb, workspace)
- [ ] Git repositories initialized
- [ ] Memory bank files created (sync-policy.md, error-patterns.md)
- [ ] MCP config created and customized
- [ ] Workspace rules created
- [ ] SSH configured (if using GitHub clone)
- [ ] Health check passes ("System: HEALTHY")
- [ ] Green MCP indicator in Cursor
- [ ] Able to say "health check" in Cursor successfully

---

## Total Setup Time

- **Quick Path** (GitHub clone): **5 minutes**
- **macOS Fresh Install**: **20 minutes**
- **WSL2 Fresh Install**: **25 minutes**
- **Initial memory creation**: **15 minutes**
- **Total to fully operational**: **40 minutes maximum**

---

## Support Resources

### Documentation Files
- `README.md` - System overview
- `SETUP.md` - Detailed setup guide  
- `ARCHITECTURE.md` - System design
- `MCP-SERVERS.md` - MCP reference
- `HEALTH-CHECK.md` - Diagnostics

### GitHub Repositories
- `cursor-memory-bank` - Pepper backup + setup docs
- `cursor-global-kb` - Knowledge base archive

---

**You now have everything needed for a production-ready, cross-platform Cursor AI Memory System with GitHub backup integration.**

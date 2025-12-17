# Coding Standards & Best Practices for Python, Go, MCP & AI

**Version**: 9.1  
**Date**: 2025-12-16  
**Purpose**: Daily programming standards for Python (uv/conda), Go, MCP servers, AI context, API/microservices, and academic research  
**Scope**: Production code, research code, API development, microservices  

---

## Table of Contents

1. [Python (uv + conda)](#1-python-uv--conda)
2. [Go (golang 1.25)](#2-go-golang-125)
3. [MCP Servers](#3-mcp-servers)
4. [AI Programming Conventions](#4-ai-programming-conventions)
5. [API & Microservices](#5-api--microservices)
6. [Academic Writing & Research Code](#6-academic-writing--research-code)
7. [Quality Gates Checklist](#7-quality-gates-checklist)

---

## 1. Python (uv + conda)

### Package Management Strategy

**Recommendation**: Use **uv** as primary, **conda** for system deps

```bash
# Setup with uv (10-100x faster than pip)
uv init myproject && cd myproject
uv sync                          # Create venv + install deps

# Conditional: Use conda for system-level deps only
conda create -n research python=3.11 cudatoolkit pytorch::pytorch pytorch::pytorch-cuda -c pytorch
source activate research
uv sync                          # Then uv for Python packages
```

**uv advantages**: Fast, reproducible, integrates with pyproject.toml  
**conda advantages**: Handles CUDA, ffmpeg, GDAL, non-Python binaries  
**Best practice**: Hybrid = conda (Python + system libs) + uv (Python packages)

### Project Structure

```
project/
├── pyproject.toml               # Python config (name, version, dependencies)
├── uv.lock                      # Reproducible lock file (commit to git)
├── README.md                    # Setup: uv sync && python -m src.main
├── src/projectname/
│   ├── __init__.py
│   ├── main.py                  # Entry point
│   └── core/
│       ├── __init__.py
│       ├── models.py            # ML models or domain models
│       ├── schemas.py           # Pydantic or dataclass schemas
│       └── service.py           # Business logic
├── tests/
│   ├── conftest.py              # pytest fixtures
│   ├── test_core.py
│   └── integration/
├── scripts/
│   ├── setup_data.py            # Data prep scripts
│   └── evaluate.py              # Evaluation scripts
└── docs/
    ├── conf.py                  # Sphinx config
    └── source/
```

### Code Standards

**Type Hints** (Required on public APIs)
```python
from typing import Optional

def process_items(
    items: list[str],
    timeout: int = 30,
    callback: Optional[Callable[[str], None]] = None,
) -> dict[str, int]:
    """
    Process items and return counts.
    
    Args:
        items: List of item names
        timeout: Max seconds per item
        callback: Optional progress callback
        
    Returns:
        Mapping of item name to count
        
    Raises:
        TimeoutError: If operation exceeds timeout
    """
    results = {}
    for item in items:
        results[item] = len(item)
    return results
```

**Error Handling** (Use custom exceptions)
```python
class DataValidationError(ValueError):
    """Raised when input data is invalid."""
    pass

class OperationError(Exception):
    """Raised on operation failure."""
    pass

try:
    validate(data)
except DataValidationError as e:
    logger.error(f"Validation failed: {e}", exc_info=True)
    raise OperationError(f"Cannot process: {e}") from e
except Exception:
    logger.exception("Unexpected error")
    raise
```

**Async Patterns** (Proper cancellation)
```python
import asyncio

async def work_with_timeout(duration: float):
    try:
        await asyncio.wait_for(long_operation(), timeout=duration)
    except asyncio.TimeoutError:
        logger.warning("Operation timed out")
    except asyncio.CancelledError:
        logger.info("Operation cancelled")
        raise
```

**Testing with pytest**
```python
# tests/test_core.py
import pytest
from src.core import process_items, DataValidationError

@pytest.fixture
def sample_data():
    return ["apple", "banana", "cherry"]

def test_process_items_success(sample_data):
    result = process_items(sample_data)
    assert len(result) == 3
    assert result["apple"] == 5

def test_process_items_validation_error():
    with pytest.raises(DataValidationError):
        process_items(None)  # Invalid input

@pytest.mark.asyncio
async def test_async_operation():
    result = await async_operation()
    assert result is not None
```

### Academic/Research Code

**Documentation with examples**
```python
"""Module for processing time-series data.

This module implements algorithms for anomaly detection in temporal sequences.
Designed for environmental monitoring applications.

Examples:
    >>> from src.signal_processing import detect_anomalies
    >>> data = [1.0, 1.1, 1.05, ..., 10.5]  # Spike at position 1000
    >>> anomalies = detect_anomalies(data, threshold=2.5)
    >>> len(anomalies)
    1
"""
```

**Experiment tracking**
```python
# Use Weights & Biases or MLflow for reproducibility
import wandb

wandb.init(project="my-research", config=config)
for epoch in range(10):
    loss = train_epoch()
    wandb.log({"loss": loss, "epoch": epoch})
    
# Always log: hyperparams, results, code version
```

**Reproducibility checklist**
- [ ] `uv.lock` committed (exact dependencies)
- [ ] Random seed set: `np.random.seed(42)`
- [ ] Dataset versioned (Git LFS or Zenodo DOI)
- [ ] Hyperparameters in config YAML
- [ ] Results logged (wandb, CSV, or results/)

---

## 2. Go (golang 1.25)

### Project Structure (Go Conventions)

```
project/
├── cmd/
│   ├── app1/main.go             # Executable 1
│   ├── app2/main.go             # Executable 2
│   └── server/main.go           # API server
├── internal/
│   └── pkg/
│       ├── domain/              # Business logic
│       │   ├── user.go
│       │   └── product.go
│       ├── repository/          # Data access
│       │   ├── user_repo.go
│       │   └── db.go
│       ├── service/             # Use cases
│       │   └── user_service.go
│       └── handler/             # HTTP handlers
│           └── user_handler.go
├── pkg/
│   └── public/                  # Public library
├── api/
│   ├── openapi.yaml             # API spec
│   └── proto/                   # Protocol buffers (if used)
├── tests/
│   └── integration_test.go      # Integration tests
├── go.mod                        # Dependency file
├── go.sum                        # Dependency lock file
└── Makefile
```

### Error Handling (Always Wrap Errors)

```go
package main

import (
    "fmt"
    "errors"
)

// Wrap errors with context
func readFile(path string) ([]byte, error) {
    data, err := ioutil.ReadFile(path)
    if err != nil {
        return nil, fmt.Errorf("read file: %w", err)  // Wrap with context
    }
    return data, nil
}

// Use errors.Is() and errors.As() for type checking
func handleError(err error) {
    if errors.Is(err, context.Canceled) {
        log.Printf("Operation cancelled")
    } else if errors.Is(err, context.DeadlineExceeded) {
        log.Printf("Operation timed out")
    } else {
        log.Printf("Error: %v", err)
    }
}

// Custom error types for more context
type ValidationError struct {
    Field   string
    Message string
}

func (e *ValidationError) Error() string {
    return fmt.Sprintf("validation error: %s=%s", e.Field, e.Message)
}
```

### Context Usage (First Parameter)

```go
// ALWAYS pass context.Context as first parameter
func Fetch(ctx context.Context, id string) (Data, error) {
    // Check context cancellation
    select {
    case <-ctx.Done():
        return nil, fmt.Errorf("fetch cancelled: %w", ctx.Err())
    default:
    }
    
    // Use context for timeouts
    req, _ := http.NewRequestWithContext(ctx, "GET", url, nil)
    resp, err := http.DefaultClient.Do(req)
    if err != nil {
        if ctx.Err() == context.DeadlineExceeded {
            return nil, fmt.Errorf("request timeout: %w", err)
        }
        return nil, fmt.Errorf("request failed: %w", err)
    }
    return parseResponse(resp), nil
}

// Use errgroup for concurrent operations with context
import "golang.org/x/sync/errgroup"

func ProcessConcurrently(ctx context.Context, items []Item) error {
    g, ctx := errgroup.WithContext(ctx)
    
    for _, item := range items {
        item := item  // Capture for closure
        g.Go(func() error {
            return ProcessItem(ctx, item)
        })
    }
    
    return g.Wait()  // Returns first error or nil
}
```

### Interfaces (Consumer-Defined)

```go
// Good: Define interfaces where you USE them, not where you implement
package main

// Consumer defines minimal interface
type Reader interface {
    Read(p []byte) (n int, err error)
}

// Provider implements more methods
type bufio.Reader struct { ... }

// This allows loose coupling and easy testing
type Service struct {
    reader Reader
}

func NewService(r Reader) *Service {
    return &Service{reader: r}
}
```

### Naming Conventions

```go
// Exported (uppercase) - can be used from other packages
func PublicFunc() {}
type PublicStruct struct {}
const MaxRetries = 3

// Unexported (lowercase) - internal only
func privateFunc() {}
type privateStruct struct {}
const defaultTimeout = 30

// Acronyms are uppercase: HTTPServer, not HttpServer
type HTTPServer struct {}
func (s *HTTPServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {}

// Receiver names: short, consistent (not 'this' or 'self')
func (s *Service) Process() {}
func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {}
```

### Table-Driven Tests

```go
func TestValidate(t *testing.T) {
    tests := []struct {
        name    string
        input   string
        wantErr bool
        wantVal int
    }{
        {"valid", "42", false, 42},
        {"invalid", "abc", true, 0},
        {"empty", "", true, 0},
        {"overflow", "999999999999", true, 0},
    }
    
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            val, err := Validate(tt.input)
            if (err != nil) != tt.wantErr {
                t.Errorf("Validate() error = %v, wantErr %v", err, tt.wantErr)
            }
            if val != tt.wantVal {
                t.Errorf("Validate() = %d, want %d", val, tt.wantVal)
            }
        })
    }
}
```

### Concurrency Best Practices

```go
// Use sync.WaitGroup for simple goroutine coordination
var wg sync.WaitGroup
for i := 0; i < 10; i++ {
    wg.Add(1)
    go func(i int) {
        defer wg.Done()
        Process(i)
    }(i)
}
wg.Wait()

// Use channels for communication
ch := make(chan Result, 10)  // Buffered channel
go func() {
    defer close(ch)
    ch <- Process()
}()
for r := range ch {
    log.Printf("Result: %+v", r)
}

// Timeout pattern
select {
case result := <-ch:
    log.Printf("Got result: %+v", result)
case <-time.After(5 * time.Second):
    log.Printf("Timeout")
}
```

---

## 3. MCP Servers

### Principles (Stateless, Versioned, Validated)

| Principle | Implementation |
|-----------|-----------------|
| **Stateless** | No side-effects in prompts; actions → tools |
| **Deterministic** | Same input → same output always |
| **Versioned** | Semantic versioning (1.0.0 → 1.1.0 → 2.0.0) |
| **Validated** | JSON Schema enforcement at boundary |
| **Secure** | OAuth 2.1, RBAC, input sanitization |

### Prompt Template Structure

```yaml
# prompts/summarizer.yaml
id: summarizer
version: 2.0.0
status: stable  # or "canary" (testing), "deprecated" (old)
owner: platform-team
description: "Summarize text with bullet points or narrative"

arguments:
  type: object
  required: [text]
  properties:
    text:
      type: string
      minLength: 50
      maxLength: 20000
      description: "Text to summarize"
    style:
      type: string
      enum: [bulleted, narrative]
      default: bulleted
    include_citations:
      type: boolean
      default: true
  additionalProperties: false  # No unknown args

messages:
  - role: system
    content: "You are a concise summarizer. Use the requested style."
  - role: user
    content: "Text to summarize:\n\n{{text}}"
```

### Deployment with Versioning

```yaml
# .github/workflows/deploy.yaml (CI/CD)

deployments:
  summarizer:
    versions:
      - version: 1.0.0
        status: deprecated
        sunset_date: 2025-12-31
      - version: 2.0.0
        status: stable
        traffic: 90%
      - version: 2.1.0-rc1
        status: canary
        traffic: 10%
        metrics:
          max_p95_latency_ms: 1200
          max_error_rate: 0.02
```

### Security Controls

```yaml
# security/scopes.yaml (RBAC)
scopes:
  prompts:
    summarizer:
      stable:
        allow: [read]
        min_auth: basic
      canary:
        allow: [read]
        min_auth: oauth
        required_roles: [beta_tester]
    custom:
      allow: [read, write, delete]
      min_auth: oauth
      required_roles: [admin]
```

### Observability (OpenTelemetry)

```python
# MCP server with OTel instrumentation
from opentelemetry import trace, metrics

tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)

tool_duration = meter.create_histogram("tool_duration_ms")
tool_errors = meter.create_counter("tool_errors")

def invoke_tool(tool_name: str, args: dict):
    with tracer.start_as_current_span(f"tool/{tool_name}") as span:
        span.set_attribute("tool.name", tool_name)
        start = time.time()
        try:
            result = tools[tool_name](**args)
            return result
        except Exception as e:
            span.set_attribute("error", True)
            tool_errors.add(1, {"tool": tool_name})
            raise
        finally:
            duration = (time.time() - start) * 1000
            tool_duration.record(duration, {"tool": tool_name})
```

---

## 4. AI Programming Conventions

### Context Engineering (3 Layers)

| Layer | Purpose | Example |
|-------|---------|---------|
| **Instructional** | Goals, constraints, format | "Output JSON. Be concise. Cite sources." |
| **Knowledge** | Domain facts, examples, patterns | Code snippets, API docs, data schema |
| **Tool** | Available functions, APIs | List of available tools and parameters |

### Memory Strategy (Auto-Injected Rules File)

```markdown
## Memory Quality Gates

BEFORE storing memory, ask:
1. Needed EVERY conversation? → Yes = Rules, No = Pepper
2. Fits 1 paragraph? → Yes = Rules, No = Pepper  
3. Rule/pattern, not task? → Yes = Memorize, No = Don't

## Memory Types

| Type | Storage | Frequency |
|------|---------|-----------|
| Pattern (error X → solution Y) | .cursor/rules | Every conversation |
| Procedure (10-step guide) | Pepper Memory | On-demand |
| Investigation (root cause analysis) | global-kb/archive | Manual |
```

### Prompt Optimization Checklist

```
SPECIFICITY:
- [ ] Use concrete examples (not "process data" → "sort CSV by timestamp")
- [ ] Define "success" explicitly (what does "good output" look like?)
- [ ] Avoid ambiguous terms

TASK SEGMENTATION:
- [ ] Break into subtasks if > 3 steps
- [ ] Number steps clearly
- [ ] Define expected output after each step

CONTEXT BUDGET:
- [ ] Estimate remaining context window
- [ ] Remove irrelevant details
- [ ] Prioritize recent/relevant info

ERROR MITIGATION:
- [ ] Ask for reasoning before conclusion
- [ ] Request citations for facts
- [ ] Explicit "if uncertain, say so"
```

### Context Window Management

```python
# Estimate tokens before sending to AI
def estimate_tokens(text: str) -> int:
    """Rough estimate: 1 token ≈ 4 characters"""
    return len(text) // 4

# Monitor context usage
max_context = 128000  # Claude 3.5 Sonnet
system_prompt_tokens = estimate_tokens(system_prompt)
conversation_tokens = sum(estimate_tokens(msg) for msg in history)
available = max_context - system_prompt_tokens - conversation_tokens

if available < 5000:
    # Summarize history or trim old messages
    history = summarize_old_messages(history)
```

---

## 5. API & Microservices

### Design-First with OpenAPI

```yaml
# api/openapi.yaml (Write BEFORE coding)
openapi: 3.1.0
info:
  title: User Service API
  version: 1.0.0
servers:
  - url: https://api.example.com/v1

paths:
  /users:
    get:
      summary: List users
      parameters:
        - name: limit
          in: query
          schema: { type: integer, default: 10 }
        - name: offset
          in: query
          schema: { type: integer, default: 0 }
      responses:
        '200':
          content:
            application/json:
              schema: { $ref: '#/components/schemas/UserList' }
        '401':
          description: Unauthorized

    post:
      summary: Create user
      requestBody:
        required: true
        content:
          application/json:
            schema: { $ref: '#/components/schemas/CreateUserRequest' }
      responses:
        '201':
          description: User created
          content:
            application/json:
              schema: { $ref: '#/components/schemas/User' }

components:
  schemas:
    User:
      type: object
      required: [id, name, email]
      properties:
        id: { type: string, format: uuid }
        name: { type: string }
        email: { type: string, format: email }
        created_at: { type: string, format: date-time }
```

### Microservices Patterns

**API Gateway Pattern**
```
Client → API Gateway (auth, rate limit, routing) → Service 1
                                                 → Service 2
```

**Circuit Breaker Pattern**
```
State: CLOSED (normal) → OPEN (fail fast) → HALF_OPEN (test) → CLOSED
Error threshold: 50% failure rate for 30s → OPEN
Recovery check: Every 60s, try 1 request → if OK, CLOSED
```

**Retry with Exponential Backoff**
```go
func RetryWithBackoff(operation func() error) error {
    var err error
    for attempt := 0; attempt < 5; attempt++ {
        err = operation()
        if err == nil {
            return nil
        }
        if attempt < 4 {
            delay := time.Duration(math.Pow(2, float64(attempt))+rand.Float64()) * time.Second
            time.Sleep(delay)
        }
    }
    return fmt.Errorf("operation failed after 5 attempts: %w", err)
}
```

### API Best Practices

```yaml
# Versioning in URL
GET /v1/users/{id}
GET /v2/users/{id}

# Request/Response format
Request headers:
  - X-Request-ID: uuid (for tracing)
  - Authorization: Bearer <token>

Response:
  {
    "data": {...},
    "meta": {
      "request_id": "uuid",
      "timestamp": "2025-12-16T15:30:00Z"
    },
    "errors": null
  }

# Pagination
GET /v1/users?limit=10&offset=0
Response:
  {
    "data": [...],
    "pagination": {
      "limit": 10,
      "offset": 0,
      "total": 100,
      "has_next": true
    }
  }

# Error responses
{
  "errors": [
    {
      "code": "INVALID_REQUEST",
      "message": "Email is required",
      "field": "email"
    }
  ],
  "meta": {
    "request_id": "uuid"
  }
}
```

---

## 6. Academic Writing & Research Code

### Reproducibility Standards

```
Project structure for publications:

paper/
├── paper.md                     # Main manuscript
├── figures/
│   ├── fig1_results.pdf
│   └── fig1_code.py            # Code that generates figure
├── code/
│   ├── src/                     # Research code
│   ├── pyproject.toml           # With exact version pins
│   ├── uv.lock                  # Locked dependencies
│   └── README.md               # Exact reproduction steps
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md               # Dataset citation, source
├── results/
│   ├── tables.csv              # Generated by reproduce script
│   └── metrics.json            # Model performance
└── reproduce.py                # Single command to regenerate all
```

### Citation Management

```bibtex
# Use consistent citation style (IEEE shown)
@article{author2024title,
  title={Research Title},
  author={Author, A. and Author, B.},
  journal={Journal Name},
  volume={10},
  number={2},
  pages={123--145},
  year={2024},
  doi={10.1234/example}
}

# In code:
# Based on methodology from [1]: https://doi.org/10.1234/example
# Implementation references appendix A.1
```

### Experiment Logging

```python
import json
from datetime import datetime

# Log all hyperparameters
config = {
    "model": "resnet50",
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100,
    "random_seed": 42,
    "dataset_version": "v2.1"  # Important!
}

# Log results
results = {
    "timestamp": datetime.now().isoformat(),
    "train_loss": 0.23,
    "val_accuracy": 0.87,
    "test_accuracy": 0.86,
    "hyperparams": config,
    "code_version": "3a7f2b1"  # Git commit hash
}

# Save for paper
with open("results/metrics.json", "w") as f:
    json.dump(results, f, indent=2)
```

---

## 7. Quality Gates Checklist

### All Languages/Projects

```
BEFORE COMMITTING:
- [ ] No TODO comments left
- [ ] No hardcoded credentials or API keys
- [ ] Error handling for all failure modes
- [ ] Unit tests passing
- [ ] Code review ready (clear names, <20 lines per function)
- [ ] Linting passing
- [ ] Dependencies reproducible (lock file committed)
```

### Python

```
- [ ] `uv sync` reproduces environment
- [ ] Type hints on all public APIs
- [ ] Docstrings (module, class, function)
- [ ] `ruff check .` and `mypy .` passing
- [ ] >= 80% test coverage
- [ ] No print() → use logger instead
```

### Go

```
- [ ] `go fmt` applied
- [ ] `go vet` passing
- [ ] Errors wrapped with context
- [ ] Context passed to all concurrent operations
- [ ] `golangci-lint run` passing (100+ linters)
- [ ] Table-driven tests for edge cases
```

### MCP Servers

```
- [ ] Prompts have JSON Schema validation
- [ ] Version number incremented (semver)
- [ ] Canary version deployed (<10% traffic)
- [ ] SLO metrics monitored (p95 latency, error rate)
- [ ] Security scopes enforced (RBAC)
- [ ] OpenTelemetry traces logged
```

### API/Microservices

```
- [ ] OpenAPI spec matches implementation
- [ ] All endpoints have request IDs for tracing
- [ ] Error responses include structured error codes
- [ ] Retry logic with exponential backoff
- [ ] Rate limiting configured
- [ ] Database queries have timeouts
```

### Academic/Research

```
- [ ] Dataset versioned and DOI provided
- [ ] Hyperparameters logged in results/
- [ ] Random seeds fixed (reproducibility)
- [ ] Code packaged with requirements.txt or uv.lock
- [ ] Figures generated by code (not manually edited)
- [ ] Citations complete with DOIs
```

---

## Reference: Command Checklists

### Python (uv)

```bash
# Setup
uv init myproject && cd myproject
uv add requests pydantic pytest

# Development
uv sync                          # Install from lock
uv run python -m src.main        # Run code
uv run pytest tests/             # Test
uv run ruff check .              # Lint
uv run mypy .                    # Type check
uv lock                          # Update lockfile

# Publishing
uv build                         # Create wheel
uv publish                       # To PyPI
```

### Go

```bash
# Setup
go mod init example.com/project
go get github.com/some/package

# Development
go run cmd/server/main.go        # Run
go test ./...                    # Test
go fmt ./...                     # Format
go vet ./...                     # Lint
golangci-lint run                # Full lint

# Building
go build -o bin/server ./cmd/server/main.go
go test -cover ./...             # Coverage
```

### Docker (MCP Server)

```dockerfile
# Dockerfile for MCP server
FROM python:3.11-slim

WORKDIR /app
RUN pip install uv

COPY pyproject.toml uv.lock ./
RUN uv sync

COPY src/ src/

EXPOSE 8000
CMD ["uv", "run", "python", "-m", "src.server"]
```

---

**This file is a living document.** Update based on team standards, framework updates (uv improvements, Go 1.26+, MCP spec changes).  
**Version**: 9.1 | **Date**: 2025-12-16

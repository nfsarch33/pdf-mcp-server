# GitHub Milestone & Issue Template: v0.7.0 Sprint

**Instructions:** Copy these templates into GitHub and create as Issues/Milestone for v0.7.0.

---

## MILESTONE: v0.7.0 - Cleanup & Stabilization

**Title:** v0.7.0 - Cleanup & Stabilization

**Description:**
```markdown
# v0.7.0: Cleanup & Stabilization

Target: February 14, 2026

Transition from consolidation (v0.6.0) to stability. Remove deprecated APIs, 
enhance error handling, optimize performance, improve developer experience.

## Goals
- ✅ Remove 5 deprecated API groups (25+ functions)
- ✅ Structured error handling with actionable guidance
- ✅ Performance optimizations (OCR cache, lazy imports)
- ✅ Enhanced documentation (OpenAPI schema, CLI tool explorer)
- ✅ 180+ tests passing, zero regressions

## Release Notes Preview
- **Removed:** 25+ deprecated functions (see MIGRATION_v0.7.0.md)
- **Added:** Structured error types, performance optimizations
- **Changed:** Error messages now include actionable suggestions
- **Docs:** Auto-generated OpenAPI schema, interactive CLI explorer

## Breaking Changes
See `MIGRATION_v0.7.0.md` for detailed migration guide and automated scripts.
```

**Due Date:** 2026-02-14

---

## ISSUE 1: Remove Text Manipulation Tools

**Title:** Remove deprecated text manipulation tools (insert/edit/remove_text)

**Labels:** `breaking-change`, `deprecation`, `v0.7.0`, `core-api`

**Description:**
```markdown
## Summary
Remove `insert_text()`, `edit_text()`, `remove_text()` functions.
These have been superseded by annotation-based APIs in v0.5.0+.

## Migration
- ❌ `insert_text(pdf_path, page, x, y, text)` 
- ✅ Use `add_text_annotation(pdf_path, page, x, y, text, ...)`

- ❌ `edit_text(pdf_path, page, text_id, new_text)`
- ✅ Use `update_text_annotation(pdf_path, page, annotation_id, new_text)`

- ❌ `remove_text(pdf_path, page, text_id)`
- ✅ Use `remove_text_annotation(pdf_path, page, annotation_id)`

## Acceptance Criteria
- [ ] All three functions removed from `pdf_tools.py`
- [ ] All docstring references updated
- [ ] Deprecation warnings removed from code
- [ ] Examples/docs updated to use annotation APIs
- [ ] Tests updated to use new APIs (~5-10 tests affected)
- [ ] 180 tests still passing

## Related
- Superseded by: v0.5.0 annotation APIs
- Part of: v0.7.0 Milestone
```

**Assignee:** @nfsarch33

**Estimate:** 2-3 hours

---

## ISSUE 2: Consolidate Extract Text Functions

**Title:** Consolidate extract_text_* functions into unified extract_text()

**Labels:** `breaking-change`, `deprecation`, `v0.7.0`, `core-api`

**Description:**
```markdown
## Summary
Remove `extract_text_native()`, `extract_text_ocr()`, `extract_text_smart()`, 
`extract_text_with_confidence()`. These are superseded by unified `extract_text(engine=..., confidence=...)` API.

## Deprecated Functions
- ❌ `extract_text_native(pdf_path, page)` 
  → ✅ `extract_text(pdf_path, page, engine='native')`

- ❌ `extract_text_ocr(pdf_path, page, lang='eng')`
  → ✅ `extract_text(pdf_path, page, engine='ocr', language='eng')`

- ❌ `extract_text_smart(pdf_path, page)`
  → ✅ `extract_text(pdf_path, page, engine='smart')`

- ❌ `extract_text_with_confidence(pdf_path, page)`
  → ✅ `extract_text(pdf_path, page, engine='auto', confidence=True)`

## Acceptance Criteria
- [ ] All 4 functions removed
- [ ] `extract_text()` supports all engine options
- [ ] `extract_text()` supports `confidence=True` parameter
- [ ] All old examples replaced with unified API usage
- [ ] 180 tests passing
- [ ] Benchmark shows no performance regression
```

**Assignee:** @nfsarch33

**Estimate:** 3-4 hours

---

## ISSUE 3: Unify Split PDF Functions

**Title:** Consolidate split_pdf_by_* functions into unified split_pdf()

**Labels:** `breaking-change`, `deprecation`, `v0.7.0`, `core-api`

**Description:**
```markdown
## Summary
Remove `split_pdf_by_bookmarks()`, `split_pdf_by_pages()`. 
Superseded by unified `split_pdf(mode='pages'|'bookmarks')` API.

## Deprecated Functions
- ❌ `split_pdf_by_bookmarks(pdf_path)` 
  → ✅ `split_pdf(pdf_path, mode='bookmarks')`

- ❌ `split_pdf_by_pages(pdf_path, pages_per_chunk)`
  → ✅ `split_pdf(pdf_path, mode='pages', chunk_size=pages_per_chunk)`

## Acceptance Criteria
- [ ] Both functions removed
- [ ] `split_pdf(mode='pages', chunk_size=...)` works
- [ ] `split_pdf(mode='bookmarks')` works
- [ ] All examples use unified API
- [ ] 180 tests passing
```

**Assignee:** @nfsarch33

**Estimate:** 2-3 hours

---

## ISSUE 4: Unify Export Functions

**Title:** Consolidate export_to_* functions into unified export_pdf()

**Labels:** `breaking-change`, `deprecation`, `v0.7.0`, `core-api`

**Description:**
```markdown
## Summary
Remove `export_to_markdown()`, `export_to_json()`.
Superseded by unified `export_pdf(format='markdown'|'json')` API.

## Deprecated Functions
- ❌ `export_to_markdown(pdf_path)` 
  → ✅ `export_pdf(pdf_path, format='markdown')`

- ❌ `export_to_json(pdf_path, include_metadata=False)`
  → ✅ `export_pdf(pdf_path, format='json', include_metadata=False)`

## Acceptance Criteria
- [ ] Both functions removed
- [ ] `export_pdf(format='markdown')` works
- [ ] `export_pdf(format='json', include_metadata=True)` works
- [ ] 180 tests passing
```

**Assignee:** @nfsarch33

**Estimate:** 2 hours

---

## ISSUE 5: Migrate Metadata Function

**Title:** Remove get_full_metadata() → migrate to get_pdf_metadata(full=True)

**Labels:** `breaking-change`, `deprecation`, `v0.7.0`, `core-api`

**Description:**
```markdown
## Summary
Remove `get_full_metadata()` function. 
Superseded by parameter: `get_pdf_metadata(full=True)`.

## Deprecated Function
- ❌ `get_full_metadata(pdf_path)` 
  → ✅ `get_pdf_metadata(pdf_path, full=True)`

## Acceptance Criteria
- [ ] Function removed
- [ ] `get_pdf_metadata(full=True)` works correctly
- [ ] 180 tests passing
```

**Assignee:** @nfsarch33

**Estimate:** 1-2 hours

---

## ISSUE 6: Add Structured Error Handling

**Title:** Add structured error types with actionable guidance

**Labels:** `enhancement`, `v0.7.0`, `developer-experience`

**Description:**
```markdown
## Summary
Improve error messages by adding structured error types with suggestion/actionable guidance.

## Error Types to Add
1. `PDFToolError` - Base exception
2. `FormNotFoundError` - Form structure error
3. `OCRError` - OCR processing failure
4. `SigningError` - Signature/certificate error
5. `ValidationError` - Input validation error
6. `FileFormatError` - Unsupported PDF format
7. `ResourceError` - File/memory resource error

## Implementation
1. Create new error class hierarchy in `exceptions.py`
2. Update all tool functions to raise structured errors
3. Add `__str__()` that formats user-friendly message with suggestion
4. Add tests for error conditions

## Acceptance Criteria
- [ ] All 7 error types defined
- [ ] All tools updated to use structured errors (~20 tools)
- [ ] Error messages include suggestions
- [ ] Tests verify error messages
- [ ] No functionality changed, only error quality improved
```

**Assignee:** @nfsarch33

**Estimate:** 4-5 hours

---

## ISSUE 7: Performance Optimizations

**Title:** Cache OCR language availability + lazy-load optional dependencies

**Labels:** `enhancement`, `v0.7.0`, `performance`

**Description:**
```markdown
## Summary
Improve performance by caching OCR language availability 
and lazy-loading optional dependencies (pyzbar, pytesseract).

## Implementation

### 1. Cache OCR Language Availability
Implement caching with TTL (1 hour)

### 2. Lazy-Load Optional Dependencies
Only import when needed in respective functions

## Benefits
- Faster initialization if OCR not used
- Reduced language file I/O
- Faster repeated OCR operations

## Acceptance Criteria
- [ ] OCR language cache implemented
- [ ] Optional dependencies lazy-loaded
- [ ] Performance benchmark shows 20-30% improvement for repeated OCR
- [ ] No functionality changes
- [ ] Tests verify caching works
```

**Assignee:** @nfsarch33

**Estimate:** 3-4 hours

---

## ISSUE 8: Generate OpenAPI/JSON Schema

**Title:** Auto-generate OpenAPI schema from docstrings

**Labels:** `documentation`, `v0.7.0`, `developer-experience`

**Description:**
```markdown
## Summary
Generate OpenAPI 3.0 / JSON Schema documentation from tool docstrings.

## Deliverables
1. **OpenAPI 3.0 Spec** (`docs/openapi.json`)
   - All 60 tools documented
   - Request/response schemas
   - Error responses
   - Auto-generated from docstrings

2. **Human-Readable Docs** (`docs/api-reference.md`)
   - Tool categories
   - Descriptions
   - Parameters with types
   - Return values
   - Example usage

## Tools
Consider: `pydantic-openapi-generator` or manual Markdown generation
Should be automated (CI hook) to stay in sync

## Acceptance Criteria
- [ ] OpenAPI spec generated and valid
- [ ] All 60 tools documented
- [ ] API reference markdown generated
- [ ] Included in CI/CD (auto-regenerate on push)
- [ ] Linked from README
```

**Assignee:** @nfsarch33

**Estimate:** 4-5 hours

---

## ISSUE 9: Interactive CLI Tool Explorer

**Title:** Add interactive CLI to explore and test tools

**Labels:** `feature`, `v0.7.0`, `developer-experience`

**Description:**
```markdown
## Summary
Create interactive CLI to list, describe, and test PDF tools.

## Commands

### List Tools
```bash
$ pdf-mcp-server list-tools
Total: 60 tools

[Form Tools] (6)
  - fill_form_fields: Fill PDF form fields with data
  - detect_form: Detect if PDF contains form
  ...
```

### Describe Tool
```bash
$ pdf-mcp-server tool extract_text
extract_text(pdf_path, page=None, engine='auto', language='eng', confidence=False)

Extract text from PDF with multiple engines.
```

### Test Tool (Interactive)
```bash
$ pdf-mcp-server test detect_form
Provide PDF path: /path/to/form.pdf
Result: {"has_form": true, ...}
```

## Acceptance Criteria
- [ ] `list-tools` command works
- [ ] `tool <name>` shows full description
- [ ] `test <tool>` interactive tester works
- [ ] Discoverable from README
```

**Assignee:** @nfsarch33

**Estimate:** 3-4 hours

---

## ISSUE 10: Write Comprehensive Migration Guide

**Title:** Create MIGRATION_v0.7.0.md with examples and scripts

**Labels:** `documentation`, `v0.7.0`, `breaking-change`

**Description:**
```markdown
## Summary
Create comprehensive migration guide for v0.6 → v0.7 transition.

## Deliverables

### MIGRATION_v0.7.0.md
- Overview of breaking changes
- Before/after code examples for each deprecated API
- Automated migration script (regex-based find-replace)
- Troubleshooting section
- FAQ

### Migration Script
```bash
$ python scripts/migrate_v0.6_to_v0.7.py --source=/path/to/code
```
Automatically updates:
- Function calls
- Import statements
- Comments/docstrings

## Acceptance Criteria
- [ ] Migration guide covers all 5 deprecated API groups
- [ ] Before/after examples for each
- [ ] Automated migration script works
- [ ] Linked from CHANGELOG.md
- [ ] Linked from README.md
```

**Assignee:** @nfsarch33

**Estimate:** 2-3 hours

---

## ISSUE 11: Automated Testing for Deprecation Warnings

**Title:** Add deprecation warning tests to CI

**Labels:** `testing`, `v0.7.0`

**Description:**
```markdown
## Summary
Ensure deprecation warnings fire correctly for all removed APIs.

## Implementation
```python
import warnings

def test_insert_text_deprecated():
    """Verify insert_text() raises deprecation warning"""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = insert_text("test.pdf", 0, 100, 100, "text")
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "use add_text_annotation" in str(w[0].message)
```

Add similar tests for all 25 deprecated functions.

## Acceptance Criteria
- [ ] Tests created for all deprecated functions
- [ ] CI runs tests and passes
- [ ] Warnings have helpful migration hints
```

**Assignee:** @nfsarch33

**Estimate:** 2-3 hours

---

## PULL REQUEST TEMPLATE

**Title:** `feature/v0.7.0-[component]` 

**Example:** `feature/v0.7.0-remove-text-apis`

**Description:**
```markdown
## Description
Removes deprecated text manipulation tools (insert_text, edit_text, remove_text)
in preparation for v0.7.0 release.

## Changes
- Removed `insert_text()` - use `add_text_annotation()`
- Removed `edit_text()` - use `update_text_annotation()`
- Removed `remove_text()` - use `remove_text_annotation()`
- Updated 8 tests to use annotation APIs
- Updated README and examples

## Testing
- ✅ All 180 tests passing
- ✅ No regressions detected
- ✅ Deprecation tests added

## Checklist
- [x] Code follows style guide
- [x] All tests passing
- [x] Documentation updated
- [x] No breaking changes to public APIs (only removals as planned)

## Related Issues
Closes #123
Part of: v0.7.0 Milestone
```

---

## GitHub Project Board Setup

Create a "v0.7.0 Sprint" project board with columns:

```
Backlog | Ready | In Progress | Review | Done
```

**Backlog Issues** (sorted by priority):
1. Remove deprecated APIs (Issues 1-5)
2. Structured error handling (Issue 6)
3. Performance optimizations (Issue 7)
4. Documentation generation (Issue 8)
5. CLI explorer (Issue 9)
6. Migration guide (Issue 10)
7. Deprecation warning tests (Issue 11)

**Auto-archiving:** Move to Done after PR merged + tests passing

---

## Success Criteria for v0.7.0

- ✅ **Zero deprecation warnings** in main branch
- ✅ **180 tests passing** (no regressions)
- ✅ **Migration guide** published and linked
- ✅ **Error messages** improved with suggestions
- ✅ **Performance** benchmarks show improvement
- ✅ **OpenAPI schema** generated
- ✅ **Release notes** clear about breaking changes
- ✅ **Zero critical bugs** reported in beta

---

## GitHub CLI Commands

```bash
# Create milestone
gh api repos/nfsarch33/pdf-mcp-server/milestones \
  -f title="v0.7.0 - Cleanup & Stabilization" \
  -f description="Remove deprecated APIs, enhance error handling, improve developer experience" \
  -f due_date="2026-02-14"

# Create issue
gh issue create \
  --title "Remove deprecated text manipulation tools" \
  --body "$(cat ISSUE_BODY.md)" \
  --label "breaking-change,deprecation,v0.7.0" \
  --milestone "v0.7.0"
```

---

**Ready to start v0.7.0 Sprint!**

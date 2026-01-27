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
